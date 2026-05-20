"""
proxy_v6/request_queue.py
Async request queue with model-swap scheduling, llama.cpp slot accounting,
and GPU memory budget enforcement.
"""

import asyncio
import time
from collections import Counter, OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

import httpx

from config import (
    BATCH_WINDOW_MS,
    GPU_HEADROOM_GB,
    MAX_QUEUE_SIZE,
    NUM_GPUS,
    TOTAL_USABLE_MEMORY_GB,
    UNLOAD_IDLE_MODELS_BEFORE_LOAD,
)
from model_resources import (
    effective_model_memory_gb,
    effective_parallel_slots,
    estimate_base_memory_gb,
    estimate_slot_memory_gb,
)
from models import MODEL_REGISTRY
from process_manager import process_manager


MEMORY_BUDGET_GB = TOTAL_USABLE_MEMORY_GB - (GPU_HEADROOM_GB * NUM_GPUS)


@dataclass
class QueuedRequest:
    model_id: str
    request_body: dict
    api_key: str
    request_id: int
    endpoint_type: str = "chat"
    stream: bool = False
    future: Optional[asyncio.Future] = None
    enqueued_at: float = field(default_factory=time.monotonic)


@dataclass
class ModelRuntimeState:
    model_id: str
    parallel_slots: int
    semaphore: asyncio.Semaphore
    load_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    inflight: int = 0
    loading: bool = False


class MemoryBudget:
    def __init__(self, budget_gb: float = MEMORY_BUDGET_GB):
        self._budget_gb = budget_gb
        self._loaded: OrderedDict[str, float] = OrderedDict()

    @property
    def used_gb(self) -> float:
        return sum(self._loaded.values())

    @property
    def free_gb(self) -> float:
        return self._budget_gb - self.used_gb

    @property
    def budget_gb(self) -> float:
        return self._budget_gb

    def model_memory_gb(self, model_id: str) -> float:
        model = MODEL_REGISTRY.get(model_id)
        if not model:
            return 50.0

        return effective_model_memory_gb(model, budget_gb=self._budget_gb)

    def base_memory_gb(self, model_id: str) -> float:
        model = MODEL_REGISTRY.get(model_id)
        return estimate_base_memory_gb(model) if model else 0.0

    def slot_memory_gb(self, model_id: str) -> float:
        model = MODEL_REGISTRY.get(model_id)
        return estimate_slot_memory_gb(model) if model else 0.0

    def is_loaded(self, model_id: str) -> bool:
        return model_id in self._loaded

    def touch(self, model_id: str):
        mem = self.model_memory_gb(model_id)
        if model_id in self._loaded:
            self._loaded[model_id] = mem
            self._loaded.move_to_end(model_id)
        else:
            self._loaded[model_id] = mem

    def evict(self, model_id: str):
        self._loaded.pop(model_id, None)

    def loaded_model_ids(self) -> List[str]:
        return list(self._loaded.keys())

    def fits(self, model_id: str) -> bool:
        if self.is_loaded(model_id):
            return True
        return self.model_memory_gb(model_id) <= self.free_gb

    def models_to_evict(self, model_id: str, protected: set[str]) -> List[str]:
        if self.is_loaded(model_id):
            return []

        needed = self.model_memory_gb(model_id)
        deficit = needed - self.free_gb
        if deficit <= 0:
            return []

        model = MODEL_REGISTRY.get(model_id)
        candidates = [
            loaded_id for loaded_id in self._loaded.keys()
            if loaded_id != model_id and loaded_id not in protected
        ]
        if model and model.group == "large_exclusive":
            return candidates

        to_evict = []
        freed = 0.0
        for loaded_id in candidates:
            loaded_mem = self._loaded[loaded_id]
            to_evict.append(loaded_id)
            freed += loaded_mem
            if freed >= deficit:
                break

        return to_evict if freed >= deficit else []

    def swap_cost(self, model_id: str) -> float:
        if self.is_loaded(model_id):
            return 0.0
        return self.model_memory_gb(model_id)

    def snapshot(self) -> dict:
        return {
            "budget_gb": self._budget_gb,
            "used_gb": self.used_gb,
            "free_gb": self.free_gb,
            "loaded": [
                {"model_id": model_id, "effective_memory_gb": memory_gb}
                for model_id, memory_gb in self._loaded.items()
            ],
            "lru_order": list(self._loaded.keys()),
        }


class RequestQueue:
    def __init__(self):
        self._queue: Optional[asyncio.Queue[QueuedRequest]] = None
        self._memory = MemoryBudget()
        self._batch_window_s = BATCH_WINDOW_MS / 1000.0
        self._running = False
        self._processor_task: Optional[asyncio.Task] = None
        self._forward_fn: Optional[Callable[..., Awaitable[Any]]] = None
        self._runtimes: Dict[str, ModelRuntimeState] = {}
        self._memory_condition = asyncio.Condition()
        self._state_lock = asyncio.Lock()
        self._request_seq = 0
        self._request_states: OrderedDict[int, dict] = OrderedDict()

    def set_forward_fn(self, fn: Callable[..., Awaitable[Any]]):
        self._forward_fn = fn

    async def start(self):
        if self._running:
            return
        self._queue = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
        self._running = True
        self._processor_task = asyncio.create_task(self._process_loop())

    async def stop(self):
        self._running = False
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass

    async def enqueue(self, model_id: str, request_body: dict, api_key: str) -> Any:
        request_id = await self._next_request_id()
        req = QueuedRequest(
            model_id=model_id,
            request_body=request_body,
            api_key=api_key,
            request_id=request_id,
            future=asyncio.get_running_loop().create_future(),
        )
        await self._set_request_state(req, "queued")
        if self._queue is None:
            await self._remove_request_state(request_id)
            raise RuntimeError("request_queue not started")
        await self._queue.put(req)
        return await req.future

    async def _next_request_id(self) -> int:
        async with self._state_lock:
            self._request_seq += 1
            return self._request_seq

    async def _set_request_state(self, req: QueuedRequest, state: str):
        async with self._state_lock:
            self._request_states[req.request_id] = {
                "request_id": req.request_id,
                "model_id": req.model_id,
                "endpoint_type": req.endpoint_type,
                "stream": req.stream,
                "state": state,
                "enqueued_at": req.enqueued_at,
                "wait_s": max(0.0, time.monotonic() - req.enqueued_at),
            }

    async def _remove_request_state(self, request_id: int):
        async with self._state_lock:
            self._request_states.pop(request_id, None)

    def _runtime_for(self, model_id: str) -> ModelRuntimeState:
        model = MODEL_REGISTRY.get(model_id)
        slots = effective_parallel_slots(model) if model else 1
        state = self._runtimes.get(model_id)
        if state is None or state.parallel_slots != slots:
            state = ModelRuntimeState(
                model_id=model_id,
                parallel_slots=slots,
                semaphore=asyncio.Semaphore(slots),
            )
            self._runtimes[model_id] = state
        return state

    def _busy_model_ids(self) -> set[str]:
        return {
            model_id for model_id, state in self._runtimes.items()
            if state.inflight > 0 or state.loading
        }

    async def _process_loop(self):
        while self._running:
            try:
                first = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                batch = [first]

                deadline = time.monotonic() + self._batch_window_s
                while time.monotonic() < deadline:
                    try:
                        remaining = max(0.001, deadline - time.monotonic())
                        batch.append(await asyncio.wait_for(self._queue.get(), timeout=remaining))
                    except asyncio.TimeoutError:
                        break

                ordered = self._schedule_batch(batch)
                for group in self._group_ordered_requests(ordered):
                    tasks = [
                        asyncio.create_task(self._ensure_memory_and_forward(req))
                        for req in group
                    ]
                    await asyncio.gather(*tasks, return_exceptions=True)

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[request_queue] Error in process loop: {e}")
                await asyncio.sleep(0.1)

    def _group_ordered_requests(self, ordered: List[QueuedRequest]) -> List[List[QueuedRequest]]:
        groups: List[List[QueuedRequest]] = []
        for req in ordered:
            if not groups or groups[-1][0].model_id != req.model_id:
                groups.append([req])
            else:
                groups[-1].append(req)
        return groups

    def _schedule_batch(self, batch: List[QueuedRequest]) -> List[QueuedRequest]:
        model_groups: Dict[str, List[QueuedRequest]] = {}
        for req in batch:
            model_groups.setdefault(req.model_id, []).append(req)

        models = list(model_groups.keys())
        n = len(models)
        if n <= 1:
            return batch

        inf = float("inf")
        n_states = 1 << n
        dp = [[inf] * n for _ in range(n_states)]
        parent = [[-1] * n for _ in range(n_states)]

        for i in range(n):
            dp[1 << i][i] = self._memory.swap_cost(models[i])

        for mask in range(n_states):
            for last in range(n):
                if dp[mask][last] == inf:
                    continue
                for nxt in range(n):
                    if mask & (1 << nxt):
                        continue
                    new_mask = mask | (1 << nxt)
                    cost = dp[mask][last] + self._memory.swap_cost(models[nxt])
                    if cost < dp[new_mask][nxt]:
                        dp[new_mask][nxt] = cost
                        parent[new_mask][nxt] = last

        full_mask = n_states - 1
        best_last = min(range(n), key=lambda i: dp[full_mask][i])

        path = []
        mask = full_mask
        last = best_last
        while last != -1:
            path.append(last)
            prev = parent[mask][last]
            mask ^= (1 << last)
            last = prev
        path.reverse()

        ordered = []
        for idx in path:
            ordered.extend(model_groups[models[idx]])
        return ordered

    async def ensure_model_loaded(self, model_id: str) -> int:
        model_info = MODEL_REGISTRY.get(model_id)
        if not model_info:
            raise ValueError(f"Model '{model_id}' not found in registry")
        parallel_slots = effective_parallel_slots(model_info, budget_gb=self._memory.budget_gb)
        slot_memory_gb = estimate_slot_memory_gb(model_info)
        if parallel_slots > 1 and slot_memory_gb <= 0:
            raise RuntimeError(
                f"Model '{model_id}' has parallel_slots={parallel_slots} but no "
                "positive slot memory estimate. Set ctx_size/extra_args --ctx-size "
                "and ensure the local GGUF metadata is readable, or set slot_memory_gb."
            )

        runtime = self._runtime_for(model_id)
        async with runtime.load_lock:
            async with self._memory_condition:
                if model_id in process_manager.active_processes:
                    self._memory.touch(model_id)
                    self._memory_condition.notify_all()
                    return process_manager.model_ports[model_id]

                needed = self._memory.model_memory_gb(model_id)
                if needed > self._memory.budget_gb:
                    raise RuntimeError(
                        f"Model '{model_id}' needs {needed:.1f} GB, "
                        f"exceeding memory budget {self._memory.budget_gb:.1f} GB"
                    )

                if UNLOAD_IDLE_MODELS_BEFORE_LOAD:
                    while True:
                        protected = self._busy_model_ids()
                        other_loaded = [
                            loaded_id
                            for loaded_id in self._memory.loaded_model_ids()
                            if loaded_id != model_id
                        ]
                        to_evict = [
                            loaded_id
                            for loaded_id in other_loaded
                            if loaded_id not in protected
                        ]
                        if to_evict:
                            print(
                                f"[request_queue] Unloading idle models {to_evict} "
                                f"before loading '{model_id}'"
                            )
                            for evict_id in to_evict:
                                await process_manager.unload_model(evict_id)
                                self._memory.evict(evict_id)
                            continue
                        if other_loaded:
                            await self._memory_condition.wait()
                            continue
                        break

                while not self._memory.fits(model_id):
                    to_evict = self._memory.models_to_evict(model_id, self._busy_model_ids())
                    if to_evict:
                        print(f"[request_queue] Evicting models {to_evict} to fit '{model_id}'")
                        for evict_id in to_evict:
                            await process_manager.unload_model(evict_id)
                            self._memory.evict(evict_id)
                        continue
                    await self._memory_condition.wait()

                self._memory.touch(model_id)
                runtime.loading = True

            load_succeeded = False
            try:
                await process_manager.load_model(model_id, model_info)
                load_succeeded = True
            finally:
                async with self._memory_condition:
                    runtime.loading = False
                    if load_succeeded:
                        self._memory.touch(model_id)
                    else:
                        self._memory.evict(model_id)
                    self._memory_condition.notify_all()

            return process_manager.model_ports[model_id]

    @asynccontextmanager
    async def model_slot(
        self,
        model_id: str,
        endpoint_type: str,
        stream: bool = False,
        request_id: Optional[int] = None,
    ):
        created_record = request_id is None
        if request_id is None:
            request_id = await self._next_request_id()

        req = QueuedRequest(
            model_id=model_id,
            request_body={},
            api_key="",
            request_id=request_id,
            endpoint_type=endpoint_type,
            stream=stream,
        )

        runtime = self._runtime_for(model_id)
        acquired = False
        try:
            await self._set_request_state(req, "waiting_slot")
            await runtime.semaphore.acquire()
            acquired = True
            runtime.inflight += 1
            await self._set_request_state(req, "waiting_model")
            port = await self.ensure_model_loaded(model_id)
            await self._set_request_state(req, "inflight")
            yield port
        finally:
            if acquired:
                runtime.inflight = max(0, runtime.inflight - 1)
                runtime.semaphore.release()
                async with self._memory_condition:
                    if model_id in process_manager.active_processes:
                        self._memory.touch(model_id)
                    self._memory_condition.notify_all()
            if created_record or request_id is not None:
                await self._remove_request_state(request_id)

    async def _ensure_memory_and_forward(self, req: QueuedRequest):
        try:
            if self._forward_fn is None:
                raise RuntimeError("No forward function set on RequestQueue")

            async with self.model_slot(
                req.model_id,
                endpoint_type=req.endpoint_type,
                stream=req.stream,
                request_id=req.request_id,
            ) as port:
                result = await self._forward_fn(req.model_id, port, req.request_body, req.api_key)

            if not req.future.done():
                req.future.set_result(result)

        except Exception as e:
            if not req.future.done():
                req.future.set_exception(e)

    async def unload_model(self, model_id: str, force: bool = False):
        runtime = self._runtime_for(model_id)
        if runtime.inflight > 0 and not force:
            raise RuntimeError(f"Model '{model_id}' has {runtime.inflight} inflight request(s)")

        async with self._memory_condition:
            await process_manager.unload_model(model_id)
            self._memory.evict(model_id)
            self._memory_condition.notify_all()

    async def unload_all(self, force: bool = False) -> list[str]:
        loaded_models = list(process_manager.active_processes.keys())
        unloaded = []
        for model_id in loaded_models:
            await self.unload_model(model_id, force=force)
            unloaded.append(model_id)
        return unloaded

    async def snapshot(self, include_backend: bool = True) -> dict:
        now = time.monotonic()
        async with self._state_lock:
            requests = []
            for state in self._request_states.values():
                item = dict(state)
                item["wait_s"] = max(0.0, now - item["enqueued_at"])
                requests.append(item)

        queued_counts = Counter(
            item["model_id"] for item in requests
            if item["state"] in {"queued", "waiting_model", "waiting_slot"}
        )
        inflight_counts = {
            model_id: state.inflight
            for model_id, state in self._runtimes.items()
            if state.inflight > 0
        }

        async with self._memory_condition:
            memory = self._memory.snapshot()

        loaded_models = []
        for model_id, process in process_manager.active_processes.items():
            model = MODEL_REGISTRY.get(model_id)
            runtime = self._runtime_for(model_id)
            port = process_manager.model_ports.get(model_id)
            loaded = {
                "model_id": model_id,
                "backend": model.backend if model else None,
                "port": port,
                "pid": process.pid,
                "returncode": process.poll(),
                "parallel_slots": runtime.parallel_slots,
                "inflight": runtime.inflight,
                "base_memory_gb": self._memory.base_memory_gb(model_id),
                "slot_memory_gb": self._memory.slot_memory_gb(model_id),
                "effective_memory_gb": self._memory.model_memory_gb(model_id),
            }
            if include_backend and model and model.backend == "llamacpp" and port:
                loaded["llamacpp"] = await self._llamacpp_snapshot(port)
            loaded_models.append(loaded)

        return {
            "queue": {
                "max_size": MAX_QUEUE_SIZE,
                "depth": self._queue.qsize() if self._queue else 0,
                "queued_by_model": dict(queued_counts),
                "inflight_by_model": inflight_counts,
                "requests": requests,
            },
            "memory": memory,
            "loaded_models": loaded_models,
        }

    async def _llamacpp_snapshot(self, port: int) -> dict:
        snapshot = {}
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=2.0) as client:
            for name, path in {"props": "/props", "slots": "/slots"}.items():
                try:
                    response = await client.get(path)
                    snapshot[name] = response.json() if response.status_code == 200 else {
                        "error": f"HTTP {response.status_code}",
                        "body": response.text[:200],
                    }
                except Exception as exc:
                    snapshot[name] = {"error": str(exc)}
        return snapshot


request_queue = RequestQueue()
