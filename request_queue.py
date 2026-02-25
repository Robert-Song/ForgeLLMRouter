"""
proxy_v6/request_queue.py
Async request batching request_queue with DP scheduling, LRU tracking,
and GPU memory budget enforcement.

Design:
  - Incoming requests are enqueued with their target model_id.
  - A short batch window accumulates requests.
  - The scheduler uses an LRU cache to track recently-loaded models.
  - DP (dynamic programming) computes the optimal loading order that
    minimizes total model-swap cost.  Since the number of distinct models
    in any batch window is small (n ≤ ~5), DP over subsets is fast.
  - Memory budget: before forwarding any request, the request_queue checks
    whether the target model fits in remaining GPU memory.  If not, it
    evicts the least-recently-used loaded models (via unload endpoint) 
    until enough memory is free.

"""

import asyncio
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Awaitable, Any, Set

import httpx

from config import (
    MAX_QUEUE_SIZE, BATCH_WINDOW_MS, LRU_CACHE_SIZE,
    TOTAL_USABLE_MEMORY_GB, GPU_HEADROOM_GB, NUM_GPUS,
)
from models import MODEL_REGISTRY, ModelInfo
from process_manager import process_manager


# Effective memory budget = total usable - headroom per GPU
MEMORY_BUDGET_GB = TOTAL_USABLE_MEMORY_GB - (GPU_HEADROOM_GB * NUM_GPUS)


@dataclass
class QueuedRequest:
    # A single request waiting to be forwarded.
    model_id: str
    request_body: dict
    api_key: str
    future: Optional[asyncio.Future] = None
    enqueued_at: float = field(default_factory=time.monotonic)


# GPU Memory Budget Tracker

class MemoryBudget:
    """
    Tracks which models are currently loaded and how much GPU memory
    they consume.
    Provides LRU-based eviction to free space.

    This is the proxy's estimate loaded models
    - We call touch() after a successful forward (model is loaded)
    - We call evict() after we tell llama-swap to unload a model
    """

    def __init__(self, budget_gb: float = MEMORY_BUDGET_GB):
        self._budget_gb = budget_gb
        # model_id → memory_gb, ordered by access time (LRU = first)
        self._loaded: OrderedDict[str, float] = OrderedDict()

    @property
    def used_gb(self) -> float:
        # Total GPU memory currently in use by loaded models.
        return sum(self._loaded.values())

    @property
    def free_gb(self) -> float:
        # Estimated free GPU memory.
        return self._budget_gb - self.used_gb

    @property
    def budget_gb(self) -> float:
        return self._budget_gb

    def is_loaded(self, model_id: str) -> bool:
        return model_id in self._loaded

    def touch(self, model_id: str):
        # Mark model as loaded / recently used.  Moves to end of LRU.
        model = MODEL_REGISTRY.get(model_id)
        mem = model.memory_gb if model else 0.0

        if model_id in self._loaded:
            self._loaded.move_to_end(model_id)
        else:
            self._loaded[model_id] = mem

    def evict(self, model_id: str):
        # Remove model from the loaded set (after unloading).
        self._loaded.pop(model_id, None)

    def fits(self, model_id: str) -> bool:
        # Check if model fits in current free memory.
        if self.is_loaded(model_id):
            return True  # already loaded, no extra memory needed
        model = MODEL_REGISTRY.get(model_id)
        needed = model.memory_gb if model else 50.0
        return needed <= self.free_gb

    def models_to_evict(self, model_id: str) -> List[str]:
        # Return list of model_ids to evict (LRU order) to make room
        # for the given model.  Returns empty list if it already fits.
        if self.is_loaded(model_id):
            return []

        model = MODEL_REGISTRY.get(model_id)
        needed = model.memory_gb if model else 50.0
        deficit = needed - self.free_gb

        if deficit <= 0:
            return []  # already fits

        # Check model's group — large_exclusive models need everything cleared
        if model and model.group == "large_exclusive":
            return list(self._loaded.keys())

        # Evict LRU models until we have enough space
        to_evict = []
        freed = 0.0
        for loaded_id, loaded_mem in self._loaded.items():
            if loaded_id == model_id:
                continue  # don't evict the model we're about to load
            to_evict.append(loaded_id)
            freed += loaded_mem
            if freed >= deficit:
                break

        if freed < deficit:
            # Can't free enough memory even by evicting everything.
            # Still return what we can — llama-swap will handle the rest
            # or it will OOM and we'll get a 503.
            print(
                f"[MemoryBudget] WARNING: model '{model_id}' needs "
                f"{needed:.1f} GB but only {self.free_gb + freed:.1f} GB "
                f"can be freed (budget: {self._budget_gb:.1f} GB)"
            )
            # FORCE EVICTION: even if we are over budget, we should evict what we can. 
            # We must not give up and drop the request. The backend might still fit it (e.g., shared layers, smaller context)
            return to_evict

        return to_evict

    def swap_cost(self, model_id: str) -> float:
        # Estimate cost of loading this model (for DP scheduler).
        # 0 if already loaded, else memory_gb (proxy for load time).
        if self.is_loaded(model_id):
            return 0.0
        model = MODEL_REGISTRY.get(model_id)
        return model.memory_gb if model else 50.0

    def status(self) -> str:
        # Human-readable status string.
        loaded_strs = [
            f"  {mid}: {mem:.1f} GB" for mid, mem in self._loaded.items()
        ]
        return (
            f"[MemoryBudget] {self.used_gb:.1f}/{self._budget_gb:.1f} GB used "
            f"({self.free_gb:.1f} GB free)\n"
            + ("\n".join(loaded_strs) if loaded_strs else "  (no models loaded)")
        )


#  Request request_queue with Memory Budget
class RequestQueue:
    """
    Async request_queue that batches incoming requests and schedules them
    in an order that minimizes model swaps, while enforcing a
    GPU memory budget via LRU eviction.
    """

    def __init__(self):
        self._queue: Optional[asyncio.Queue[QueuedRequest]] = None
        self._memory = MemoryBudget()
        self._batch_window_s = BATCH_WINDOW_MS / 1000.0
        self._running = False
        self._processor_task: Optional[asyncio.Task] = None
        self._forward_fn: Optional[Callable[..., Awaitable[Any]]] = None

    def set_forward_fn(self, fn: Callable[..., Awaitable[Any]]):
        # Set the async function that actually forwards a request over HTTP.
        # Signature: async def forward(model_id, port, request_body, api_key) -> response_data
        self._forward_fn = fn

    async def start(self):
        # Start the background batch processor.
        if self._running:
            return
        self._queue = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
        self._running = True
        self._processor_task = asyncio.create_task(self._process_loop())

    async def stop(self):
        # Stop the batch processor.
        self._running = False
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass

    async def enqueue(self, model_id: str, request_body: dict, api_key: str) -> Any:
        # Add a request to the request_queue and wait for its result.
        # Returns the response data from the forward function.
        req = QueuedRequest(
            model_id=model_id,
            request_body=request_body,
            api_key=api_key,
            future=asyncio.get_running_loop().create_future(),
        )
        if self._queue is None:
            raise RuntimeError("request_queue not started")
        await self._queue.put(req)
        return await req.future

    async def _process_loop(self):
        # Main loop: collect a batch of requests, schedule them optimally,
        # then forward each to llama-swap (with memory eviction).
        while self._running:
            try:
                # Wait for the first request
                first = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                batch = [first]

                # Collect more requests within the batch window
                deadline = time.monotonic() + self._batch_window_s
                while time.monotonic() < deadline:
                    try:
                        remaining = max(0.001, deadline - time.monotonic())
                        req = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                        batch.append(req)
                    except asyncio.TimeoutError:
                        break

                # Schedule and forward
                ordered = self._schedule_batch(batch)
                for req in ordered:
                    await self._ensure_memory_and_forward(req)

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                # Don't let the processor die on transient errors
                print(f"[request_queue] Error in process loop: {e}")
                await asyncio.sleep(0.1)

    def _schedule_batch(self, batch: List[QueuedRequest]) -> List[QueuedRequest]:
        # Given a batch of requests, order them to minimize total swap cost.
        # Uses DP over the set of distinct models (bitmask DP since n is small).
        # Requests for the same model are grouped together.
        
        # Group requests by model
        model_groups: Dict[str, List[QueuedRequest]] = {}
        for req in batch:
            model_groups.setdefault(req.model_id, []).append(req)

        models = list(model_groups.keys())
        n = len(models)

        if n <= 1:
            return batch  # no scheduling needed

        # DP: find optimal ordering of n models to minimize swap cost
        # State: bitmask of visited models, last model index
        # Cost: swap_cost from last model to next model
        INF = float('inf')
        n_states = 1 << n

        # dp[mask][last] = min total cost to visit models in `mask`, ending at `last`
        dp = [[INF] * n for _ in range(n_states)]
        parent = [[-1] * n for _ in range(n_states)]

        # Initialize: starting from each model
        for i in range(n):
            dp[1 << i][i] = self._memory.swap_cost(models[i])

        # Fill DP table
        for mask in range(n_states):
            for last in range(n):
                if dp[mask][last] == INF:
                    continue
                for nxt in range(n):
                    if mask & (1 << nxt):
                        continue  # already visited
                    new_mask = mask | (1 << nxt)
                    cost = dp[mask][last] + self._memory.swap_cost(models[nxt])
                    if cost < dp[new_mask][nxt]:
                        dp[new_mask][nxt] = cost
                        parent[new_mask][nxt] = last

        # Find best end state
        full_mask = n_states - 1
        best_last = min(range(n), key=lambda i: dp[full_mask][i])

        # Reconstruct path
        path = []
        mask = full_mask
        last = best_last
        while last != -1:
            path.append(last)
            prev = parent[mask][last]
            mask ^= (1 << last)
            last = prev
        path.reverse()

        # Build ordered result: for each model in optimal order, emit all its requests
        ordered = []
        for idx in path:
            ordered.extend(model_groups[models[idx]])

        return ordered

    #  Memory-aware forwarding
    async def ensure_model_loaded(self, model_id: str) -> int:
        # Ensure model fits in memory (evicting if necessary), load it via process manager, 
        # and touch memory tracker. Returns the allocated port.
        if not self._memory.fits(model_id):
            to_evict = self._memory.models_to_evict(model_id)
            if to_evict:
                print(f"[request_queue] Evicting models {to_evict} to fit '{model_id}'")
                for evict_id in to_evict:
                    from process_manager import process_manager
                    await process_manager.unload_model(evict_id)
                    self._memory.evict(evict_id)
                    
        model_info = MODEL_REGISTRY.get(model_id)
        if not model_info:
            raise ValueError(f"Model '{model_id}' not found in registry")
            
        from process_manager import process_manager
        await process_manager.load_model(model_id, model_info)
        self._memory.touch(model_id)
        return process_manager.model_ports[model_id]

    async def _ensure_memory_and_forward(self, req: QueuedRequest):
        # Ensure enough GPU memory for the requested model, evict if needed,
        # spawn the model via process_manager, then forward the request.
        try:
            if self._forward_fn is None:
                raise RuntimeError("No forward function set on RequestQueue")

            # Step 1 & Step 2: Ensure loaded and get port
            port = await self.ensure_model_loaded(req.model_id)

            # Step 3: forward the request to the designated port
            result = await self._forward_fn(req.model_id, port, req.request_body, req.api_key)

            if not req.future.done():
                req.future.set_result(result)

        except Exception as e:
            if not req.future.done():
                req.future.set_exception(e)


# Module-level singleton
request_queue = RequestQueue()
