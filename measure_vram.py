"""
Measure llama.cpp model VRAM and cache results in model_info.db.

The measurement runs the target model by itself with --parallel 1 and then
--parallel 2. The first run measures base + one slot; the delta between the
second and first run estimates per-additional-slot VRAM.
"""

import argparse
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from typing import Optional

import httpx


_LLAMACPP_CONTEXT_ALIASES = {
    "-c",
    "--ctx-size",
    "--ctx_size",
}

_LLAMACPP_PARALLEL_ALIASES = {
    "-np",
    "--parallel",
}

_LLAMACPP_CONT_BATCHING_ALIASES = {
    "-cb",
    "-nocb",
    "--cont-batching",
    "--no-cont-batching",
}

def _configure_import_path(config_dir: Optional[str]):
    if config_dir:
        resolved_dir = os.path.abspath(os.path.expanduser(config_dir))
        config_path = os.path.join(resolved_dir, "config.py")
        if not os.path.isfile(config_path):
            raise SystemExit(f"--config-dir must contain config.py: {resolved_dir}")
        sys.path.insert(0, resolved_dir)


def _pop_flag_value(args: list[str], aliases: set[str]) -> tuple[list[str], Optional[str]]:
    filtered = []
    value = None
    i = 0
    while i < len(args):
        arg = args[i]
        flag, sep, inline_value = arg.partition("=")

        if flag in aliases:
            if sep:
                value = inline_value
            elif i + 1 < len(args):
                i += 1
                value = args[i]
            i += 1
            continue

        filtered.append(arg)
        i += 1

    return filtered, value


def _drop_flags(args: list[str], aliases: set[str]) -> list[str]:
    return [arg for arg in args if arg.partition("=")[0] not in aliases]


def _parse_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _allocate_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _gpu_process_memory_mb() -> dict[int, int]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )

    memory: dict[int, int] = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
            used_mb = int(parts[1])
        except ValueError:
            continue
        memory[pid] = memory.get(pid, 0) + used_mb
    return memory


def _process_tree() -> dict[int, dict]:
    output = subprocess.check_output(["ps", "-e", "-o", "pid,ppid,args"], text=True)
    processes = {}
    for line in output.splitlines()[1:]:
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
        except ValueError:
            continue
        processes[pid] = {"ppid": ppid, "args": parts[2]}
    return processes


def _descendant_pids(root_pid: int) -> set[int]:
    processes = _process_tree()
    pids = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, info in processes.items():
            if pid not in pids and info["ppid"] in pids:
                pids.add(pid)
                changed = True
    return pids


def _process_vram_gb(root_pid: int) -> float:
    gpu_memory = _gpu_process_memory_mb()
    pids = _descendant_pids(root_pid)
    return sum(gpu_memory.get(pid, 0) for pid in pids) / 1024.0


def _terminate_process(process: subprocess.Popen, timeout: float = 10.0):
    if process.poll() is not None:
        return

    try:
        if hasattr(os, "killpg"):
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        else:
            process.terminate()
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        if hasattr(os, "killpg"):
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        else:
            process.kill()
        process.wait(timeout=5)
    except ProcessLookupError:
        pass


def _wait_for_health(port: int, process: subprocess.Popen, timeout: float = 1800.0):
    deadline = time.time() + timeout
    url = f"http://127.0.0.1:{port}/v1/models"
    with httpx.Client(timeout=2.0) as client:
        while time.time() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"llama-server exited with code {process.poll()}")
            try:
                response = client.get(url)
                if response.status_code == 200:
                    return
            except httpx.RequestError:
                pass
            time.sleep(2)
    raise TimeoutError(f"Timed out waiting for llama-server on port {port}")


def _build_llamacpp_args(model_info, parallel_slots: int) -> list[str]:
    args = shlex.split(getattr(model_info, "extra_args", "") or "")
    args, ctx_from_args = _pop_flag_value(args, _LLAMACPP_CONTEXT_ALIASES)
    args, _ = _pop_flag_value(args, _LLAMACPP_PARALLEL_ALIASES)
    args = _drop_flags(args, _LLAMACPP_CONT_BATCHING_ALIASES)

    per_slot_ctx_size = getattr(model_info, "ctx_size", None)
    if per_slot_ctx_size is None:
        per_slot_ctx_size = _parse_int(ctx_from_args)

    built_args = ["--parallel", str(parallel_slots), "--cont-batching"]
    if per_slot_ctx_size:
        built_args.extend(["--ctx-size", str(int(per_slot_ctx_size) * parallel_slots)])
    built_args.extend(args)
    return built_args


def _measurement_command(model_info, parallel_slots: int, port: int) -> list[str]:
    from config import GGUF_MODEL_DIR, LLAMA_SERVER_BIN

    model_path = os.path.join(GGUF_MODEL_DIR, model_info.backend_id)
    cmd = [LLAMA_SERVER_BIN, "-m", model_path, "--port", str(port)]
    if getattr(model_info, "mmproj_id", None):
        cmd.extend(["--mmproj", os.path.join(GGUF_MODEL_DIR, model_info.mmproj_id)])
    cmd.extend(_build_llamacpp_args(model_info, parallel_slots))
    return cmd


def _measurement_env(model_info) -> dict:
    from config import BACKEND_LD_LIBRARY_PATH, CUDA_LIB_PATH, GPU_VISIBLE_DEVICES, HF_TOKEN

    env = os.environ.copy()
    env["HF_TOKEN"] = HF_TOKEN
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if BACKEND_LD_LIBRARY_PATH is not None:
        env["LD_LIBRARY_PATH"] = BACKEND_LD_LIBRARY_PATH
    elif CUDA_LIB_PATH:
        env["LD_LIBRARY_PATH"] = f"{CUDA_LIB_PATH}:{env.get('LD_LIBRARY_PATH', '')}"

    if getattr(model_info, "gpu_index", None) is not None:
        visible_devices = [int(model_info.gpu_index)]
    else:
        visible_devices = list(GPU_VISIBLE_DEVICES)
    if visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(idx) for idx in visible_devices)
    return env


def measure_parallel_memory_gb(model_info, parallel_slots: int, log_dir: str = "model_logs") -> float:
    os.makedirs(log_dir, exist_ok=True)
    port = _allocate_port()
    cmd = _measurement_command(model_info, parallel_slots, port)
    env = _measurement_env(model_info)

    safe_model_id = model_info.model_id.replace(":", "_").replace("/", "_")
    log_path = os.path.join(log_dir, f"{safe_model_id}.measure-p{parallel_slots}.log")
    kwargs = {}
    if hasattr(os, "setsid"):
        kwargs["preexec_fn"] = os.setsid
    elif hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

    with open(log_path, "w") as log_file:
        log_file.write(f"[measure_vram] model_id={model_info.model_id}\n")
        log_file.write(f"[measure_vram] parallel_slots={parallel_slots}\n")
        log_file.write(f"[measure_vram] cmd={shlex.join(cmd)}\n\n")
        log_file.flush()

        process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            **kwargs,
        )
        try:
            _wait_for_health(port, process)
            time.sleep(2)
            measured_gb = _process_vram_gb(process.pid)
            if measured_gb <= 0:
                raise RuntimeError("nvidia-smi did not report VRAM for the measurement process")
            return measured_gb
        finally:
            _terminate_process(process)


def measure_model(model_id: str, force: bool = False) -> dict:
    from model_resources import (
        estimate_slot_memory_gb,
        get_measured_model_memory,
        save_measured_model_memory,
    )
    from models import MODEL_REGISTRY

    if model_id not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_id}' not found in registry")

    model_info = MODEL_REGISTRY[model_id]
    if model_info.backend != "llamacpp":
        raise ValueError(f"VRAM measurement currently supports llamacpp only: {model_id}")

    existing = get_measured_model_memory(model_info)
    if existing is not None and not force:
        return existing

    parallel_1_memory_gb = measure_parallel_memory_gb(model_info, 1)
    try:
        parallel_2_memory_gb = measure_parallel_memory_gb(model_info, 2)
    except Exception as exc:
        print(f"[measure_vram] parallel=2 measurement failed for {model_id}: {exc}")
        parallel_2_memory_gb = None

    if parallel_2_memory_gb is None:
        base_memory_gb = 0.0
        slot_memory_gb = max(parallel_1_memory_gb, estimate_slot_memory_gb(model_info))
    else:
        slot_memory_gb = max(0.0, parallel_2_memory_gb - parallel_1_memory_gb)
        if slot_memory_gb <= 0:
            slot_memory_gb = estimate_slot_memory_gb(model_info)
        base_memory_gb = max(0.0, parallel_1_memory_gb - slot_memory_gb)

    save_measured_model_memory(
        model_info,
        base_memory_gb=base_memory_gb,
        slot_memory_gb=slot_memory_gb,
        parallel_1_memory_gb=parallel_1_memory_gb,
        parallel_2_memory_gb=parallel_2_memory_gb,
    )

    return {
        "base_memory_gb": base_memory_gb,
        "slot_memory_gb": slot_memory_gb,
        "parallel_1_memory_gb": parallel_1_memory_gb,
        "parallel_2_memory_gb": parallel_2_memory_gb,
        "source": "measure_vram.py",
    }


def ensure_model_measured(model_id: str) -> dict:
    return measure_model(model_id, force=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure llama.cpp model VRAM into model_info.db")
    parser.add_argument("--config-dir", help="Directory containing alternate config.py")
    parser.add_argument("--model", help="Model ID to measure")
    parser.add_argument("--all", action="store_true", help="Measure every llamacpp model in the registry")
    parser.add_argument("--force", action="store_true", help="Refresh measurements even if DB rows exist")
    return parser.parse_args()


def main():
    args = parse_args()
    _configure_import_path(args.config_dir)

    from models import MODEL_REGISTRY

    if not args.model and not args.all:
        raise SystemExit("Provide --model MODEL_ID or --all")

    model_ids = [args.model] if args.model else [
        model_id
        for model_id, model_info in MODEL_REGISTRY.items()
        if model_info.backend == "llamacpp"
    ]

    for model_id in model_ids:
        print(f"[measure_vram] measuring {model_id}", flush=True)
        result = measure_model(model_id, force=args.force)
        print(
            f"[measure_vram] {model_id}: "
            f"base={result['base_memory_gb']:.3f} GB, "
            f"slot={result['slot_memory_gb']:.3f} GB, "
            f"p1={result['parallel_1_memory_gb']:.3f} GB, "
            f"p2={result.get('parallel_2_memory_gb') or 0.0:.3f} GB"
        )


if __name__ == "__main__":
    main()
