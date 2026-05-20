import asyncio
import logging
import subprocess
import os
import signal
import shlex
import time
import httpx
from typing import Dict, Optional

from model_resources import effective_parallel_slots

# collection of functions that handle model loading and unloading

logger = logging.getLogger(__name__)

_CONTEXT_LENGTH_FLAGS = {
    "llamacpp": "--ctx-size",
    "vllm": "--max-model-len",
}

_CONTEXT_LENGTH_ALIASES = {
    "-c",
    "--ctx-size",
    "--ctx_size",
    "--max-model-len",
    "--max_model_len",
}

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

_TOKENIZER_ALIASES = {
    "--tokenizer",
}


def _args_include_flag(args: list[str], aliases: set[str]) -> bool:
    return any(arg.partition("=")[0] in aliases for arg in args)


def _normalize_backend_args(backend: str, extra_args: str) -> list[str]:
    args = shlex.split(extra_args)
    context_flag = _CONTEXT_LENGTH_FLAGS.get(backend)
    if context_flag is None:
        return args

    normalized_args = []
    i = 0
    while i < len(args):
        arg = args[i]
        flag, sep, value = arg.partition("=")

        if flag in _CONTEXT_LENGTH_ALIASES:
            normalized_args.append(context_flag)
            if sep:
                normalized_args.append(value)
            elif i + 1 < len(args):
                i += 1
                normalized_args.append(args[i])
            else:
                logger.warning("Context length flag %s was provided without a value", arg)
            i += 1
            continue

        normalized_args.append(arg)
        i += 1

    return normalized_args


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
            else:
                logger.warning("Flag %s was provided without a value", arg)
            i += 1
            continue

        filtered.append(arg)
        i += 1

    return filtered, value


def _drop_flags(args: list[str], aliases: set[str]) -> list[str]:
    return [arg for arg in args if arg.partition("=")[0] not in aliases]


def _parse_int(value: Optional[str], flag_name: str) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        logger.warning("Ignoring invalid integer value for %s: %s", flag_name, value)
        return None


def _build_llamacpp_extra_args(model_info) -> list[str]:
    args = shlex.split(model_info.extra_args or "")
    args, ctx_from_args = _pop_flag_value(args, _LLAMACPP_CONTEXT_ALIASES)
    args, _ = _pop_flag_value(args, _LLAMACPP_PARALLEL_ALIASES)
    args = _drop_flags(args, _LLAMACPP_CONT_BATCHING_ALIASES)

    parallel_slots = effective_parallel_slots(model_info)
    per_slot_ctx_size = getattr(model_info, "ctx_size", None)
    if per_slot_ctx_size is None:
        per_slot_ctx_size = _parse_int(ctx_from_args, "ctx-size")

    built_args = ["--parallel", str(parallel_slots), "--cont-batching"]
    if per_slot_ctx_size:
        built_args.extend(["--ctx-size", str(int(per_slot_ctx_size) * parallel_slots)])
    built_args.extend(args)
    return built_args


def _resolve_vllm_model_path(backend_id: str, gguf_model_dir: str, vllm_model_dir: str) -> str:
    if os.path.isabs(backend_id) or os.path.exists(backend_id):
        return backend_id

    vllm_model_path = os.path.join(vllm_model_dir, backend_id)
    if os.path.exists(vllm_model_path):
        return vllm_model_path

    if backend_id.lower().endswith(".gguf"):
        return os.path.join(gguf_model_dir, backend_id)

    return backend_id


def _is_gguf_model(model_path: str) -> bool:
    return model_path.lower().endswith(".gguf")


class ModelProcessManager:
    # Manages subprocesses for model backends (vllm and llama.cpp).
    def __init__(self, port_start: int, port_end: int):
        self.port_start = port_start
        self.port_end = port_end
        self.active_processes: Dict[str, subprocess.Popen] = {}
        self.model_ports: Dict[str, int] = {}
        self.used_ports = set()
    
    def _allocate_port(self) -> int:
        # Find the next available port in the configured range.
        for port in range(self.port_start, self.port_end + 1):
            if port not in self.used_ports:
                self.used_ports.add(port)
                return port
        raise RuntimeError("No available ports in the configured range.")

    def _release_port(self, port: int):
        # Release a previously allocated port.
        if port in self.used_ports:
            self.used_ports.remove(port)

    async def check_health(self, port: int, process: subprocess.Popen, timeout: int = 1800) -> bool:
        # Poll the local endpoint to check if the subprocess is ready to accept requests.
        url = f"http://127.0.0.1:{port}/v1/models"
        start_time = time.time()
        
        async with httpx.AsyncClient() as client:
            while time.time() - start_time < timeout:
                try:
                    response = await client.get(url, timeout=2.0)
                    if response.status_code == 200:
                        return True
                except httpx.RequestError:
                    pass
                
                # Check if the process died prematurely while we were waiting
                if process.poll() is not None:
                    logger.error(f"Process on port {port} terminated unexpectedly with exit code {process.poll()}")
                    return False
                    
                await asyncio.sleep(2)
        return False

    async def load_model(self, model_id: str, model_info):
        # Spawns a new subprocess for the requested model based on its 'backend' config.
        # Waits for the health check to pass before returning.
        if model_id in self.active_processes:
            logger.info(f"Model {model_id} is already loaded.")
            return

        if not hasattr(model_info, "backend"):
            raise ValueError(f"Invalid model_info object for model {model_id}")

        port = self._allocate_port()
        self.model_ports[model_id] = port
        backend = model_info.backend
        
        from config import (
            GGUF_MODEL_DIR, VLLM_MODEL_DIR, LLAMA_SERVER_BIN, 
            CUDA_LIB_PATH, HF_TOKEN, NUM_GPUS
        )
        
        env = os.environ.copy()
        env["HF_TOKEN"] = HF_TOKEN
        if CUDA_LIB_PATH:
            env["LD_LIBRARY_PATH"] = f"{CUDA_LIB_PATH}:{env.get('LD_LIBRARY_PATH', '')}"

        if getattr(model_info, "gpu_index", None) is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(model_info.gpu_index)

        # Build the command based on backend
        if backend == "vllm":
            model_path = _resolve_vllm_model_path(
                model_info.backend_id, GGUF_MODEL_DIR, VLLM_MODEL_DIR
            )
            extra_args = _normalize_backend_args(backend, model_info.extra_args)
            cmd = [
                "python", "-m", "vllm.entrypoints.openai.api_server",
                "--model", model_path,
                "--port", str(port),
                "--tensor-parallel-size", str(NUM_GPUS),
            ]
            hf_repo = getattr(model_info, "hf_repo", None)
            if hf_repo and _is_gguf_model(model_path) and not _args_include_flag(extra_args, _TOKENIZER_ALIASES):
                cmd.extend(["--tokenizer", hf_repo])
            elif _is_gguf_model(model_path) and not _args_include_flag(extra_args, _TOKENIZER_ALIASES):
                logger.warning(
                    "vLLM GGUF model %s has no hf_repo or --tokenizer configured; "
                    "multimodal GGUF models usually require one.",
                    model_id,
                )
            cmd.extend(extra_args)
        else: # llamacpp
            model_path = os.path.join(GGUF_MODEL_DIR, model_info.backend_id)
            cmd = [
                LLAMA_SERVER_BIN,
                "-m", model_path,
                "--port", str(port),
            ]
            if getattr(model_info, "mmproj_id", None):
                mmproj_path = os.path.join(GGUF_MODEL_DIR, model_info.mmproj_id)
                cmd.extend(["--mmproj", mmproj_path])
            cmd.extend(_build_llamacpp_extra_args(model_info))

        logger.info(f"Starting {backend} backend for {model_id} on port {port}")
        
        # Start the process group so we can kill easily (Unix only)
        kwargs = {}
        if hasattr(os, 'setsid'):
            kwargs['preexec_fn'] = os.setsid
        elif hasattr(subprocess, 'CREATE_NEW_PROCESS_GROUP'):
            kwargs['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP
            
        # Create logs directory if it doesn't exist
        os.makedirs("model_logs", exist_ok=True)
        safe_model_id = model_id.replace(':', '_').replace('/', '_')
        log_file = open(f"model_logs/{safe_model_id}.log", "w")
        
        process = subprocess.Popen(
            cmd, 
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            **kwargs
        )
        
        # Store process info
        process._log_file = log_file # Keep reference to close later
        self.active_processes[model_id] = process
        self.model_ports[model_id] = port
        
        # Wait for the model to be ready
        logger.info(f"Waiting for {model_id} to become healthy on port {port}...")
        is_healthy = await self.check_health(port, process, timeout=1800) # Wait up to 30 minutes
        
        if not is_healthy:
            logger.error(f"Health check failed for {model_id} on port {port}. Terminating.")
            await self.unload_model(model_id)
            raise RuntimeError(f"Failed to load model {model_id} (crashed or timeout). Check model_logs/{safe_model_id}.log")
            
        logger.info(f"Model {model_id} is healthy and ready.")

    async def unload_model(self, model_id: str):
        # unload a model by killing its subprocess and releasing the port.
        import subprocess # HACK: Local import to prevent UnboundLocalError
        if model_id not in self.active_processes:
            return

        process = self.active_processes.pop(model_id)
        port = self.model_ports.pop(model_id)
        
        logger.info(f"Unloading model {model_id} from port {port}")
        
        try:
            # Send SIGTERM to the process group (Unix) or terminate tree (Windows)
            if hasattr(os, 'killpg'):
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            else:
                subprocess.run(['taskkill', '/F', '/T', '/PID', str(process.pid)], capture_output=True)
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning(f"Process {model_id} did not terminate gracefully. Killing it.")
            if hasattr(os, 'killpg'):
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            else:
                subprocess.run(['taskkill', '/F', '/T', '/PID', str(process.pid)], capture_output=True)
        except ProcessLookupError:
            pass # poke dead
            
        if hasattr(process, '_log_file') and process._log_file:
            try:
                process._log_file.close()
            except:
                pass
                
        self._release_port(port)
        logger.info(f"Model {model_id} unloaded successfully.")

    async def cleanup(self):
        # make sure to kill all managed subprocesses during shutdown.
        models = list(self.active_processes.keys())
        for model_id in models:
            await self.unload_model(model_id)

from config import MODEL_PORT_START, MODEL_PORT_END
process_manager = ModelProcessManager(MODEL_PORT_START, MODEL_PORT_END)
