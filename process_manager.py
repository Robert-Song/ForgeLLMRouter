import asyncio
import logging
import subprocess
import os
import signal
import time
import httpx
from typing import Dict, Optional

# collection of functions that handle model loading and unloading

logger = logging.getLogger(__name__)

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
        
        import shlex
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
            cmd = [
                "python", "-m", "vllm.entrypoints.openai.api_server",
                "--model", model_info.backend_id,
                "--port", str(port),
                "--tensor-parallel-size", str(NUM_GPUS),
            ]
            if model_info.extra_args:
                cmd.extend(shlex.split(model_info.extra_args))
        else: # llamacpp
            model_path = os.path.join(GGUF_MODEL_DIR, model_info.backend_id)
            cmd = [
                LLAMA_SERVER_BIN,
                "-m", model_path,
                "--port", str(port),
            ]
            if model_info.extra_args:
                cmd.extend(shlex.split(model_info.extra_args))

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

