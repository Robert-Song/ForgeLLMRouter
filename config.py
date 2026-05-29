"""
proxy_v6/config.py
HPC resource constants and tuning knobs.
Change the values at the top to match your HPC environment.
"""

### HPC RESOURCE CONSTANTS ###

# GPU memory available for model loading (in GB). Set a GPU to 0 to exclude it
# from both the scheduler budget and backend CUDA visibility.
GPU_MEMORY_GB = [42, 42, 42] #birck, temp
GPU_VISIBLE_DEVICES = [idx for idx, gb in enumerate(GPU_MEMORY_GB) if gb > 0]
NUM_GPUS = len(GPU_VISIBLE_DEVICES)
TOTAL_USABLE_MEMORY_GB = sum(GPU_MEMORY_GB)

# Safety margin: don't fill GPU to 100%, leave headroom (# in GB per visible GPU)
GPU_HEADROOM_GB = 3


# TIMEZONE SETTING
import os as _os
import time as _time

# Force timezone to Eastern Time (EST/EDT)
_os.environ['TZ'] = 'America/New_York'
try:
    _time.tzset()
except AttributeError:
    pass

### MODEL FILE PATHS ###

# Base directory for locally-stored GGUF files (llama.cpp models)
GGUF_MODEL_DIR = "/.gavea/store/song669/temp/gguf_models/"
#GGUF_MODEL_DIR = "/scratch2/ribeirob/models"

# Directory for vLLM/HuggingFace model downloads
VLLM_MODEL_DIR = "/.gavea/store/song669/temp/vllm_models/"
#VLLM_MODEL_DIR = "/scratch2/ribeirob/models"

# HuggingFace token for gated models (set via env)
import glob as _glob
import os as _os
HF_TOKEN = _os.environ.get("HF_TOKEN", "")#HACK NEED TO CHANGE THIS LATER
# CUDA library path (needed by llama-server/llama-swap)
_REPO_DIR = _os.path.dirname(_os.path.abspath(__file__))
_CUDA_LIB_PATHS = [
    "/p/cuda-12.3/targets/x86_64-linux/lib",
]
for _package in ("cuda_runtime", "cublas", "nccl"):
    _CUDA_LIB_PATHS.extend(
        _glob.glob(
            _os.path.join(
                _REPO_DIR,
                ".venv",
                "lib",
                "python*",
                "site-packages",
                "nvidia",
                _package,
                "lib",
            )
        )
    )
CUDA_LIB_PATH = _os.pathsep.join(
    dict.fromkeys(path for path in _CUDA_LIB_PATHS if _os.path.isdir(path))
)

# Optional exact LD_LIBRARY_PATH for spawned model backends. Use this on machines
# where the default CUDA path would select a runtime newer than the installed
# driver supports.
BACKEND_LD_LIBRARY_PATH = _os.environ.get("FORGE_BACKEND_LD_LIBRARY_PATH")

# llama-server binary path (from llama.cpp build)
# XXX: this line is CPU only version, only use to test, do not use actually
# LLAMA_SERVER_BIN = "/.gavea/store/song669/temp/llama-b7999/llama-server"

# this is the GPU version
#LLAMA_SERVER_BIN = "/.gavea/store/song669/temp/llama.cpp/build/bin/llama-server"
LLAMA_SERVER_BIN = "/.gavea/store/song669/clean/ForgeLLMRouter/llama.cpp/build/bin/llama-server"

### PROXY / SERVER SETTINGS ###

PROXY_HOST = "0.0.0.0"
PROXY_PORT = 8002

# this is for handling multiple models (via ports)
# XXX: increase range if running on very large HPC that can load more than 80 models
MODEL_PORT_START = 8010
MODEL_PORT_END = 8090

# Health check timeout (in seconds)
HEALTH_CHECK_TIMEOUT = 600

### request_queue / BATCHING SETTINGS

MAX_QUEUE_SIZE = 64                # max pending requests in the request_queue
BATCH_WINDOW_MS = 50               # ms to wait for batch accumulation
LRU_CACHE_SIZE = 128               # max entries in LRU scheduling cache

# Upper bound for automatically-selected llama.cpp parallel slots per model.
# Models can still set a smaller explicit parallel_slots value in models.py.
MAX_PARALLEL_SLOTS = 16

# llama.cpp's fit/load behavior can reserve substantially more VRAM than the
# static GGUF tensor + KV estimates, especially with parallel slots. When a new
# model does not fit under the scheduler budget, unload idle models before load.
UNLOAD_IDLE_MODELS_BEFORE_LOAD = True


### DATABASE
DB_PATH = "token_usage.db"
