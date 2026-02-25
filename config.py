"""
proxy_v6/config.py
HPC resource constants and tuning knobs.
Change the values at the top to match your HPC environment.
"""

### HPC RESOURCE CONSTANTS ###

# GPU memory available for model loading (in GB).
# Set conservatively to leave room for other users (I recommend so...?)
GPU_MEMORY_GB = [90, 90]           # Available VRAM per GPU (list, one per GPU) 96GB? I think
NUM_GPUS = len(GPU_MEMORY_GB)
TOTAL_USABLE_MEMORY_GB = sum(GPU_MEMORY_GB)  # 180 GB for aiforge

# Safety margin: don't fill GPU to 100%, leave headroom (# in GB per GPU)
GPU_HEADROOM_GB = 2


### MODEL FILE PATHS ###

# Base directory for locally-stored GGUF files (llama.cpp models)
GGUF_MODEL_DIR = "/scratch2/ribeirob/models"

# Directory for vLLM/HuggingFace model downloads
VLLM_MODEL_DIR = "/scratch2/ribeirob/models"

# HuggingFace token for gated models (set via env)
import os as _os
HF_TOKEN = _os.environ.get("HF_TOKEN", "")
# CUDA library path (needed by llama-server/llama-swap)
CUDA_LIB_PATH = "/p/cuda-12.3/targets/x86_64-linux/lib"

# llama-server binary path (from llama.cpp build)
# XXX: this line is CPU only version, only use to test, do not use actually
# LLAMA_SERVER_BIN = "/.gavea/store/song669/temp/llama-b7999/llama-server"

# this is the GPU version
LLAMA_SERVER_BIN = "/.gavea/store/song669/temp/llama.cpp/build/bin/llama-server"
# LLAMA_SERVER_BIN = "llama.cpp/build/bin/llama-server"

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


### DATABASE
DB_PATH = "token_usage.db"
