# OpenAI Proxy v6

A FastAPI-based proxy server designed to host and route OpenAI-compatible API requests to local language models, dynamically managing memory and subprocesses (`vLLM` and `llama.cpp`).

## Features
- **Dynamic Model Loading/Unloading:** Automatically starts and stops model subprocesses based on configured GPU memory budgets.
- **Smart Batching & Scheduling:** Queues requests and orders them to minimize the overhead of swapping models in and out of memory.
- **API Key & Quota Management:** Tracks token usage and limits access via a local SQLite database.

## Architecture & Files

- `main.py`: Entry point to run the proxy server or the key management CLI.
- `proxy_server.py`: Core FastAPI application that handles incoming endpoints, request routing, and authentication/quota checks.
- `process_manager.py`: Manages spawning, health-checking, and cleanly terminating model backends (`llama-server` and `vLLM`) on dynamic ports.
- `request_queue.py`: Handles request batching and GPU memory tracking. Decides when models need to be loaded or evicted to fit within the memory budget.
- `models.py`: The `MODEL_REGISTRY` cataloging all supported models, their required backend, and estimated memory footprints.
- `db.py`: SQLite database operations for managing API keys and tracking token quotas.
- `cli.py`: Interactive terminal menu for database management (adding keys, updating limits).
- `config.py`: System configuration, including available GPU memory, file paths, proxy settings, and batching constraints.


## Setup

1. **Install dependencies:** Create a Conda environment and install the required Python packages using `uv`.
```bash
./setup.sh
```

2. **Build `llama.cpp`:** The proxy uses the `llama-server` binary for hosting GGUF models. You need to clone and build it.
```bash
# Clone the repository
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp

# Build with CUDA support (for GPU)
make GGML_CUDA=1

# Or, build for CPU only
# make

# The compiled binary will be located at:
# ./llama-server
```
Make sure to update `LLAMA_SERVER_BIN` in `config.py` to point to the `llama-server` binary you just built.
HACK: I have put the llama.cpp/build/bin/llama-server in the openai-proxyv6 folder for now. Please change it later.

3. Please read config.py
Everything need to be manually configured is in this file.
This file contains the configuration for the proxy server.

## Usage

```bash
# Start the proxy server
python main.py

# Open interactive API key management menu
python main.py --manage
```
