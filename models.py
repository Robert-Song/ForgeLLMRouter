"""
proxy_v6/models.py
Central model catalog with backend type, memory estimates, and ID mappings.

Each model has:
  - model_id:     The Ollama-style tag used in API requests
  - backend:      "llamacpp" or "vllm"
  - backend_id:   GGUF filename (for llamacpp) or HuggingFace model ID (for vllm)
  - memory_gb:    Estimated GPU VRAM needed to load the model
  - model_type:   "chat", "embedding", or "reranker"

Backend decision rule:
  - Small models (≤~10B params), embeddings, rerankers → llamacpp
    (faster loading, acceptable inference for low concurrency)
  - Large "smart" chat models → vllm
    (paged attention, better throughput, auto-downloads from HuggingFace)
"""
# IMPORTANT TODO: dengcao/Qwen3-Reranker-8B:F16	and dengcao/Qwen3-Reranker-8B:Q8_0 should be reconsidered.
# GGUF not good for reranker
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from config import GPU_MEMORY_GB, GPU_HEADROOM_GB, TOTAL_USABLE_MEMORY_GB


@dataclass
class ModelInfo:
    model_id: str                   # Ollama-style tag (API request name)
    backend: str                    # "llamacpp" or "vllm"
    backend_id: str                 # GGUF filename or HuggingFace ID
    memory_gb: float                # Estimated GPU VRAM (GB)
    model_type: str                 # "chat", "embedding", "reranker"
    group: str = ""                 # Assigned later by build_groups()
    gpu_index: Optional[int] = None # Which GPU to place on (None = auto)
    extra_args: str = ""            # Extra CLI flags for the backend


# MODEL REGISTRY
# Edit this list to add/remove/modify models.
# Memory estimates are approximate (model weights + KV cache overhead).
# -> you can also actually try to run and measure the memory usage

MODEL_REGISTRY: Dict[str, ModelInfo] = {}

def _register(*models: ModelInfo):
    for m in models:
        MODEL_REGISTRY[m.model_id] = m

_register(
    # ---- CHAT MODELS (small → llamacpp) ----
    ModelInfo(
        model_id="gemma3:4b-it-q8_0",
        backend="llamacpp",
        backend_id="gemma-3-4b-it-Q8_0.gguf",
        memory_gb=9.0,
        model_type="chat",
        extra_args="--ctx-size 65536",
    ),

    # ---- CHAT MODELS (medium → llamacpp for GGUF quantized) ----
    ModelInfo(
        model_id="olmo-3:32b-think-q8_0",
        backend="llamacpp",
        backend_id="olmo-3-32b-think-Q8_0.gguf",
        memory_gb=41.0, #34+4(KV)
        model_type="chat",
        extra_args="--ctx-size 65536",
    ),
    ModelInfo(
        model_id="nemotron-3-nano:30b-a3b-q8_0",
        backend="llamacpp",
        backend_id="nemotron-3-nano-30b-a3b-Q8_0.gguf",
        memory_gb=35.0,
        model_type="chat",
        extra_args="--ctx-size 65536",
    ),
    ModelInfo(
        model_id="gemma3:27b-it-q8_0",
        backend="llamacpp",
        backend_id="gemma-3-27b-it-Q8_0.gguf",
        memory_gb=37.0,
        model_type="chat",
        extra_args="--ctx-size 65536",
    ),
    ModelInfo(
        model_id="gemma3:27b",
        backend="llamacpp",
        backend_id="gemma-3-27b-Q8_0.gguf",
        memory_gb=37.0,
        model_type="chat",
        extra_args="--ctx-size 65536",
    ),
    ModelInfo(
        model_id="qwen3:30b-a3b-thinking-2507-q4_K_M",
        backend="llamacpp",
        backend_id="Qwen3-30B-A3B-Thinking-2507-Q4_K_M.gguf",
        memory_gb=26,
        model_type="chat",
        extra_args="--ctx-size 65536",
    ),
    ModelInfo(
        model_id="qwen3.5:27b-q4_K_M",
        backend="llamacpp",
        backend_id="qwen3.5-27b-q4_K_M.gguf",
        memory_gb=22.75,
        model_type="chat",
        extra_args="--ctx-size 65536",
    ),
    ModelInfo(
        model_id="qwen3.5:35b-a3b-q4_K_M",
        backend="llamacpp",
        backend_id="qwen3.5-35b-a3b-q4_K_M.gguf",
        memory_gb=25.31,
        model_type="chat",
        extra_args="--ctx-size 65536",
    ),
    ModelInfo(
        model_id="qwen3.5-122b-a10b-q4_K_M",
        backend="llamacpp",
        backend_id="Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf",
        memory_gb=77.04,
        model_type="chat",
        extra_args="--ctx-size 65536",
    ),
    ModelInfo(
        model_id="GLM-4.5-Air-q4_K_M",
        backend="llamacpp",
        backend_id="GLM-4.5-Air-Q4_K_M-00001-of-00002.gguf",
        memory_gb=52.20,
        model_type="chat",
        extra_args="--ctx-size 65536",
    ),
    ModelInfo(
        model_id="MiniMax-M2.5-q4_K_M",
        backend="llamacpp",
        backend_id="MiniMax-M2.5-Q4_K_M-00001-of-00004.gguf",
        memory_gb=47.29,
        model_type="chat",
        extra_args="--ctx-size 65536",
    ),
    ModelInfo(
        model_id="gemma4:26b-a4b-it-q8_0",
        backend="llamacpp",
        backend_id="gemma-4-26B-A4B-it-Q8_0.gguf",
        memory_gb=35.0,
        model_type="chat",
        extra_args="--ctx-size 65536",
    ),
    ModelInfo(
        model_id="gemma4:31b-it-q8_0",
        backend="llamacpp",
        backend_id="gemma-4-31B-it-Q8_0.gguf",
        memory_gb=40.0,
        model_type="chat",
        extra_args="--ctx-size 65536",
    ),

    # ---- CHAT MODELS (large → vllm (currently unused)) ----
    ModelInfo(
        model_id="llama3.3:70b-instruct-q8_0",
        backend="llamacpp",
        backend_id="Llama-3.3-70B-Instruct-Q8_0-00001-of-00002.gguf",
        memory_gb=91.5,
        model_type="chat",
        extra_args="--ctx-size 65536 --n-gpu-layers 99",
    ),
    ModelInfo(
        model_id="mistral-large:123b",
        backend="llamacpp",
        backend_id="Mistral-Large-Instruct-2411-Q4_K_M-00001-of-00002.gguf",
        memory_gb=94.0,
        model_type="chat",
        extra_args="--ctx-size 65536 --n-gpu-layers 99",
    ),
    ModelInfo(
        model_id="qwen3:235b-a22b",
        backend="llamacpp",
        backend_id="Qwen_Qwen3-235B-A22B-Instruct-2507-Q4_K_M-00001-of-00004.gguf",
        memory_gb=142.0,
        model_type="chat",
        extra_args="--ctx-size 32768 --n-gpu-layers 99",
    ),
    ModelInfo(
        model_id="command-a:111b",
        backend="llamacpp",
        backend_id="CohereForAI_c4ai-command-a-03-2025-Q4_K_M-00001-of-00002.gguf",
        memory_gb=74.0,
        model_type="chat",
        extra_args="--ctx-size 65536 --n-gpu-layers 99",
    ),
    ModelInfo(
        model_id="gpt-oss:120b",
        backend="llamacpp",
        backend_id="huizimao_gpt-oss-120b-uncensored-bf16-MXFP4_MOE-00001-of-00002.gguf",
        memory_gb=64.5,
        model_type="chat",
        extra_args="--ctx-size 65536 --n-gpu-layers 99",
    ),

    # ---- EMBEDDING MODELS ----
    ModelInfo(
        model_id="qwen3-embedding:8b-fp16",
        backend="llamacpp",
        backend_id="qwen3-embedding-8b-fp16.gguf",
        memory_gb=18.0,
        model_type="embedding",
        extra_args="--embedding --ctx-size 8192",
    ),
    ModelInfo(
        model_id="qwen3-embedding:8b-q8_0",
        backend="llamacpp",
        backend_id="qwen3-embedding-8b-Q8_0.gguf",
        memory_gb=11.5,
        model_type="embedding",
        extra_args="--embedding --ctx-size 8192",
    ),
    ModelInfo(
        model_id="qwen3-embedding:4b-q8_0",
        backend="llamacpp",
        backend_id="qwen3-embedding-4b-Q8_0.gguf",
        memory_gb=7.5,
        model_type="embedding",
        extra_args="--embedding --ctx-size 8192",
    ),
    ModelInfo(
        model_id="qwen3-embedding:0.6b",
        backend="llamacpp",
        backend_id="qwen3-embedding-0.6b.gguf",
        memory_gb=4.5,
        model_type="embedding",
        extra_args="--embedding --ctx-size 8192",
    ),
    ModelInfo(
        model_id="qwen3-embedding:4b",
        backend="llamacpp",
        backend_id="qwen3-embedding-4b.gguf",
        memory_gb=11.5,
        model_type="embedding",
        extra_args="--embedding --ctx-size 8192",
    ),
    ModelInfo(
        model_id="qwen3-embedding:8b",
        backend="llamacpp",
        backend_id="qwen3-embedding-8b.gguf",
        memory_gb=18.0,
        model_type="embedding",
        extra_args="--embedding --ctx-size 8192",
    ),
    ModelInfo(
        model_id="nomic-embed-text:latest",
        backend="llamacpp",
        backend_id="nomic-embed-text-v1.5.f16.gguf",
        memory_gb=2.0,
        model_type="embedding",
        extra_args="--embedding --ctx-size 8192",
    ),
    ModelInfo(
        model_id="embeddinggemma:latest",
        backend="llamacpp",
        backend_id="embeddinggemma-latest.gguf",
        memory_gb=5.0,
        model_type="embedding",
        extra_args="--embedding --ctx-size 8192",
    ),

    # ---- RERANKER MODELS ----
    ModelInfo(
        model_id="hf.co/jinaai/jina-reranker-v3-GGUF:BF16",
        backend="llamacpp",
        backend_id="jina-reranker-v3-BF16.gguf",
        memory_gb=19.5,
        model_type="reranker",
        extra_args="--reranking",
    ),
    ModelInfo(
        model_id="dengcao/Qwen3-Reranker-8B:F16",
        backend="llamacpp",
        backend_id="Qwen3-Reranker-8B-F16.gguf",
        memory_gb=16.0,
        model_type="reranker",
        extra_args="--reranking",
    ),
    ModelInfo(
        model_id="dengcao/Qwen3-Reranker-0.6B:F16",
        backend="llamacpp",
        backend_id="Qwen3-Reranker-0.6B-F16.gguf",
        memory_gb=8.5,
        model_type="reranker",
        extra_args="--reranking",
    ),
    ModelInfo(
        model_id="dengcao/Qwen3-Reranker-8B:Q8_0",
        backend="llamacpp",
        backend_id="Qwen3-Reranker-8B-Q8_0.gguf",
        memory_gb=9.0,
        model_type="reranker",
        extra_args="--reranking",
    ),
    ModelInfo(
        model_id="hf.co/mradermacher/colbertv2.0-GGUF:F16",
        backend="llamacpp",
        backend_id="colbertv2.0-F16.gguf",
        memory_gb=2.0,
        model_type="reranker",
        extra_args="--reranking",
    ),
)

def get_models_by_type(model_type: str) -> List[ModelInfo]:
    """Return all models of a given type (chat, embedding, reranker)."""
    return [m for m in MODEL_REGISTRY.values() if m.model_type == model_type]


def get_models_by_backend(backend: str) -> List[ModelInfo]:
    """Return all models using a given backend (llamacpp, vllm)."""
    return [m for m in MODEL_REGISTRY.values() if m.backend == backend]


def get_model(model_id: str) -> Optional[ModelInfo]:
    """Look up a model by its API-facing model_id."""
    return MODEL_REGISTRY.get(model_id)


def list_all_model_ids() -> List[str]:
    """Return all registered model IDs."""
    return list(MODEL_REGISTRY.keys())
