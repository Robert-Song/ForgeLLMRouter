"""
proxy_v6/models.py
Central model catalog with backend type, local model IDs, and runtime hints.

Each model has:
  - model_id:     The Ollama-style tag used in API requests
  - backend:      "llamacpp" or "vllm"
  - backend_id:   GGUF filename (for llamacpp) or HuggingFace model ID (for vllm)
  - hf_repo:      Optional HuggingFace repo/path for vLLM tokenizer/config metadata
  - parallel_slots: llama.cpp server slots for same-model parallel requests
  - ctx_size:     Per-slot llama.cpp context size; launcher multiplies by slots
  - slot_memory_gb: Optional override for extra VRAM per additional llama.cpp slot
  - model_type:   "chat", "embedding", or "reranker"
  - modalities:   Supported input modalities, e.g. ("text", "image")

Backend decision rule:
  - Small models (≤~10B params), embeddings, rerankers → llamacpp
    (faster loading, acceptable inference for low concurrency)
  - Large "smart" chat models → vllm
    (paged attention, better throughput, auto-downloads from HuggingFace)
"""
# IMPORTANT TODO: dengcao/Qwen3-Reranker-8B:F16	and dengcao/Qwen3-Reranker-8B:Q8_0 should be reconsidered.
# GGUF not good for reranker
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ModelInfo:
    model_id: str                   # Ollama-style tag (API request name)
    backend: str                    # "llamacpp" or "vllm"
    backend_id: str                 # GGUF filename or HuggingFace ID
    model_type: str                 # "chat", "embedding", "reranker"
    group: str = ""                 # Assigned later by build_groups()
    gpu_index: Optional[int] = None # Which GPU to place on (None = auto)
    extra_args: str = ""            # Extra CLI flags for the backend
    modalities: tuple[str, ...] = ("text",)  # Supported input modalities
    mmproj_id: Optional[str] = None # Multimodal projector GGUF filename
    hf_repo: Optional[str] = None   # Original HF repo/path for vLLM GGUF tokenizer/config
    parallel_slots: int = 0         # 0 = auto maximum that estimated VRAM allows
    ctx_size: Optional[int] = None  # Per-slot context; total llama.cpp ctx is ctx_size * slots
    slot_memory_gb: float = 0.0     # Optional override for extra VRAM per additional slot


# MODEL REGISTRY
# Edit this list to add/remove/modify models.

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
        model_type="chat",
        extra_args="--ctx-size 65536",
        hf_repo="google/gemma-3-4b-it",
    ),
    ModelInfo(
        model_id="gemma4:e4b-q4_K_M",
        backend="llamacpp",
        backend_id="gemma-4-E4B.Q4_K_M.gguf",
        model_type="chat",
        extra_args="--ctx-size 65536",
    ),
    ModelInfo(
        model_id="qwen3:4b-q4_K_M",
        backend="llamacpp",
        backend_id="Qwen3-4B-Q4_K_M.gguf",
        model_type="chat",
        extra_args="--ctx-size 65536",
    ),
    ModelInfo(
        model_id="qwen3:4b-instruct-2507-q4_K_M",
        backend="llamacpp",
        backend_id="Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        model_type="chat",
        extra_args="--ctx-size 65536",
    ),
    ModelInfo(
        model_id="qwen3:4b-thinking-2507-q4_K_M",
        backend="llamacpp",
        backend_id="Qwen3-4B-Thinking-2507-Q4_K_M.gguf",
        model_type="chat",
        extra_args="--ctx-size 65536",
    ),
    ModelInfo(
        model_id="qwen3:8b-q4_K_M",
        backend="llamacpp",
        backend_id="Qwen3-8B-Q4_K_M.gguf",
        model_type="chat",
        extra_args="--ctx-size 65536",
    ),
    ModelInfo(
        model_id="qwen3.5:9b-q4_K_M",
        backend="llamacpp",
        backend_id="Qwen3.5-9B-Q4_K_M.gguf",
        model_type="chat",
        extra_args="--ctx-size 65536",
    ),

    # ---- CHAT MODELS (medium → llamacpp for GGUF quantized) ----
    ModelInfo(
        model_id="olmo-3:32b-think-q8_0",
        backend="llamacpp",
        backend_id="olmo-3-32b-think-Q8_0.gguf",
        model_type="chat",
        extra_args="--ctx-size 65536",
    ),
    ModelInfo(
        model_id="nemotron-3-nano:30b-a3b-q8_0",
        backend="llamacpp",
        backend_id="nemotron-3-nano-30b-a3b-Q8_0.gguf",
        model_type="chat",
        extra_args="--ctx-size 65536",
    ),
    ModelInfo(
        model_id="gemma3:27b-it-q8_0",
        backend="llamacpp",
        #backend_id="gemma-3-27b-it-Q8_0.gguf",
        backend_id="gemma-3-27b-it-Q4_K_M.gguf",
        model_type="chat",
        extra_args="--ctx-size 65536",
        hf_repo="google/gemma-3-27b-it",
    ),
    ModelInfo(
        model_id="gemma3:27b-it-q4_K_M",
        backend="llamacpp",
        backend_id="gemma-3-27b-it-Q4_K_M.gguf",
        model_type="chat",
        extra_args="--ctx-size 65536",
        hf_repo="google/gemma-3-27b-it",
    ),
    ModelInfo(
        model_id="gemma3:12b-it-q4_K_M",
        backend="llamacpp",
        backend_id="gemma-3-12b-it-Q4_K_M.gguf",
        model_type="chat",
        extra_args="--ctx-size 65536",
        hf_repo="google/gemma-3-12b-it",
    ),
    ModelInfo(
        model_id="gemma3:27b",
        backend="llamacpp",
        backend_id="gemma-3-27b-Q8_0.gguf",
        model_type="chat",
        extra_args="--ctx-size 65536",
    ),
    ModelInfo(
        model_id="qwen3:30b-a3b-thinking-2507-q4_K_M",
        backend="llamacpp",
        backend_id="Qwen3-30B-A3B-Thinking-2507-Q4_K_M.gguf",
        model_type="chat",
        extra_args="--ctx-size 65536",
    ),
    ModelInfo(
        model_id="qwen3:14b-q4_K_M",
        backend="llamacpp",
        backend_id="Qwen3-14B-Q4_K_M.gguf",
        model_type="chat",
        extra_args="--ctx-size 65536",
    ),
    ModelInfo(
        model_id="qwen3:30b-a3b-q4_K_M",
        backend="llamacpp",
        backend_id="Qwen3-30B-A3B-Q4_K_M.gguf",
        model_type="chat",
        extra_args="--ctx-size 65536",
    ),
    ModelInfo(
        model_id="qwen3:30b-a3b-instruct-2507-q4_K_M",
        backend="llamacpp",
        backend_id="Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf",
        model_type="chat",
        extra_args="--ctx-size 65536",
    ),
    ModelInfo(
        model_id="qwen3:32b-q4_K_M",
        backend="llamacpp",
        backend_id="Qwen3-32B-Q4_K_M.gguf",
        model_type="chat",
        extra_args="--ctx-size 65536",
    ),
    ModelInfo(
        model_id="qwen3.5:27b-q4_K_M",
        backend="llamacpp",
        backend_id="Qwen3.5-27B-Q4_K_M.gguf",
        model_type="chat",
        extra_args="--ctx-size 65536",
    ),
    ModelInfo(
        model_id="qwen3.5:35b-a3b-q4_K_M",
        backend="llamacpp",
        backend_id="Qwen3.5-35B-A3B-Q4_K_M.gguf",
        model_type="chat",
        extra_args="--ctx-size 65536",
    ),
    ModelInfo(
        model_id="qwen3.6:35b-a3b-ud-q4_K_M",
        backend="llamacpp",
        backend_id="Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
        model_type="chat",
        extra_args="--ctx-size 65536",
    ),
    ModelInfo(
        model_id="qwen3-next:80b-a3b-instruct-q4_K_M",
        backend="llamacpp",
        backend_id="Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf",
        model_type="chat",
        extra_args="--ctx-size 65536",
    ),
    ModelInfo(
        model_id="qwen3-next:80b-a3b-thinking-q4_K_M",
        backend="llamacpp",
        backend_id="Qwen3-Next-80B-A3B-Thinking-Q4_K_M.gguf",
        model_type="chat",
        extra_args="--ctx-size 65536",
    ),
    ModelInfo(
        model_id="qwen3.5-122b-a10b-q4_K_M",
        backend="llamacpp",
        backend_id="Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf",
        model_type="chat",
        extra_args="--ctx-size 65536",
    ),
    ModelInfo(
        model_id="qwen3.5:122b-a10b-q4_K_M",
        backend="llamacpp",
        backend_id="Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf",
        model_type="chat",
        extra_args="--ctx-size 65536",
    ),
    ModelInfo(
        model_id="GLM-4.5-Air-q4_K_M",
        backend="llamacpp",
        backend_id="GLM-4.5-Air-Q4_K_M-00001-of-00002.gguf",
        model_type="chat",
        extra_args="--ctx-size 65536",
    ),
    ModelInfo(
        model_id="MiniMax-M2.5-q4_K_M",
        backend="llamacpp",
        backend_id="MiniMax-M2.5-Q4_K_M-00001-of-00004.gguf",
        model_type="chat",
        extra_args="--ctx-size 65536",
    ),
    ModelInfo(
        model_id="gemma4:26b-a4b-it-q8_0",
        backend="llamacpp",
        backend_id="gemma-4-26B-A4B-it-UD-Q8_K_XL.gguf",
        model_type="chat",
        extra_args="--ctx-size 65536",
        hf_repo="google/gemma-4-26B-A4B",
    ),
    ModelInfo(
        model_id="gemma4:31b-it-q8_0",
        backend="llamacpp",
        backend_id="gemma-4-31B-it-Q8_0.gguf",
        model_type="chat",
        extra_args="--ctx-size 65536",
        modalities=("text", "image"),
        mmproj_id="mmproj-BF16.gguf",
        hf_repo="google/gemma-4-31B-it",
    ),

    # ---- CHAT MODELS (large → vllm (currently unused)) ----
    ModelInfo(
        model_id="llama3.3:70b-instruct-q8_0",
        backend="llamacpp",
        #backend_id="Llama-3.3-70B-Instruct-Q8_0-00001-of-00002.gguf",
        backend_id="Llama-3.3-70B-Instruct-Q4_K_M.gguf",
        model_type="chat",
        extra_args="--ctx-size 65536",
    ),
    ModelInfo(
        model_id="llama3.3:70b-instruct-q4_K_M",
        backend="llamacpp",
        backend_id="Llama-3.3-70B-Instruct-Q4_K_M.gguf",
        model_type="chat",
        extra_args="--ctx-size 65536",
    ),
    ModelInfo(
        model_id="mistral-large:123b",
        backend="llamacpp",
        backend_id="Mistral-Large-Instruct-2411-Q4_K_M-00001-of-00002.gguf",
        model_type="chat",
        extra_args="--ctx-size 65536",
    ),
    ModelInfo(
        model_id="qwen3:235b-a22b",
        backend="llamacpp",
        backend_id="Qwen_Qwen3-235B-A22B-Instruct-2507-Q4_K_M-00001-of-00004.gguf",
        model_type="chat",
        extra_args="--ctx-size 32768",
    ),
    ModelInfo(
        model_id="qwen3:235b-a22b-instruct-2507-q4_K_M",
        backend="llamacpp",
        backend_id="Qwen3-235B-A22B-Instruct-2507-Q4_K_M-00001-of-00003.gguf",
        model_type="chat",
        extra_args="--ctx-size 32768",
    ),
    ModelInfo(
        model_id="qwen3:235b-a22b-thinking-2507-q4_K_M",
        backend="llamacpp",
        backend_id="Qwen3-235B-A22B-Thinking-2507-Q4_K_M-00001-of-00003.gguf",
        model_type="chat",
        extra_args="--ctx-size 32768",
    ),
    ModelInfo(
        model_id="command-a:111b",
        backend="llamacpp",
        backend_id="CohereForAI_c4ai-command-a-03-2025-Q4_K_M-00001-of-00002.gguf",
        model_type="chat",
        extra_args="--ctx-size 65536",
    ),
    ModelInfo(
        model_id="gpt-oss:120b",
        backend="llamacpp",
        backend_id="gpt-oss-120b-Q4_K_M-00001-of-00002.gguf",
        #backend_id="huizimao_gpt-oss-120b-uncensored-bf16-MXFP4_MOE-00001-of-00002.gguf",
        model_type="chat",
        extra_args="--ctx-size 65536",
    ),

    # ---- EMBEDDING MODELS ----
    ModelInfo(
        model_id="qwen3-embedding:8b-fp16",
        backend="llamacpp",
        backend_id="qwen3-embedding-8b-fp16.gguf",
        model_type="embedding",
        extra_args="--embedding --ctx-size 8192",
    ),
    ModelInfo(
        model_id="qwen3-embedding:8b-q8_0",
        backend="llamacpp",
        backend_id="qwen3-embedding-8b-Q8_0.gguf",
        model_type="embedding",
        extra_args="--embedding --ctx-size 8192 --cache-ram 0",
        #parallel_slots=1,
    ),
    ModelInfo(
        model_id="qwen3-embedding:4b-q8_0",
        backend="llamacpp",
        backend_id="qwen3-embedding-4b-Q8_0.gguf",
        model_type="embedding",
        extra_args="--embedding --ctx-size 8192",
    ),
    ModelInfo(
        model_id="qwen3-embedding:0.6b",
        backend="llamacpp",
        backend_id="qwen3-embedding-0.6b.gguf",
        model_type="embedding",
        extra_args="--embedding --ctx-size 8192",
    ),
    ModelInfo(
        model_id="qwen3-embedding:4b",
        backend="llamacpp",
        backend_id="qwen3-embedding-4b.gguf",
        model_type="embedding",
        extra_args="--embedding --ctx-size 8192",
    ),
    ModelInfo(
        model_id="qwen3-embedding:8b",
        backend="llamacpp",
        backend_id="qwen3-embedding-8b.gguf",
        model_type="embedding",
        extra_args="--embedding --ctx-size 8192",
    ),
    ModelInfo(
        model_id="nomic-embed-text:latest",
        backend="llamacpp",
        backend_id="nomic-embed-text-v1.5.f16.gguf",
        model_type="embedding",
        extra_args="--embedding --ctx-size 8192",
    ),
    ModelInfo(
        model_id="embeddinggemma:latest",
        backend="llamacpp",
        backend_id="embeddinggemma-latest.gguf",
        model_type="embedding",
        extra_args="--embedding --ctx-size 8192",
    ),

    # ---- RERANKER MODELS ----
    ModelInfo(
        model_id="hf.co/jinaai/jina-reranker-v3-GGUF:BF16",
        backend="llamacpp",
        backend_id="jina-reranker-v3-BF16.gguf",
        model_type="reranker",
        extra_args="--reranking",
    ),
    ModelInfo(
        model_id="dengcao/Qwen3-Reranker-8B:F16",
        backend="llamacpp",
        backend_id="Qwen3-Reranker-8B-F16.gguf",
        model_type="reranker",
        extra_args="--reranking",
    ),
    ModelInfo(
        model_id="dengcao/Qwen3-Reranker-0.6B:F16",
        backend="llamacpp",
        backend_id="Qwen3-Reranker-0.6B-F16.gguf",
        model_type="reranker",
        extra_args="--reranking",
    ),
    ModelInfo(
        model_id="dengcao/Qwen3-Reranker-8B:Q8_0",
        backend="llamacpp",
        backend_id="Qwen3-Reranker-8B-Q8_0.gguf",
        model_type="reranker",
        extra_args="--reranking",
    ),
    ModelInfo(
        model_id="hf.co/mradermacher/colbertv2.0-GGUF:F16",
        backend="llamacpp",
        backend_id="colbertv2.0-F16.gguf",
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
