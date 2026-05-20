"""
proxy_v6/proxy_server.py
FastAPI proxy
mostly extracted from proxy_v3
"""

from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import StreamingResponse, JSONResponse
from contextlib import asynccontextmanager
from typing import Optional
import asyncio
import httpx
import json

import logging
import time

logger = logging.getLogger("uvicorn.error")

from db import init_db, get_key_limit, get_total_usage, log_usage
from models import MODEL_REGISTRY, list_all_model_ids
from request_queue import request_queue
from process_manager import process_manager


# ============================================================================
#  FastAPI Application
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    init_db()
    request_queue.set_forward_fn(_forward_to_model)
    await request_queue.start()
    print("[Proxy] request_queue started. Process manager ready.")
    yield
    await request_queue.stop()
    await process_manager.cleanup()

app = FastAPI(title="OpenAI Proxy v6", lifespan=lifespan)

@app.middleware("http")
async def custom_access_log(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    # HTTP status code might literally be an error code but uvicorn just says OK
    status_text = "OK" if response.status_code < 400 else "ERROR"
    model = getattr(request.state, "model", "N/A")
    
    client_host = request.client.host if request.client else "unknown"
    client_port = request.client.port if request.client else 0
    
    logger.info(f'{client_host}:{client_port} - "{request.method} {request.url.path} HTTP/{request.scope.get("http_version", "1.1")}" {response.status_code} {status_text} - Model: {model} ({process_time:.2f}s)')
    
    return response


# ============================================================================
#  Helpers
# ============================================================================

def estimate_tokens(text: str) -> int:
    """Rough estimation of tokens (~1 token per 4 characters)."""
    return len(text) // 4


IMAGE_TOKEN_ESTIMATE = 1024


def _is_image_content_part(part) -> bool:
    """Return True for OpenAI-style image content parts."""
    if not isinstance(part, dict):
        return False

    part_type = part.get("type")
    return part_type in {"image_url", "input_image"} or "image_url" in part


def _messages_contain_image(messages) -> bool:
    """Detect whether chat messages include image input."""
    if not isinstance(messages, list):
        return False

    for message in messages:
        if not isinstance(message, dict):
            continue

        content = message.get("content")
        if not isinstance(content, list):
            continue

        if any(_is_image_content_part(part) for part in content):
            return True

    return False


def _estimate_chat_quota_tokens(messages) -> int:
    """Estimate chat quota while avoiding counting base64 image payload bytes."""
    text_chunks = []
    image_count = 0

    if not isinstance(messages, list):
        messages = [messages]

    for message in messages:
        content = message.get("content", "") if isinstance(message, dict) else message

        if isinstance(content, str):
            text_chunks.append(content)
            continue

        if not isinstance(content, list):
            continue

        for part in content:
            if isinstance(part, str):
                text_chunks.append(part)
            elif isinstance(part, dict):
                if _is_image_content_part(part):
                    image_count += 1
                elif "text" in part:
                    text_chunks.append(str(part.get("text", "")))

    text_estimate = estimate_tokens(" ".join(text_chunks)) * 2
    image_estimate = image_count * IMAGE_TOKEN_ESTIMATE
    return text_estimate + image_estimate


def _extract_api_key(authorization: Optional[str]) -> str:
    """Extract and validate API key from the Authorization header."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing authorization header")
    api_key = authorization.replace("Bearer ", "")
    if get_key_limit(api_key) is None:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key


def _check_quota(api_key: str, estimated_tokens: int):
    """Check if the API key has enough quota for the estimated usage."""
    lifetime_limit = get_key_limit(api_key)
    current_usage = get_total_usage(api_key)
    if current_usage + estimated_tokens > lifetime_limit:
        remaining = max(0, lifetime_limit - current_usage)
        raise HTTPException(
            status_code=429,
            detail=(
                f"Token limit exceeded (estimated). "
                f"Used: {current_usage}/{lifetime_limit} tokens (lifetime). "
                f"Remaining: {remaining}"
            ),
        )


# ============================================================================
#  Forward to model backend
# ============================================================================

async def _forward_to_model(model_id: str, port: int, request_body: dict, api_key: str):
    """
    Forward a non-streaming request to the specific model backend and return the response JSON.
    This function is used by the request_queue's batch processor.
    """
    max_retries = 10
    base_delay = 3.0  # seconds

    async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=3600.0) as client:
        # Determine endpoint based on model type
        model = MODEL_REGISTRY.get(model_id)
        if model and model.model_type == "embedding":
            endpoint = "/v1/embeddings"
        elif model and model.model_type == "reranker":
            endpoint = "/v1/rerank"
        else:
            endpoint = "/v1/chat/completions"

        for attempt in range(max_retries + 1):
            response = await client.post(endpoint, json=request_body)

            if response.status_code == 502 and attempt < max_retries:
                delay = min(base_delay * (1.5 ** attempt), 30.0)
                print(
                    f"[Forward] 502 from model backend for '{model_id}' "
                    f"(attempt {attempt + 1}/{max_retries + 1}), "
                    f"retrying in {delay:.1f}s..."
                )
                await asyncio.sleep(delay)
                continue

            if response.status_code != 200:
                request_size = len(json.dumps(request_body))
                print(f"[Forward] Error from model backend for '{model_id}': status={response.status_code}, request_size={request_size}, response={response.text[:500]}")
            
            response.raise_for_status()
            return response.json()


async def _stream_from_model(
    client: httpx.AsyncClient,
    port: int,
    request_body: dict,
    api_key: str,
):
    """Stream response from a model backend and log usage from the final chunk."""
    prompt_tokens = 0
    completion_tokens = 0

    try:
        async with client.stream(
            "POST",
            f"http://127.0.0.1:{port}/v1/chat/completions",
            json=request_body,
            timeout=3600.0,
        ) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if line:
                    yield f"{line}\n\n"
                    # Parse usage from final SSE chunk
                    if line.startswith("data: ") and line != "data: [DONE]":
                        try:
                            chunk_data = json.loads(line[6:])
                            if "usage" in chunk_data and chunk_data["usage"]:
                                usage = chunk_data["usage"]
                                prompt_tokens = usage.get("prompt_tokens", 0)
                                completion_tokens = usage.get("completion_tokens", 0)
                        except json.JSONDecodeError:
                            continue
    finally:
        total_tokens = prompt_tokens + completion_tokens
        if total_tokens > 0:
            log_usage(api_key, total_tokens, prompt_tokens, completion_tokens)


# ============================================================================
#  API Endpoints
# ============================================================================

@app.post("/v1/admin/unload_all")
async def admin_unload_all(authorization: Optional[str] = Header(None)):
    """Admin endpoint to surgically terminate all currently loaded models and free VRAM."""
    api_key = _extract_api_key(authorization)
    # Optional: ensure api_key belongs to an admin user
    
    loaded_models = await request_queue.unload_all(force=True)
        
    return JSONResponse(content={"status": "success", "unloaded": loaded_models})

@app.delete("/v1/admin/unload/{model_id}")
async def admin_unload_model(model_id: str, authorization: Optional[str] = Header(None)):
    """Admin endpoint to unload a specific model by ID."""
    api_key = _extract_api_key(authorization)
    
    if model_id not in process_manager.active_processes:
        return JSONResponse(status_code=404, content={"status": "error", "detail": "Model not loaded"})
        
    await request_queue.unload_model(model_id, force=True)
    return JSONResponse(content={"status": "success", "unloaded": model_id})

@app.get("/v1/admin/status")
async def admin_status(authorization: Optional[str] = Header(None)):
    """Inspectable proxy status: queue, inflight requests, loaded models, memory, and llama.cpp slots."""
    api_key = _extract_api_key(authorization)
    return JSONResponse(content=await request_queue.snapshot(include_backend=True))

@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """Proxy endpoint for chat completions → model backend."""

    api_key = _extract_api_key(authorization)
    body = await request.json()

    request.state.model = body.get("model", "") #for model logging

    # Validate model exists in our registry
    requested_model = body.get("model", "")
    if requested_model not in MODEL_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Model '{requested_model}' not found in registry. "
                f"Available: {list_all_model_ids()}"
            ),
        )

    model_info = MODEL_REGISTRY[requested_model]
    messages = body.get("messages", [])
    has_image_input = _messages_contain_image(messages)
    if has_image_input and "image" not in model_info.modalities:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Model '{requested_model}' does not support image input. "
                "Use a model with image modality support."
            ),
        )

    # Pre-flight quota check. Count text normally and charge a fixed estimate
    # per image instead of counting base64 payload bytes as text tokens.
    estimated = _estimate_chat_quota_tokens(messages)
    _check_quota(api_key, estimated)

    # Streaming path: hold a scheduler slot until the stream is closed.
    if body.get("stream", False):
        try:
            lease = request_queue.model_slot(requested_model, endpoint_type="chat", stream=True)
            port = await lease.__aenter__()
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
        client = httpx.AsyncClient(timeout=3600.0)

        async def stream_with_cleanup():
            exc = None
            try:
                async for chunk in _stream_from_model(client, port, body, api_key):
                    yield chunk
            except Exception as e:
                exc = e
                raise
            finally:
                await client.aclose()
                if exc:
                    await lease.__aexit__(type(exc), exc, exc.__traceback__)
                else:
                    await lease.__aexit__(None, None, None)

        return StreamingResponse(
            stream_with_cleanup(),
            media_type="text/event-stream",
        )

    # Non-streaming path: use the batch request_queue
    try:
        response_data = await request_queue.enqueue(
            model_id=requested_model,
            request_body=body,
            api_key=api_key,
        )

        # Log usage
        if "usage" in response_data:
            usage_data = response_data["usage"]
            log_usage(
                api_key,
                usage_data.get("total_tokens", 0),
                usage_data.get("prompt_tokens", 0),
                usage_data.get("completion_tokens", 0),
            )

        return JSONResponse(content=response_data)

    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail="Unable to connect to model backend. Is it running?",
        )
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="Request to model backend timed out.",
        )
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Model backend returned error {e.response.status_code}: {e.response.text[:200]}",
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/v1/embeddings")
async def create_embeddings(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """Proxy endpoint for embeddings → model backend."""

    api_key = _extract_api_key(authorization)
    body = await request.json()

    request.state.model = body.get("model", "") #for model logging

    # Extract input text
    input_data = body.get("input", "")
    if isinstance(input_data, str):
        input_texts = [input_data]
    elif isinstance(input_data, list):
        input_texts = input_data
    else:
        raise HTTPException(status_code=400, detail="Invalid input format")

    # Quota check
    total_input_text = " ".join(str(t) for t in input_texts)
    estimated = estimate_tokens(total_input_text)
    _check_quota(api_key, estimated)

    model = body.get("model", "nomic-embed-text:latest")

    # Forward to model backend
    try:
        async with request_queue.model_slot(model, endpoint_type="embedding") as port:
            async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=3600.0) as client:
                response = await client.post("/v1/embeddings", json=body)
                response.raise_for_status()
                response_data = response.json()

                # Log usage
                if "usage" in response_data:
                    usage_data = response_data["usage"]
                    log_usage(
                        api_key,
                        usage_data.get("total_tokens", estimated),
                        usage_data.get("prompt_tokens", estimated),
                        0,
                    )
                else:
                    log_usage(api_key, estimated, estimated, 0)

                return JSONResponse(content=response_data)

    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail="Unable to connect to model backend. Is it running?",
        )
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Request to model backend timed out.")
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Model backend error {e.response.status_code}: {e.response.text[:200]}",
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/v1/rerank")
async def rerank(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """Proxy endpoint for reranking → model backend."""

    api_key = _extract_api_key(authorization)
    body = await request.json()

    request.state.model = body.get("model", "") #for model logging

    # Quota check on query + documents
    query = body.get("query", "")
    documents = body.get("documents", [])
    text_blob = query + " ".join(str(d) for d in documents)
    estimated = estimate_tokens(text_blob)
    _check_quota(api_key, estimated)

    model = body.get("model")
    if not model or model not in MODEL_REGISTRY:
        raise HTTPException(status_code=400, detail="Valid model must be specified for rerank")

    # Forward to model backend
    try:
        async with request_queue.model_slot(model, endpoint_type="rerank") as port:
            async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=3600.0) as client:
                response = await client.post("/v1/rerank", json=body)
                response.raise_for_status()
                response_data = response.json()
                log_usage(api_key, estimated, estimated, 0)
                return JSONResponse(content=response_data)

    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail="Unable to connect to model backend. Is it running?",
        )
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Model backend error {e.response.status_code}: {e.response.text[:200]}",
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/v1/models")
async def list_models(authorization: Optional[str] = Header(None)):
    """List available models from the proxy registry."""

    api_key = _extract_api_key(authorization)

    from datetime import datetime
    openai_models = {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": int(datetime.now().timestamp()),
                "owned_by": "aiforge",
            }
            for model_id in MODEL_REGISTRY
        ],
    }
    return JSONResponse(content=openai_models)


@app.get("/usage/{api_key}")
async def get_usage(api_key: str):
    """Get current usage for an API key."""
    lifetime_limit = get_key_limit(api_key)
    if lifetime_limit is None:
        raise HTTPException(status_code=404, detail="API key not found")

    total_usage = get_total_usage(api_key)

    return {
        "api_key": api_key,
        "lifetime_limit": lifetime_limit,
        "total_used": total_usage,
        "remaining": max(0, lifetime_limit - total_usage),
        "usage_percentage": (total_usage / lifetime_limit * 100) if lifetime_limit > 0 else 0,
    }


@app.get("/v1/usage/{api_key}")
async def get_usage_v1(api_key: str):
    """Get current usage for an API key (OpenAI-compatible path)."""
    return await get_usage(api_key)


@app.get("/health")
async def health():
    """Health check for the proxy itself."""
    return {"status": "ok"}
