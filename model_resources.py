import os
import re
import shlex
import sqlite3
import time
from functools import lru_cache
from typing import Optional

from gguf import GGUFReader, GGUFValueType

import config as runtime_config
from config import (
    GGUF_MODEL_DIR,
    GPU_HEADROOM_GB,
    MAX_PARALLEL_SLOTS,
    NUM_GPUS,
    TOTAL_USABLE_MEMORY_GB,
)


MEMORY_BUDGET_GB = TOTAL_USABLE_MEMORY_GB - (GPU_HEADROOM_GB * NUM_GPUS)
KV_CACHE_DTYPE_BYTES = 2
MODEL_MEMORY_SAFETY_FACTOR = 1.10
SLOT_MEMORY_SAFETY_FACTOR = 1.15
MODEL_INFO_DB_PATH = getattr(runtime_config, "MODEL_INFO_DB_PATH", None)


def _model_info_db_path() -> str:
    if MODEL_INFO_DB_PATH:
        return os.path.abspath(MODEL_INFO_DB_PATH)

    config_file = getattr(runtime_config, "__file__", None)
    if config_file:
        config_dir = os.path.dirname(os.path.abspath(config_file))
        return os.path.join(config_dir, "model_info.db")

    return os.path.abspath("model_info.db")


def model_info_db_path() -> str:
    return _model_info_db_path()


def _init_model_info_db(conn: sqlite3.Connection):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS model_vram_measurements (
            model_id TEXT PRIMARY KEY,
            backend TEXT NOT NULL,
            backend_id TEXT NOT NULL,
            model_type TEXT NOT NULL,
            extra_args TEXT NOT NULL,
            base_memory_gb REAL NOT NULL,
            slot_memory_gb REAL NOT NULL,
            parallel_1_memory_gb REAL NOT NULL,
            parallel_2_memory_gb REAL,
            measured_at REAL NOT NULL,
            source TEXT NOT NULL
        )
        """
    )
    conn.commit()


def _model_signature(model_info) -> tuple[str, str, str, str, str]:
    return (
        getattr(model_info, "model_id", ""),
        getattr(model_info, "backend", ""),
        getattr(model_info, "backend_id", ""),
        getattr(model_info, "model_type", ""),
        getattr(model_info, "extra_args", "") or "",
    )


def get_measured_model_memory(model_info) -> Optional[dict]:
    model_id, backend, backend_id, model_type, extra_args = _model_signature(model_info)
    if not model_id:
        return None

    try:
        with sqlite3.connect(_model_info_db_path()) as conn:
            _init_model_info_db(conn)
            row = conn.execute(
                """
                SELECT backend, backend_id, model_type, extra_args,
                       base_memory_gb, slot_memory_gb,
                       parallel_1_memory_gb, parallel_2_memory_gb,
                       measured_at, source
                FROM model_vram_measurements
                WHERE model_id = ?
                """,
                (model_id,),
            ).fetchone()
    except sqlite3.Error as exc:
        print(f"[MemoryBudget] Could not read {_model_info_db_path()}: {exc}")
        return None

    if row is None:
        return None

    row_backend, row_backend_id, row_model_type, row_extra_args = row[:4]
    if (
        row_backend != backend
        or row_backend_id != backend_id
        or row_model_type != model_type
        or row_extra_args != extra_args
    ):
        return None

    return {
        "base_memory_gb": float(row[4]),
        "slot_memory_gb": float(row[5]),
        "parallel_1_memory_gb": float(row[6]),
        "parallel_2_memory_gb": None if row[7] is None else float(row[7]),
        "measured_at": float(row[8]),
        "source": row[9],
    }


def has_measured_model_memory(model_info) -> bool:
    return get_measured_model_memory(model_info) is not None


def save_measured_model_memory(
    model_info,
    base_memory_gb: float,
    slot_memory_gb: float,
    parallel_1_memory_gb: float,
    parallel_2_memory_gb: Optional[float],
    source: str = "measure_vram.py",
):
    model_id, backend, backend_id, model_type, extra_args = _model_signature(model_info)
    if not model_id:
        raise ValueError("model_info has no model_id")

    with sqlite3.connect(_model_info_db_path()) as conn:
        _init_model_info_db(conn)
        conn.execute(
            """
            INSERT INTO model_vram_measurements (
                model_id, backend, backend_id, model_type, extra_args,
                base_memory_gb, slot_memory_gb,
                parallel_1_memory_gb, parallel_2_memory_gb,
                measured_at, source
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(model_id) DO UPDATE SET
                backend = excluded.backend,
                backend_id = excluded.backend_id,
                model_type = excluded.model_type,
                extra_args = excluded.extra_args,
                base_memory_gb = excluded.base_memory_gb,
                slot_memory_gb = excluded.slot_memory_gb,
                parallel_1_memory_gb = excluded.parallel_1_memory_gb,
                parallel_2_memory_gb = excluded.parallel_2_memory_gb,
                measured_at = excluded.measured_at,
                source = excluded.source
            """,
            (
                model_id,
                backend,
                backend_id,
                model_type,
                extra_args,
                float(base_memory_gb),
                float(slot_memory_gb),
                float(parallel_1_memory_gb),
                None if parallel_2_memory_gb is None else float(parallel_2_memory_gb),
                time.time(),
                source,
            ),
        )
        conn.commit()


def _field_value(reader: GGUFReader, key: str):
    field = reader.fields.get(key)
    if field is None or not field.data:
        return None
    value = field.parts[field.data[0]]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        if field.types and field.types[0] == GGUFValueType.STRING:
            try:
                return bytes(value).decode("utf-8")
            except (TypeError, ValueError, UnicodeDecodeError):
                return value
        if len(value) == 1:
            return value[0]
    return value


def parse_ctx_size(extra_args: str) -> Optional[int]:
    args = shlex.split(extra_args or "")
    i = 0
    while i < len(args):
        arg = args[i]
        flag, sep, value = arg.partition("=")
        if flag in {"-c", "--ctx-size", "--ctx_size"}:
            if sep:
                try:
                    return int(value)
                except ValueError:
                    return None
            if i + 1 < len(args):
                try:
                    return int(args[i + 1])
                except ValueError:
                    return None
        i += 1
    return None


def resolve_gguf_path(model_info) -> Optional[str]:
    backend_id = getattr(model_info, "backend_id", "")
    if not backend_id.lower().endswith(".gguf"):
        return None
    if os.path.isabs(backend_id):
        return backend_id
    return os.path.join(GGUF_MODEL_DIR, backend_id)


def _gguf_part_paths(path: str) -> list[str]:
    match = re.search(r"-00001-of-(\d+)(\.gguf)$", path)
    if not match:
        return [path]

    count = int(match.group(1))
    width = len(match.group(1))
    return [
        re.sub(
            r"-00001-of-\d+(\.gguf)$",
            f"-{idx:05d}-of-{count:0{width}d}\\1",
            path,
        )
        for idx in range(1, count + 1)
    ]


@lru_cache(maxsize=256)
def _gguf_metadata(path: str) -> dict:
    reader = GGUFReader(path)
    architecture = _field_value(reader, "general.architecture")
    if not architecture:
        return {}

    prefix = str(architecture)
    return {
        "architecture": prefix,
        "block_count": _field_value(reader, f"{prefix}.block_count"),
        "context_length": _field_value(reader, f"{prefix}.context_length"),
        "embedding_length": _field_value(reader, f"{prefix}.embedding_length"),
        "head_count": _field_value(reader, f"{prefix}.attention.head_count"),
        "head_count_kv": _field_value(reader, f"{prefix}.attention.head_count_kv"),
        "key_length": _field_value(reader, f"{prefix}.attention.key_length"),
        "value_length": _field_value(reader, f"{prefix}.attention.value_length"),
    }


@lru_cache(maxsize=256)
def _gguf_tensor_bytes(path: str) -> int:
    reader = GGUFReader(path)
    return sum(int(tensor.n_bytes) for tensor in reader.tensors)


def estimate_base_memory_gb(model_info) -> float:
    measured = get_measured_model_memory(model_info)
    if measured is not None:
        return measured["base_memory_gb"]

    path = resolve_gguf_path(model_info)
    if not path:
        return 0.0

    paths = _gguf_part_paths(path)
    if not all(os.path.exists(part_path) for part_path in paths):
        return 0.0

    try:
        total_bytes = sum(_gguf_tensor_bytes(part_path) for part_path in paths)
        mmproj_id = getattr(model_info, "mmproj_id", None)
        if mmproj_id:
            mmproj_path = mmproj_id if os.path.isabs(mmproj_id) else os.path.join(GGUF_MODEL_DIR, mmproj_id)
            if os.path.exists(mmproj_path):
                total_bytes += _gguf_tensor_bytes(mmproj_path)
        return total_bytes / (1024 ** 3) * MODEL_MEMORY_SAFETY_FACTOR
    except Exception as exc:
        print(f"[MemoryBudget] Could not estimate base memory from GGUF for {path}: {exc}")
        return 0.0


def estimate_slot_memory_gb(model_info) -> float:
    override = float(getattr(model_info, "slot_memory_gb", 0.0) or 0.0)
    if override > 0:
        return override

    measured = get_measured_model_memory(model_info)
    if measured is not None:
        return measured["slot_memory_gb"]

    path = resolve_gguf_path(model_info)
    if not path or not os.path.exists(path):
        return 0.0

    try:
        meta = _gguf_metadata(path)
    except Exception as exc:
        print(f"[MemoryBudget] Could not read GGUF metadata for {path}: {exc}")
        return 0.0

    n_layer = meta.get("block_count")
    ctx_size = getattr(model_info, "ctx_size", None) or parse_ctx_size(getattr(model_info, "extra_args", ""))
    if ctx_size is None:
        ctx_size = meta.get("context_length")

    key_length = meta.get("key_length")
    value_length = meta.get("value_length")
    if key_length is None or value_length is None:
        embedding_length = meta.get("embedding_length")
        head_count = meta.get("head_count")
        head_count_kv = meta.get("head_count_kv") or head_count
        if embedding_length and head_count and head_count_kv:
            head_dim = int(embedding_length) / int(head_count)
            key_length = value_length = head_dim * int(head_count_kv)

    if not n_layer or not ctx_size or not key_length or not value_length:
        return 0.0

    bytes_total = (
        int(ctx_size)
        * int(n_layer)
        * (float(key_length) + float(value_length))
        * KV_CACHE_DTYPE_BYTES
    )
    return bytes_total / (1024 ** 3) * SLOT_MEMORY_SAFETY_FACTOR


def effective_parallel_slots(model_info, budget_gb: float = MEMORY_BUDGET_GB) -> int:
    configured = int(getattr(model_info, "parallel_slots", 0) or 0)
    if configured > 0:
        return max(1, min(MAX_PARALLEL_SLOTS, configured))

    base_memory = estimate_base_memory_gb(model_info)
    slot_memory = estimate_slot_memory_gb(model_info)
    if slot_memory <= 0 or base_memory >= budget_gb:
        return 1

    slots = int((budget_gb - base_memory) // slot_memory)
    return max(1, min(MAX_PARALLEL_SLOTS, slots))


def effective_model_memory_gb(model_info, budget_gb: float = MEMORY_BUDGET_GB) -> float:
    slots = effective_parallel_slots(model_info, budget_gb=budget_gb)
    return estimate_base_memory_gb(model_info) + slots * estimate_slot_memory_gb(model_info)
