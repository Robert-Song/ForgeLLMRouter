"""
proxy_v6/db.py
Token management database — extracted from openai_proxy_vLLM_v3.py.
Uses SQLite for API key storage and usage tracking.
(previous code)
"""

import sqlite3
import secrets
from typing import Optional, List, Dict
from config import DB_PATH


def init_db():
    """Initialize SQLite database for token tracking."""
    conn = sqlite3.connect(DB_PATH, detect_types=0)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS api_keys (
            key TEXT PRIMARY KEY,
            lifetime_limit INTEGER NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS usage_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            api_key TEXT NOT NULL,
            tokens_used INTEGER NOT NULL,
            request_tokens INTEGER,
            response_tokens INTEGER,
            timestamp TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (api_key) REFERENCES api_keys(key)
        )
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_usage_key
        ON usage_logs(api_key)
    """)

    conn.commit()
    conn.close()


def get_total_usage(api_key: str) -> int:
    """Get total tokens used lifetime for an API key."""
    conn = sqlite3.connect(DB_PATH, detect_types=0)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT COALESCE(SUM(tokens_used), 0)
        FROM usage_logs
        WHERE api_key = ?
    """, (api_key,))

    usage = cursor.fetchone()[0]
    conn.close()
    return usage


def get_key_limit(api_key: str) -> Optional[int]:
    """Get the lifetime token limit for an API key."""
    conn = sqlite3.connect(DB_PATH, detect_types=0)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT lifetime_limit
        FROM api_keys
        WHERE key = ?
    """, (api_key,))

    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None


def log_usage(api_key: str, tokens_used: int, request_tokens: int = 0, response_tokens: int = 0):
    """Log token usage for an API key."""
    conn = sqlite3.connect(DB_PATH, detect_types=0)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO usage_logs (api_key, tokens_used, request_tokens, response_tokens)
        VALUES (?, ?, ?, ?)
    """, (api_key, tokens_used, request_tokens, response_tokens))

    conn.commit()
    conn.close()


def generate_api_key() -> str:
    """Generate a secure random API key."""
    return f"aiforge-{secrets.token_urlsafe(32)}"


def add_api_key(api_key: str, lifetime_limit: int) -> bool:
    """Add a new API key with lifetime token limit."""
    conn = sqlite3.connect(DB_PATH, detect_types=0)
    cursor = conn.cursor()

    try:
        cursor.execute("""
            INSERT INTO api_keys (key, lifetime_limit)
            VALUES (?, ?)
        """, (api_key, lifetime_limit))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def remove_api_key(api_key: str) -> bool:
    """Remove an API key and its usage logs."""
    conn = sqlite3.connect(DB_PATH, detect_types=0)
    cursor = conn.cursor()

    cursor.execute("SELECT key FROM api_keys WHERE key = ?", (api_key,))
    if not cursor.fetchone():
        conn.close()
        return False

    cursor.execute("DELETE FROM usage_logs WHERE api_key = ?", (api_key,))
    cursor.execute("DELETE FROM api_keys WHERE key = ?", (api_key,))

    conn.commit()
    conn.close()
    return True


def update_key_limit(api_key: str, new_limit: int) -> bool:
    """Update the lifetime token limit for an API key."""
    conn = sqlite3.connect(DB_PATH, detect_types=0)
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE api_keys
        SET lifetime_limit = ?
        WHERE key = ?
    """, (new_limit, api_key))

    rows_affected = cursor.rowcount
    conn.commit()
    conn.close()
    return rows_affected > 0


def reset_usage(api_key: str) -> bool:
    """Reset usage for an API key (delete all usage logs)."""
    conn = sqlite3.connect(DB_PATH, detect_types=0)
    cursor = conn.cursor()

    cursor.execute("SELECT key FROM api_keys WHERE key = ?", (api_key,))
    if not cursor.fetchone():
        conn.close()
        return False

    cursor.execute("DELETE FROM usage_logs WHERE api_key = ?", (api_key,))

    conn.commit()
    conn.close()
    return True


def list_api_keys() -> List[Dict]:
    """List all API keys with their limits and current usage."""
    conn = sqlite3.connect(DB_PATH, detect_types=0)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT key, lifetime_limit, created_at
        FROM api_keys
        ORDER BY created_at DESC
    """)

    keys = cursor.fetchall()
    conn.close()

    result = []
    for key, limit, created_at in keys:
        usage = get_total_usage(key)
        result.append({
            'key': key,
            'lifetime_limit': limit,
            'total_usage': usage,
            'remaining': max(0, limit - usage),
            'created_at': created_at
        })

    return result
