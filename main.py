"""
proxy_v6/main.py
Entry point for the proxy v6 system.

Usage:
  python main.py                  # Start proxy server
  python main.py --config-dir DIR # Start with DIR/config.py
  python main.py --manage         # Key management CLI
"""

import argparse
import atexit
import copy
import importlib
import logging
import os
import sys
import time


def _parse_config_dir(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--config-dir",
        default=os.environ.get("FORGE_CONFIG_DIR"),
        help="Directory containing an alternate config.py",
    )
    args, _ = parser.parse_known_args(argv)
    return args.config_dir


def _load_config(config_dir: str | None):
    if config_dir:
        resolved_dir = os.path.abspath(os.path.expanduser(config_dir))
        config_path = os.path.join(resolved_dir, "config.py")
        if not os.path.isfile(config_path):
            raise SystemExit(f"--config-dir must contain config.py: {resolved_dir}")
        sys.path.insert(0, resolved_dir)

    return importlib.import_module("config")


def main():
    config = _load_config(_parse_config_dir())

    parser = argparse.ArgumentParser(
        description="OpenAI Proxy v6 - Dual Backend (llama.cpp / vLLM)"
    )
    parser.add_argument(
        "--config-dir",
        default=os.environ.get("FORGE_CONFIG_DIR"),
        help="Directory containing an alternate config.py",
    )
    parser.add_argument(
        "--manage", action="store_true",
        help="Launch key management interface"
    )
    parser.add_argument(
        "--host", default=config.PROXY_HOST,
        help=f"Server host (default: {config.PROXY_HOST})"
    )
    parser.add_argument(
        "--port", type=int, default=config.PROXY_PORT,
        help=f"Server port (default: {config.PROXY_PORT})"
    )
    parser.add_argument(
        "--shutdown-timeout",
        type=int,
        default=30,
        help="Seconds uvicorn waits for in-flight requests before shutdown cleanup (default: 30)",
    )

    args = parser.parse_args()

    if args.manage:
        from cli import management_menu
        management_menu()
        return

    # Start the proxy server
    from db import init_db
    init_db()

    print("=" * 60)
    print("OpenAI Proxy v6 - Dual Backend")
    print("=" * 60)
    print(f"  Proxy server:    http://{args.host}:{args.port}")
    print()
    print("  To manage API keys, run:")
    print("    python main.py --manage")
    print("=" * 60)

    import uvicorn
    from proxy_server import app
    from process_manager import process_manager
    # uvicorn.run(app, host=args.host, port=args.port) -> no log version

    atexit.register(process_manager.cleanup_sync)

    log_config = copy.deepcopy(uvicorn.config.LOGGING_CONFIG)
    log_config["formatters"]["default"]["fmt"] = "%(asctime)s | %(levelprefix)s %(message)s"

    # disable uvicorn's default log, use custom middleware
    try:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_config=log_config,
            access_log=False,
            timeout_graceful_shutdown=args.shutdown_timeout,
        )
    finally:
        process_manager.cleanup_sync()


if __name__ == "__main__":
    main()
