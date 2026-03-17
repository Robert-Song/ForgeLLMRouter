"""
proxy_v6/main.py
Entry point for the proxy v6 system.

Usage:
  python main.py                  # Start proxy server
  python main.py --manage         # Key management CLI
"""

import argparse
import copy
import logging
import os
import time

from config import PROXY_HOST, PROXY_PORT


def main():
    parser = argparse.ArgumentParser(
        description="OpenAI Proxy v6 - Dual Backend (llama.cpp / vLLM)"
    )
    parser.add_argument(
        "--manage", action="store_true",
        help="Launch key management interface"
    )
    parser.add_argument(
        "--host", default=PROXY_HOST,
        help=f"Server host (default: {PROXY_HOST})"
    )
    parser.add_argument(
        "--port", type=int, default=PROXY_PORT,
        help=f"Server port (default: {PROXY_PORT})"
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
    # uvicorn.run(app, host=args.host, port=args.port) -> no log version

    log_config = copy.deepcopy(uvicorn.config.LOGGING_CONFIG)
    log_config["formatters"]["default"]["fmt"] = "%(asctime)s | %(levelprefix)s %(message)s"

    # disable uvicorn's default log, use custom middleware
    uvicorn.run(app, host=args.host, port=args.port, log_config=log_config, access_log=False)


if __name__ == "__main__":
    main()
