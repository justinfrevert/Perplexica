#!/usr/bin/env python3
import argparse
import json
import ssl
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_urls(search_config: dict) -> list[str]:
    urls = search_config.get("searxngURLs") or []
    if not isinstance(urls, list):
        urls = []

    normalized = [
        url.strip()
        for url in urls
        if isinstance(url, str) and url.strip()
    ]

    if normalized:
        return normalized

    single = search_config.get("searxngURL", "")
    if isinstance(single, str) and single.strip():
        return [single.strip()]

    return []


def build_search_url(base_url: str, query: str) -> str:
    base = base_url.rstrip("/")
    params = urlencode({"q": query, "format": "json"})
    return f"{base}/search?{params}"


def fetch_json(url: str, timeout: float, context: ssl.SSLContext) -> dict:
    req = Request(url, headers={"User-Agent": "perplexica-searxng-check/1.0"})
    with urlopen(req, timeout=timeout, context=context) as resp:
        body = resp.read()
    return json.loads(body)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check all configured SearxNG URLs for JSON search responses.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "config.json",
        help="Path to config.json (default: ./data/config.json).",
    )
    parser.add_argument(
        "--query",
        default="health check",
        help="Search query to use for testing.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Request timeout in seconds.",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS certificate verification.",
    )
    args = parser.parse_args()

    if not args.config.exists():
        print(f"Config not found: {args.config}", file=sys.stderr)
        return 2

    config = load_config(args.config)
    search_config = config.get("search", {})
    urls = normalize_urls(search_config)

    if not urls:
        print("No SearxNG URLs configured.", file=sys.stderr)
        return 2

    context = (
        ssl._create_unverified_context()
        if args.insecure
        else ssl.create_default_context()
    )

    failures = 0

    for base_url in urls:
        search_url = build_search_url(base_url, args.query)
        try:
            data = fetch_json(search_url, args.timeout, context)
            results = data.get("results", [])
            if not isinstance(results, list):
                raise ValueError("Missing or invalid 'results' array.")
            print(f"[ok] {base_url} ({len(results)} results)")
        except (HTTPError, URLError, ValueError, json.JSONDecodeError) as exc:
            failures += 1
            print(f"[fail] {base_url} ({exc})", file=sys.stderr)

    if failures:
        print(f"{failures} instance(s) failed.", file=sys.stderr)
        return 1

    print(f"All {len(urls)} instance(s) returned JSON.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
