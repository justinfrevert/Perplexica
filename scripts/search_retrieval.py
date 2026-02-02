#!/usr/bin/env python3
"""
Helper script that sends a search-focused retrieval request to the Perplexica API.
It prints the assistant response along with any sources returned.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

BASE_URL = os.environ.get("PERPLEXICA_URL", "http://localhost:3001")
FOCUS_MODE = os.environ.get("PERPLEXICA_FOCUS_MODE", "webSearch")
OPTIMIZATION_MODE = os.environ.get("PERPLEXICA_OPT_MODE", "balanced")

DEFAULT_QUERY = (
    "Donald Trump himself says that he would be a day one dictator, "
    "that he would suspend the Constitution. fact check"
)
DEFAULT_SYSTEM_INSTRUCTIONS = """
You are a search-focused research assistant. Your job is to perform the best possible web search for the provided QUERY and optional CONTEXT, then return the strongest ranked sources with rich snippets. You NEVER provide a verdict or conclusion; only retrieval results.

INSTRUCTIONS
- Prioritize recall and diversity while keeping results on-topic.
- Include multiple snippets per source when useful so downstream systems can align passages to the claim.
- Prefer sources with clear publishers and dates.
- Return ranked results; highest-quality first.
- Do not speculate or summarize beyond the snippets returned.

OUTPUT EXPECTATION
- Provide structured fields: title, url, snippet (<=300 chars, verbatim), publisher, published_date when available, and a numeric score/rank.
- If CONTEXT is provided, use it to disambiguate but do not restate it.
""".strip()

DEFAULT_CHAT_PROVIDER_ID = "c24d62fe-5851-4405-a998-a8cadf5b604f"
DEFAULT_CHAT_MODEL_KEY = "gemma3:12b-it-qat"
DEFAULT_EMBEDDING_PROVIDER_ID = "c24d62fe-5851-4405-a998-a8cadf5b604f"
DEFAULT_EMBEDDING_MODEL_KEY = "nomic-embed-text:latest"

DEFAULT_LIGHT_EVIDENCE = True

CHAT_PROVIDER_OVERRIDE = os.environ.get("SEARCH_CHAT_PROVIDER_ID")
CHAT_MODEL_OVERRIDE = os.environ.get("SEARCH_CHAT_MODEL_KEY")
EMBEDDING_PROVIDER_OVERRIDE = os.environ.get("SEARCH_EMBEDDING_PROVIDER_ID")
EMBEDDING_MODEL_OVERRIDE = os.environ.get("SEARCH_EMBEDDING_MODEL_KEY")

ModelSelection = Tuple[str, Dict[str, Any]]


def _fetch_providers() -> List[Dict[str, Any]]:
    providers_url = f"{BASE_URL}/api/providers"
    response = requests.get(providers_url, timeout=10)
    response.raise_for_status()
    return response.json().get("providers", [])


def _first_model(
    providers: List[Dict[str, Any]], model_field: str
) -> Optional[ModelSelection]:
    for provider in providers:
        models = provider.get(model_field) or []
        if models:
            return provider["id"], models[0]
    return None


def _resolve_model_selection(
    providers: List[Dict[str, Any]],
    model_field: str,
    model_label: str,
    provider_override: Optional[str],
    model_key_override: Optional[str],
    default_model: ModelSelection,
) -> ModelSelection:
    if not provider_override and not model_key_override:
        return default_model

    if provider_override:
        provider = next(
            (p for p in providers if p.get("id") == provider_override), None
        )
        if not provider:
            raise RuntimeError(
                f"No provider found with id '{provider_override}' "
                f"for {model_label} models."
            )
        models = provider.get(model_field) or []
        if not models:
            raise RuntimeError(
                f"Provider '{provider_override}' has no {model_label} models. "
                "Add one via the Perplexica settings UI first."
            )
        if model_key_override:
            match = next(
                (m for m in models if m.get("key") == model_key_override), None
            )
            if not match:
                available = ", ".join(m.get("key", "<unknown>") for m in models)
                raise RuntimeError(
                    f"Provider '{provider_override}' does not have a "
                    f"{model_label} model with key '{model_key_override}'. "
                    f"Available keys: {available or 'none'}."
                )
            return provider["id"], match
        return provider["id"], models[0]

    for provider in providers:
        models = provider.get(model_field) or []
        match = next(
            (m for m in models if m.get("key") == model_key_override), None
        )
        if match:
            return provider["id"], match

    raise RuntimeError(
        f"No {model_label} model with key '{model_key_override}' was found. "
        "Add or enable it via the Perplexica settings UI first."
    )


def _preferred_or_default(
    providers: List[Dict[str, Any]],
    model_field: str,
    model_label: str,
    provider_id: str,
    model_key: str,
    fallback_model: ModelSelection,
) -> ModelSelection:
    provider = next((p for p in providers if p.get("id") == provider_id), None)
    if provider:
        models = provider.get(model_field) or []
        match = next((m for m in models if m.get("key") == model_key), None)
        if match:
            return provider["id"], match

    print(
        f"Warning: default {model_label} model "
        f"{provider_id}:{model_key} not found; falling back to the first "
        "available model.",
        file=sys.stderr,
    )
    return fallback_model


def fetch_models(
    chat_provider_override: Optional[str] = None,
    chat_model_override: Optional[str] = None,
    embedding_provider_override: Optional[str] = None,
    embedding_model_override: Optional[str] = None,
) -> Tuple[ModelSelection, ModelSelection]:
    providers = _fetch_providers()
    default_chat = _first_model(providers, "chatModels")
    default_embedding = _first_model(providers, "embeddingModels")

    if not default_chat:
        raise RuntimeError(
            "No chat model is available. Add or enable a provider via the "
            "Perplexica settings UI first."
        )
    if not default_embedding:
        raise RuntimeError(
            "No embedding model is available. Add or enable a provider via the "
            "Perplexica settings UI first."
        )

    if (
        chat_provider_override
        or chat_model_override
        or embedding_provider_override
        or embedding_model_override
    ):
        chat_selection = _resolve_model_selection(
            providers,
            model_field="chatModels",
            model_label="chat",
            provider_override=chat_provider_override,
            model_key_override=chat_model_override,
            default_model=default_chat,
        )
        embedding_selection = _resolve_model_selection(
            providers,
            model_field="embeddingModels",
            model_label="embedding",
            provider_override=embedding_provider_override,
            model_key_override=embedding_model_override,
            default_model=default_embedding,
        )
        return chat_selection, embedding_selection

    chat_selection = _preferred_or_default(
        providers,
        model_field="chatModels",
        model_label="chat",
        provider_id=DEFAULT_CHAT_PROVIDER_ID,
        model_key=DEFAULT_CHAT_MODEL_KEY,
        fallback_model=default_chat,
    )
    embedding_selection = _preferred_or_default(
        providers,
        model_field="embeddingModels",
        model_label="embedding",
        provider_id=DEFAULT_EMBEDDING_PROVIDER_ID,
        model_key=DEFAULT_EMBEDDING_MODEL_KEY,
        fallback_model=default_embedding,
    )
    return chat_selection, embedding_selection


def load_system_instructions(path: Optional[Path]) -> str:
    if not path:
        return DEFAULT_SYSTEM_INSTRUCTIONS
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        sys.exit(f"Unable to read system instructions file {path}: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit a search-focused retrieval query via Perplexica"
    )
    parser.add_argument(
        "--query",
        default=DEFAULT_QUERY,
        help="Search query to send to Perplexica",
    )
    parser.add_argument(
        "--system-instructions",
        default=None,
        help="Override the system instructions text",
    )
    parser.add_argument(
        "--system-instructions-file",
        type=Path,
        help="Load system instructions from a file",
    )
    parser.add_argument(
        "--light-evidence",
        action="store_true",
        default=DEFAULT_LIGHT_EVIDENCE,
        help="Request a lighter evidence payload when supported",
    )
    parser.add_argument(
        "--no-light-evidence",
        action="store_false",
        dest="light_evidence",
        help="Disable the light evidence flag",
    )
    parser.add_argument(
        "--show-raw-response",
        action="store_true",
        help="Print the full JSON response returned by Perplexica",
    )
    parser.add_argument(
        "--chat-provider-id",
        default=CHAT_PROVIDER_OVERRIDE,
        help=(
            "Specify the provider id to use for the chat model "
            "(default: SEARCH_CHAT_PROVIDER_ID env var or the script default)"
        ),
    )
    parser.add_argument(
        "--chat-model-key",
        default=CHAT_MODEL_OVERRIDE,
        help=(
            "Specify the chat model key to use "
            "(default: SEARCH_CHAT_MODEL_KEY env var or the script default)"
        ),
    )
    parser.add_argument(
        "--embedding-provider-id",
        default=EMBEDDING_PROVIDER_OVERRIDE,
        help=(
            "Specify the provider id to use for the embedding model "
            "(default: SEARCH_EMBEDDING_PROVIDER_ID env var or the script default)"
        ),
    )
    parser.add_argument(
        "--embedding-model-key",
        default=EMBEDDING_MODEL_OVERRIDE,
        help=(
            "Specify the embedding model key to use "
            "(default: SEARCH_EMBEDDING_MODEL_KEY env var or the script default)"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    system_instructions = (
        args.system_instructions.strip()
        if args.system_instructions
        else load_system_instructions(args.system_instructions_file)
    )

    print(f"Running query:\n  \"{args.query}\"\n")
    try:
        chat_selection, embedding_selection = fetch_models(
            chat_provider_override=args.chat_provider_id,
            chat_model_override=args.chat_model_key,
            embedding_provider_override=args.embedding_provider_id,
            embedding_model_override=args.embedding_model_key,
        )
    except requests.HTTPError as exc:
        sys.exit(
            f"Perplexica API returned an error: {exc.response.status_code} "
            f"{exc.response.text}"
        )
    except Exception as exc:  # pragma: no cover - simple CLI error path
        sys.exit(f"Failed to resolve models: {exc}")

    chat_provider_id, chat_model = chat_selection
    embedding_provider_id, embedding_model = embedding_selection

    payload = {
        "chatModel": {
            "providerId": chat_provider_id,
            "key": chat_model["key"],
        },
        "embeddingModel": {
            "providerId": embedding_provider_id,
            "key": embedding_model["key"],
        },
        "optimizationMode": OPTIMIZATION_MODE,
        "focusMode": FOCUS_MODE,
        "query": args.query,
        "history": [],
        "systemInstructions": system_instructions,
        "stream": False,
        "lightEvidence": args.light_evidence,
    }

    print(
        "Requesting models:\n"
        f"  Chat      -> provider: {payload['chatModel']['providerId']}, "
        f"key: {payload['chatModel']['key']}\n"
        f"  Embedding -> provider: {payload['embeddingModel']['providerId']}, "
        f"key: {payload['embeddingModel']['key']}\n"
    )

    start_time = time.perf_counter()
    try:
        response = requests.post(
            f"{BASE_URL}/api/search", json=payload, timeout=1080
        )
        response.raise_for_status()
    except requests.HTTPError as exc:
        sys.exit(
            f"Perplexica API returned an error: {exc.response.status_code} "
            f"{exc.response.text}"
        )
    except Exception as exc:  # pragma: no cover - simple CLI error path
        sys.exit(f"Failed to run search request: {exc}")

    elapsed = time.perf_counter() - start_time
    result = response.json()
    message = result.get("message") or "<no message returned>"
    sources = result.get("sources") or []
    light_sources = result.get("lightSources") or []

    if args.show_raw_response:
        print("Raw Perplexica response:\n")
        print(json.dumps(result, indent=2, sort_keys=True), "\n")

    print("Search response:\n")
    print(message.strip(), "\n")
    print(f"Perplexica responded in {elapsed:.2f} seconds.\n")
    if not sources and not light_sources:
        print("No sources were returned.")
        return

    print("Sources:")
    for idx, src in enumerate(sources, start=1):
        metadata = src.get("metadata") or {}
        title = metadata.get("title", "Untitled source")
        url = metadata.get("url", "N/A")
        print(f"{idx}. {title} - {url}")

    if light_sources:
        print("\nLight sources (not fully crawled):")
        for idx, src in enumerate(light_sources, start=1):
            metadata = src.get("metadata") or {}
            title = metadata.get("title", "Untitled source")
            url = metadata.get("url", "N/A")
            print(f"{idx}. {title} - {url}")


if __name__ == "__main__":
    main()
