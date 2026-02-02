#!/usr/bin/env python3
"""
Simple helper script that fact-checks a hard-coded claim by calling the local
Perplexica API. It fetches the first available chat + embedding model pair,
submits a focused search request, and prints the generated verdict plus sources.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

BASE_URL = os.environ.get("PERPLEXICA_URL", "http://localhost:3000")
FOCUS_MODE = os.environ.get("PERPLEXICA_FOCUS_MODE", "webSearch")
OPTIMIZATION_MODE = os.environ.get("PERPLEXICA_OPT_MODE", "balanced")
FACTCHECK_OUTPUT_SPEC = os.environ.get(
    "FACTCHECK_OUTPUT_SPEC",
    (
        '{"verdict":"verified|false|unverifiable","rationale":"string (≤2 '
        'sentences)","evidence":[{"id":"E1","title":"string","url":"URL","quote":"verbatim '
        'excerpt"}],"confidence":0.0-1.0}'
    ),
)
DEFAULT_SYSTEM_INSTRUCTIONS_TEMPLATE = (
    """
You are an evidence-based fact checker. Given a CLAIM, optional supplemental CONTEXT, and retrieved search EVIDENCE, decide whether the claim is VERIFIED, FALSE, or UNVERIFIABLE. Rely strictly on the evidence; do not hallucinate or use prior knowledge.

CLAIM: {CLAIM}
CONTEXT (optional): {CONTEXT}

VERDICT DEFINITIONS
- verified: clear, credible evidence directly confirms the claim.
- false: credible evidence contradicts the claim or shows it depends on debunked information.
- unverifiable: evidence is missing, conflicting, speculative, or the claim is inherently normative/predictive. For moral/value judgments always answer “unverifiable”.

LEGAL CLAIMS
- Only mark “verified” when citing relevant statutes, constitutional text, or widely accepted legal consensus.

OUTPUT FORMAT
- Return exactly **one** JSON object, no Markdown fences, following this schema:
{FACTCHECK_OUTPUT_SPEC}

STRICT OUTPUT RULES
1. Populate `verdict` with `verified`, `false`, or `unverifiable`.
2. Limit `rationale` to ≤2 concise sentences referencing evidence IDs (E1, E2, …).
3. Include 1–3 evidence entries for verified/false verdicts; each needs `id`, `title`, `url`, and a short verbatim quote.
4. Quotes must be faithful to the source text; never fabricate.
5. For unverifiable verdicts, use `evidence: []` unless a specific citation explains why the claim can’t be confirmed.
6. Set `confidence` between 0 and 1 (e.g., 0.25, 0.82). Use ≥0.8 when highly certain; ≤0.5 when weak.
7. Never output hidden reasoning, chit-chat, chain-of-thought, or headings—only the JSON object.
""".strip()
)
CLAIM = (
    "when we watch Trump's lawyers show and we watch Trump talk about the hidden boxes in the state farm arena that were pulled out from under the table and put back and we don't know where they came from, we do know where they came from."
)
CONTEXT = (
    "Claim was stated around January 17th, 2026."
)
CLAIM_FIXTURE_FILENAME = "claim_fixture.json"
FIREFIGHTER_FIXTURE_FILENAME = "claim_fixture_firefighter.json"
CHAT_PROVIDER_OVERRIDE = os.environ.get("FACTCHECK_CHAT_PROVIDER_ID")
CHAT_MODEL_OVERRIDE = os.environ.get("FACTCHECK_CHAT_MODEL_KEY")
EMBEDDING_PROVIDER_OVERRIDE = os.environ.get("FACTCHECK_EMBEDDING_PROVIDER_ID")
EMBEDDING_MODEL_OVERRIDE = os.environ.get("FACTCHECK_EMBEDDING_MODEL_KEY")
ModelSelection = Tuple[str, Dict[str, Any]]


def fetch_default_models(
    chat_provider_override: Optional[str] = None,
    chat_model_override: Optional[str] = None,
    embedding_provider_override: Optional[str] = None,
    embedding_model_override: Optional[str] = None,
) -> Tuple[ModelSelection, ModelSelection]:
    """Return (provider id, model) tuples for chat and embedding models."""
    providers_url = f"{BASE_URL}/api/providers"
    response = requests.get(providers_url, timeout=10)
    response.raise_for_status()

    providers = response.json().get("providers", [])
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

    # No provider override, so search for the specified model key.
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


def extract_claim_and_context(query_text: str) -> Tuple[str, Optional[str]]:
    """Parse the human-readable claim and optional context from a query string."""
    if not query_text:
        return "", None

    text = query_text.strip()
    context: Optional[str] = None
    context_marker = "\nContext:"
    if context_marker in text:
        text, context = text.split(context_marker, 1)
        context = context.strip() or None

    prefix = "Fact-check the following claim:"
    if text.lower().startswith(prefix.lower()):
        text = text[len(prefix) :].strip()

    return text, context


def build_system_instructions(claim: str, context: Optional[str]) -> str:
    context_value = context if context else "N/A"
    return DEFAULT_SYSTEM_INSTRUCTIONS_TEMPLATE.format(
        CLAIM=claim.strip() or "<empty claim>",
        CONTEXT=context_value,
        FACTCHECK_OUTPUT_SPEC=FACTCHECK_OUTPUT_SPEC,
    )


def fact_check_claim(
    claim: str,
    context: Optional[str] = None,
    payload_override: Optional[Dict[str, Any]] = None,
    chat_provider_override: Optional[str] = None,
    chat_model_override: Optional[str] = None,
    embedding_provider_override: Optional[str] = None,
    embedding_model_override: Optional[str] = None,
) -> Tuple[Dict[str, Any], float]:
    """
    Call the Perplexica search endpoint with instructions to fact-check claim.

    Returns the JSON response plus the elapsed request time in seconds.
    """
    chat_selection, embedding_selection = fetch_default_models(
        chat_provider_override=chat_provider_override,
        chat_model_override=chat_model_override,
        embedding_provider_override=embedding_provider_override,
        embedding_model_override=embedding_model_override,
    )
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
        "query": f"Fact-check the following claim: {claim}",
        "sources": ["web"],
        "history": [],
        "systemInstructions": build_system_instructions(claim, context),
        "stream": False,
    }

    if payload_override:
        payload.update(payload_override)

    payload.setdefault("query", f"Fact-check the following claim: {claim}")
    payload["systemInstructions"] = build_system_instructions(claim, context)

    chat_override_requested = bool(chat_provider_override or chat_model_override)
    if chat_override_requested or "chatModel" not in payload:
        payload["chatModel"] = {
            "providerId": chat_provider_id,
            "key": chat_model["key"],
        }

    embedding_override_requested = bool(
        embedding_provider_override or embedding_model_override
    )
    if embedding_override_requested or "embeddingModel" not in payload:
        payload["embeddingModel"] = {
            "providerId": embedding_provider_id,
            "key": embedding_model["key"],
        }

    print(
        "Requesting models:\n"
        f"  Chat      -> provider: {payload['chatModel']['providerId']}, "
        f"key: {payload['chatModel']['key']}\n"
        f"  Embedding -> provider: {payload['embeddingModel']['providerId']}, "
        f"key: {payload['embeddingModel']['key']}\n"
    )

    start_time = time.perf_counter()
    response = requests.post(
        f"{BASE_URL}/api/search", json=payload, timeout=1080
    )
    response.raise_for_status()
    elapsed = time.perf_counter() - start_time
    return response.json(), elapsed


def load_fixture_payload(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fixture_file:
        return json.load(fixture_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simple helper script to fact-check claims via Perplexica"
    )
    parser.add_argument(
        "--use-claim-fixture",
        action="store_true",
        help=(
            "Load and send the request payload stored in the adjacent claim "
            f"fixture file ({CLAIM_FIXTURE_FILENAME})"
        ),
    )
    parser.add_argument(
        "--use-firefighter-claim-fixture",
        action="store_true",
        help=(
            "Load and send the firefighter rebuttal payload stored in the "
            f"adjacent fixture file ({FIREFIGHTER_FIXTURE_FILENAME})"
        ),
    )
    parser.add_argument(
        "--show-raw-response",
        action="store_true",
        help="Print the full JSON response returned by Perplexica",
    )
    parser.add_argument(
        "--chat-provider-id",
        default='b148cfb9-aa25-4886-b545-e79c02a24cc5',
        help=(
            "Specify the provider id to use for the chat model "
            "(default: auto-detect or FACTCHECK_CHAT_PROVIDER_ID env var)"
        ),
    )
    parser.add_argument(
        "--chat-model-key",
        # default='gpt-oss:20b',
        default='gpt-oss:20b',
        help=(
            "Specify the chat model key to use (default: first available chat "
            "model or FACTCHECK_CHAT_MODEL_KEY env var)"
        ),
    )
    parser.add_argument(
        "--embedding-provider-id",
        default='b148cfb9-aa25-4886-b545-e79c02a24cc5',
        help=(
            "Specify the provider id to use for the embedding model "
            "(default: auto-detect or FACTCHECK_EMBEDDING_PROVIDER_ID env var)"
        ),
    )
    parser.add_argument(
        "--embedding-model-key",
        default='mxbai-embed-large:335m',
        help=(
            "Specify the embedding model key to use (default: first available "
            "embedding model or FACTCHECK_EMBEDDING_MODEL_KEY env var)"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload_override: Optional[Dict[str, Any]] = None
    claim_text = CLAIM
    context_text: Optional[str] = None

    if args.use_claim_fixture and args.use_firefighter_claim_fixture:
        sys.exit("Select only one claim fixture option.")

    if args.use_claim_fixture or args.use_firefighter_claim_fixture:
        filename = (
            FIREFIGHTER_FIXTURE_FILENAME
            if args.use_firefighter_claim_fixture
            else CLAIM_FIXTURE_FILENAME
        )
        fixture_path = Path(__file__).with_name(filename)
        try:
            payload_override = load_fixture_payload(fixture_path)
            claim_text = payload_override.get("query", CLAIM)
        except FileNotFoundError:
            sys.exit(f"Missing fixture file: {fixture_path}")
        except json.JSONDecodeError as exc:
            sys.exit(f"Failed to parse fixture JSON ({fixture_path}): {exc}")
        except OSError as exc:
            sys.exit(f"Unable to read fixture file {fixture_path}: {exc}")

    parsed_claim, context_text = extract_claim_and_context(claim_text)

    print(f"Checking claim:\n  \"{claim_text}\"\n")
    try:
        result, elapsed = fact_check_claim(
            parsed_claim,
            context_text,
            payload_override,
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
        sys.exit(f"Failed to fact-check claim: {exc}")

    message = result.get("message") or "<no message returned>"
    sources = result.get("sources") or []
    light_sources = result.get("lightSources") or []

    raw_response = json.dumps(result, indent=2, sort_keys=True)

    print("Fact-check verdict:\n")
    print(message.strip(), "\n")
    print(f"Perplexica responded in {elapsed:.2f} seconds.\n")
    if not sources and not light_sources:
        print("No sources were returned.")
    else:
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

    print("\nRaw Perplexica response:\n")
    print(raw_response, "\n")


if __name__ == "__main__":
    main()
