"""
llm.py — LLM client abstraction (OpenAI / Ollama)
═══════════════════════════════════════════════════════════════════════════════
PROJECT MAP  (masmid/)
─────────────────────────────────────────────────
  main.py            Entry point, CLI parsing, launches app
  config.py          Config constants, CLI arg parsing
  llm.py           ◄ YOU ARE HERE — LLM chat/embed/parse_json
  models.py          Dataclasses: Node, Edge, EdgeChannels
  vector_store.py    VectorStore (embedding index)
  knowledge_graph.py KnowledgeGraph
  rabbi_init.py      init_hillel(), init_shammai()
  agents.py          RabbiAgent, DreamAgent
  sefaria.py         Sefaria API
  orchestrator.py    DebateOrchestrator
  log_system.py      LogSystem
  auth.py            UserStore
  prompts.py         Prompt templates
  app.py             Flask app factory + routes
  templates.py       HTML templates

EXPORTS:
  LLMError           Exception class
  llm_chat(messages, system, model, timeout) → str
  llm_embed(text) → list[float]
  parse_json(text) → dict
  check_connection() → bool   (startup health check)

DEPENDS ON:
  config.LLM_BACKEND, config.LLM_BASE_URL, config.LLM_API_KEY,
  config.LLM_MODEL, config.EMBED_MODEL, config.LLM_TIMEOUT
═══════════════════════════════════════════════════════════════════════════════
"""

import json
import re
from typing import Optional

import requests
from openai import OpenAI

import config


class LLMError(RuntimeError):
    """Raised on any LLM call failure."""
    pass


# ─── Lazy singleton client ────────────────────────────────────────────────────

_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if config.LLM_BACKEND == "openai" and not config.LLM_API_KEY:
            raise LLMError("OPENAI_API_KEY not set — export it before running.")
        _client = OpenAI(
            api_key=config.LLM_API_KEY,
            base_url=config.LLM_BASE_URL,  # None → default OpenAI; or Ollama /v1
        )
    return _client


def reset_client():
    """Force re-creation of the client (e.g. after config changes)."""
    global _client
    _client = None


# ─── Chat completion ──────────────────────────────────────────────────────────

def llm_chat(messages, system=None, model=None, timeout=None) -> str:
    """Send a chat completion request. Returns the assistant's text."""
    msgs = ([{"role": "system", "content": system}] if system else []) + messages
    try:
        client = _get_client()
        resp = client.chat.completions.create(
            model=model or config.LLM_MODEL,
            messages=msgs,
            timeout=timeout or config.LLM_TIMEOUT,
        )
    except LLMError:
        raise
    except Exception as e:
        raise LLMError(f"LLM chat error ({config.LLM_BACKEND}): {e}")

    content = resp.choices[0].message.content if resp.choices else None
    if not isinstance(content, str):
        raise LLMError(f"Bad response from {config.LLM_BACKEND}: {resp}")
    return content


# ─── Embeddings ───────────────────────────────────────────────────────────────

def llm_embed(text) -> list[float]:
    """Get an embedding vector for text.

    For Ollama, uses the OpenAI-compatible /v1/embeddings endpoint.
    If the Ollama model doesn't support embeddings via the OpenAI-compat API,
    falls back to the native Ollama /api/embeddings endpoint.
    """
    text = text[:8000]

    # Try OpenAI-compatible endpoint first (works for both backends)
    try:
        client = _get_client()
        resp = client.embeddings.create(
            model=config.EMBED_MODEL,
            input=text,
        )
        emb = resp.data[0].embedding
        if not emb:
            raise LLMError("Empty embedding returned")
        return emb
    except LLMError:
        raise
    except Exception as e:
        # For Ollama, try native endpoint as fallback
        if config.LLM_BACKEND == "ollama":
            return _ollama_native_embed(text)
        raise LLMError(f"Embed error ({config.LLM_BACKEND}): {e}")


def _ollama_native_embed(text) -> list[float]:
    """Fallback: use Ollama's native /api/embeddings endpoint."""
    try:
        base = config.LLM_BASE_URL.replace("/v1", "")  # strip /v1
        resp = requests.post(
            f"{base}/api/embeddings",
            json={"model": config.EMBED_MODEL, "prompt": text[:8000]},
            timeout=config.LLM_TIMEOUT,
        )
        resp.raise_for_status()
        emb = resp.json().get("embedding", [])
        if not emb:
            raise LLMError("Empty embedding from Ollama native API")
        return emb
    except LLMError:
        raise
    except Exception as e:
        raise LLMError(f"Ollama native embed error: {e}")


# ─── JSON parsing helper ─────────────────────────────────────────────────────

def parse_json(text) -> dict:
    """Best-effort parse of LLM JSON output (handles markdown fences, etc)."""
    t = text.strip()
    for attempt in [t, re.sub(r"^```[a-z]*\n?", "", t, flags=re.M).rstrip("`").strip()]:
        try:
            return json.loads(attempt)
        except Exception:
            pass
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return {}


# ─── Startup health check ────────────────────────────────────────────────────

def check_connection() -> bool:
    """Verify the LLM backend is reachable. Returns True on success."""
    if config.LLM_BACKEND == "ollama":
        try:
            base = config.LLM_BASE_URL.replace("/v1", "")
            r = requests.get(f"{base}/api/tags", timeout=5)
            r.raise_for_status()
            models = [m.get("name", "") for m in r.json().get("models", [])]
            print(f"[LLM] Ollama OK — available models: {models[:8]}", flush=True)
            return True
        except Exception as e:
            print(f"[LLM] ⚠ Ollama connection failed: {e}", flush=True)
            return False
    else:
        if not config.LLM_API_KEY:
            print("[LLM] ⚠ OPENAI_API_KEY not set — LLM calls will fail!", flush=True)
            return False
        try:
            client = _get_client()
            client.models.retrieve(config.LLM_MODEL)
            print(f"[LLM] OpenAI OK — {config.LLM_MODEL} accessible", flush=True)
            return True
        except Exception as e:
            print(f"[LLM] ⚠ OpenAI: {e}", flush=True)
            return False
