"""
config.py — Central configuration for Masmid
═══════════════════════════════════════════════════════════════════════════════
PROJECT MAP  (masmid/)
─────────────────────────────────────────────────
  main.py            Entry point, CLI parsing, launches app
  config.py        ◄ YOU ARE HERE — config constants, CLI arg parsing
  llm.py             LLM abstraction (OpenAI / Ollama)
  models.py          Dataclasses: Node, Edge, EdgeChannels
  vector_store.py    VectorStore (embedding index)
  knowledge_graph.py KnowledgeGraph (nodes, edges, queries)
  rabbi_init.py      init_hillel(), init_shammai() — KG seed data
  agents.py          RabbiAgent, DreamAgent
  sefaria.py         Sefaria API: fetch_daf_yomi(), fetch_daf_text()
  orchestrator.py    DebateOrchestrator
  log_system.py      LogSystem (JSONL + meta-log)
  auth.py            UserStore (JSON-backed user auth)
  prompts.py         All LLM prompt templates
  app.py             Flask app factory + all routes
  templates.py       HTML template strings (login, main UI)
  requirements.txt   Dependencies

SHARED CONSTANTS EXPORTED:
  All uppercase config values (DATA_DIR, PORT, LENSES, etc.)
  parse_cli_args()  — returns argparse.Namespace
  LLM_BACKEND       — "openai" | "ollama"
  LLM_MODEL, EMBED_MODEL, LLM_BASE_URL, LLM_API_KEY
═══════════════════════════════════════════════════════════════════════════════
"""

import argparse
import os
from pathlib import Path


# ─── CLI Argument Parsing ─────────────────────────────────────────────────────

def parse_cli_args(argv=None):
    """Parse command-line arguments. Called once from main.py."""
    p = argparse.ArgumentParser(
        description="Masmid — Daf Yomi dual-rabbi KG debate system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--ollama", metavar="MODEL",
        nargs="?", const="gpt-oss:20b",
        help="Use local Ollama backend. Default model: gpt-oss:20b. "
             "Example: --ollama  or  --ollama llama3:8b",
    )
    p.add_argument(
        "--ollama-url", default="http://localhost:11434",
        help="Ollama server base URL (default: http://localhost:11434)",
    )
    p.add_argument(
        "--ollama-embed-model", default="nomic-embed-text",
        help="Ollama embedding model (default: nomic-embed-text)",
    )
    p.add_argument(
        "--model", default=None,
        help="Override chat model for any backend",
    )
    p.add_argument(
        "--port", type=int, default=None,
        help="Server port (default: 5004 or $PORT)",
    )
    p.add_argument(
        "--data-dir", default=None,
        help="Data directory path (default: ./masmid_data or $DATA_DIR)",
    )
    p.add_argument(
        "--url-prefix", default=None,
        help="URL prefix for reverse proxy (default: /masmid or $URL_PREFIX)",
    )
    p.add_argument(
        "--context-limit", type=int, default=None,
        help="Max tokens for system prompt context (default: 4000 or $CONTEXT_LIMIT). "
             "Lower for small models, higher for large.",
    )
    return p.parse_args(argv)


# ─── Defaults (can be overridden by env vars and CLI args) ────────────────────

# These are module-level and will be mutated by apply_cli_args() in main.py
# before anything else imports them.

LLM_BACKEND    = "openai"           # "openai" | "ollama"
LLM_BASE_URL   = None               # None = default OpenAI; or Ollama URL
LLM_API_KEY    = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL      = os.getenv("OPENAI_MODEL", "gpt-4o")
EMBED_MODEL    = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
LLM_TIMEOUT    = int(os.getenv("OPENAI_TIMEOUT", "120"))
CONTEXT_LIMIT  = int(os.getenv("CONTEXT_LIMIT", "4000"))   # token budget for system prompt

DREAM_INTERVAL = int(os.getenv("DREAM_INTERVAL", "180"))
DEBATE_DELAY   = float(os.getenv("DEBATE_DELAY", "3.0"))
DATA_DIR       = Path(os.getenv("DATA_DIR", "./masmid_data"))
PORT           = int(os.getenv("PORT", "5004"))
SECRET_KEY     = os.getenv("SECRET_KEY", os.urandom(32).hex())
URL_PREFIX     = os.getenv("URL_PREFIX", "/masmid")

# KG tuning
DRIVE_BUILDUP_RATE = 0.010
SALIENCE_DECAY     = 0.988
TOP_K              = 4

# Lens vocabulary for KG interrogation
LENSES = {
    "compassion":  "compassion mercy love humanity leniency the whole person",
    "precision":   "precision exactness boundary strict letter of the law ruling",
    "narrative":   "story parable analogy teaching example metaphor",
    "logic":       "logical argument deduction proof precedent textual analysis",
    "spirit":      "spirit intention meaning purpose beyond the letter kavvanah",
    "conflict":    "disagreement tension machloket opposition challenge",
    "memory":      "memory recall past precedent learned experience tradition",
    "prayer":      "prayer worship divine service intention holiness",
    "property":    "property ownership transfer commerce financial obligation",
    "purity":      "ritual purity impurity tamei tahor cleansing",
}


def apply_cli_args(args):
    """Mutate module-level config based on parsed CLI args.
    Must be called early in main.py before other modules read these values.
    """
    # These are module-level; we reassign via globals()
    import config as _self

    if args.ollama:
        _self.LLM_BACKEND  = "ollama"
        _self.LLM_MODEL    = args.ollama        # e.g. "gpt-oss:20b"
        _self.EMBED_MODEL  = args.ollama_embed_model
        _self.LLM_BASE_URL = args.ollama_url.rstrip("/") + "/v1"
        _self.LLM_API_KEY  = "ollama"            # Ollama doesn't need a real key
        print(f"[CONFIG] Ollama backend: model={_self.LLM_MODEL} "
              f"embed={_self.EMBED_MODEL} url={_self.LLM_BASE_URL}", flush=True)
    else:
        _self.LLM_BACKEND  = "openai"
        _self.LLM_BASE_URL = None
        # LLM_API_KEY and LLM_MODEL already set from env

    # CLI overrides for both backends
    if args.model:
        _self.LLM_MODEL = args.model
    if args.port is not None:
        _self.PORT = args.port
    if args.data_dir is not None:
        _self.DATA_DIR = Path(args.data_dir)
    if args.url_prefix is not None:
        _self.URL_PREFIX = args.url_prefix
    if args.context_limit is not None:
        _self.CONTEXT_LIMIT = args.context_limit

    print(f"[CONFIG] Backend={_self.LLM_BACKEND}  Model={_self.LLM_MODEL}  "
          f"Embed={_self.EMBED_MODEL}", flush=True)
    print(f"[CONFIG] Data={_self.DATA_DIR}  Port={_self.PORT}  "
          f"Prefix='{_self.URL_PREFIX}'  ContextLimit={_self.CONTEXT_LIMIT}", flush=True)
