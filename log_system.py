"""
log_system.py — Logging: JSONL event log + Markdown meta-log
═══════════════════════════════════════════════════════════════════════════════
PROJECT MAP  (masmid/)
─────────────────────────────────────────────────
  main.py            Entry point
  config.py          Config constants
  llm.py             LLM abstraction
  models.py          Node, Edge, EdgeChannels
  vector_store.py    VectorStore
  knowledge_graph.py KnowledgeGraph
  rabbi_init.py      KG seed data
  agents.py          RabbiAgent, DreamAgent
  sefaria.py         Sefaria API
  orchestrator.py    DebateOrchestrator
  log_system.py    ◄ YOU ARE HERE — LogSystem class
  auth.py            UserStore
  prompts.py         Prompt templates
  app.py             Flask app factory + routes
  templates.py       HTML templates

EXPORTS:
  LogSystem  — log(), recent(), read_full(), read_meta(), append_meta(),
               new_session(), export_debate()

DEPENDS ON:
  config.DATA_DIR

USED BY:
  agents.py, orchestrator.py, app.py
═══════════════════════════════════════════════════════════════════════════════
"""

import json
import threading
import uuid
from datetime import datetime

import config


class LogSystem:
    def __init__(self):
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._lock       = threading.Lock()
        self._entries: list[dict] = []
        self.session_id  = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._log_file   = config.DATA_DIR / "log.jsonl"
        self._meta_file  = config.DATA_DIR / "meta_log.md"

        if not self._meta_file.exists():
            self._meta_file.write_text("# Masmid Meta-Log\n\n")
        with open(self._meta_file, "a") as f:
            f.write(f"\n## {self.session_id}\n\n")

    # ── Core ──────────────────────────────────────────────────────────────

    def log(self, source: str, etype: str, payload: dict,
            level: str = "INFO", tags: list = None) -> dict:
        e = {
            "id": str(uuid.uuid4())[:8],
            "ts": datetime.now().isoformat(),
            "session": self.session_id,
            "source": source,
            "event_type": etype,
            "level": level,
            "tags": tags or [],
            "payload": payload,
        }
        with self._lock:
            self._entries.append(e)
        with open(self._log_file, "a") as f:
            f.write(json.dumps(e) + "\n")
        return e

    def recent(self, n=80):
        with self._lock:
            return list(self._entries[-n:])

    def read_full(self, n=100):
        if not self._log_file.exists():
            return []
        lines = self._log_file.read_text().strip().splitlines()
        out = []
        for line in lines[-n:]:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
        return out

    # ── Meta-log ──────────────────────────────────────────────────────────

    def read_meta(self, chars=5000):
        if not self._meta_file.exists():
            return ""
        c = self._meta_file.read_text()
        return c[-chars:] if len(c) > chars else c

    def append_meta(self, text: str):
        with open(self._meta_file, "a") as f:
            f.write(text)

    # ── Session management ────────────────────────────────────────────────

    def new_session(self):
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with self._lock:
            self._entries = []
        self.append_meta(f"\n## {self.session_id}\n\n")

    # ── Export ────────────────────────────────────────────────────────────

    def export_debate(self, exchanges, daf_ref=""):
        lines = [f"# Masmid Debate Export — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"]
        if daf_ref:
            lines.append(f"**Daf Yomi**: {daf_ref}\n\n---\n")
        for ex in exchanges:
            ts = (ex.get("ts", ""))[:19].replace("T", " ")
            lines.append(f"\n**{ex['rabbi']}** [{ts}]\n{ex['text']}\n")
        return "\n".join(lines)
