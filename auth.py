"""
auth.py — JSON-backed user authentication store
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
  log_system.py      LogSystem
  auth.py          ◄ YOU ARE HERE — UserStore (register, authenticate)
  prompts.py         Prompt templates
  app.py             Flask app factory + routes
  templates.py       HTML templates

EXPORTS:
  UserStore  — register(), authenticate(), get_display_name()

DEPENDS ON:
  config.DATA_DIR
═══════════════════════════════════════════════════════════════════════════════
"""

import json
import threading
from datetime import datetime

from werkzeug.security import generate_password_hash, check_password_hash

import config


class UserStore:
    """Simple JSON-backed user store for registration and login."""

    def __init__(self):
        self._file = config.DATA_DIR / "users.json"
        self._lock = threading.Lock()
        self._users: dict = {}  # {username: {password_hash, display_name, created}}
        self._load()

    def _load(self):
        if self._file.exists():
            try:
                self._users = json.loads(self._file.read_text())
            except Exception:
                self._users = {}

    def _save(self):
        self._file.write_text(json.dumps(self._users, indent=2))

    def register(self, username: str, password: str,
                 display_name: str = "") -> tuple[bool, str]:
        username = username.strip().lower()
        if not username or len(username) < 2:
            return False, "Username must be at least 2 characters."
        if len(username) > 40:
            return False, "Username too long (max 40)."
        if not password or len(password) < 4:
            return False, "Password must be at least 4 characters."
        with self._lock:
            if username in self._users:
                return False, "Username already taken."
            self._users[username] = {
                "password_hash": generate_password_hash(password),
                "display_name": display_name.strip() or username,
                "created": datetime.now().isoformat(),
            }
            self._save()
        return True, "Account created."

    def authenticate(self, username: str, password: str) -> tuple[bool, str]:
        username = username.strip().lower()
        with self._lock:
            user = self._users.get(username)
        if not user:
            return False, "User not found."
        if not check_password_hash(user["password_hash"], password):
            return False, "Incorrect password."
        return True, user.get("display_name", username)

    def get_display_name(self, username: str) -> str:
        with self._lock:
            u = self._users.get(username.strip().lower(), {})
        return u.get("display_name", username)
