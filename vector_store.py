"""
vector_store.py — Embedding-based vector search with JSON persistence
═══════════════════════════════════════════════════════════════════════════════
PROJECT MAP  (masmid/)
─────────────────────────────────────────────────
  main.py            Entry point
  config.py          Config constants
  llm.py             LLM abstraction
  models.py          Node, Edge, EdgeChannels
  vector_store.py  ◄ YOU ARE HERE — VectorStore (upsert, search, remove)
  knowledge_graph.py KnowledgeGraph
  rabbi_init.py      KG seed data
  agents.py          RabbiAgent, DreamAgent
  sefaria.py         Sefaria API
  orchestrator.py    DebateOrchestrator
  log_system.py      LogSystem
  auth.py            UserStore
  prompts.py         Prompt templates
  app.py             Flask app factory + routes
  templates.py       HTML templates

EXPORTS:
  VectorStore  — upsert(), remove(), search(), purge_stale()

DEPENDS ON:
  config.DATA_DIR, config.TOP_K
  llm.llm_embed, llm.LLMError
═══════════════════════════════════════════════════════════════════════════════
"""

import json
from typing import Optional

import numpy as np

import config
from llm import llm_embed, LLMError


class VectorStore:
    def __init__(self, name: str):
        self._file = config.DATA_DIR / f"embeddings_{name}.json"
        self._store: dict = {}
        self._expected_dim: Optional[int] = None
        self._load()

    def _load(self):
        if self._file.exists():
            try:
                self._store = json.loads(self._file.read_text())
            except Exception:
                pass

    def _save(self):
        self._file.write_text(json.dumps(self._store))

    def purge_stale(self):
        if not self._store or self._expected_dim is None:
            return
        stale = [k for k, v in self._store.items() if len(v) != self._expected_dim]
        if stale:
            print(f"[EMBED] Purging {len(stale)} stale embeddings "
                  f"(dim≠{self._expected_dim})", flush=True)
            for k in stale:
                del self._store[k]
            self._save()

    def upsert(self, node_id: str, handle: str, text: str):
        try:
            emb = llm_embed(text)
            self._expected_dim = len(emb)
            self._store[f"{node_id}:{handle}"] = emb
            self._save()
        except LLMError as e:
            print(f"[EMBED WARN] {e}", flush=True)

    def remove(self, node_id: str):
        keys = [k for k in self._store if k.startswith(f"{node_id}:")]
        for k in keys:
            del self._store[k]
        self._save()

    def search(self, query: str, top_k: int = None, exclude_types=None, kg=None):
        """Cosine similarity search. Returns list of {node_id, handle, score}."""
        top_k = top_k or config.TOP_K
        if not self._store:
            return []
        try:
            qv = np.array(llm_embed(query), dtype=np.float32)
        except LLMError:
            return []

        qn = np.linalg.norm(qv)
        if qn == 0:
            return []
        qv /= qn
        qdim = len(qv)

        results = []
        for key, emb in self._store.items():
            if ":" not in key:
                continue
            node_id, handle = key.split(":", 1)
            if node_id.startswith("edge_"):
                continue
            if exclude_types and kg:
                nd = kg.get_node(node_id)
                if nd and nd.type in exclude_types:
                    continue
            v = np.array(emb, dtype=np.float32)
            if len(v) != qdim:
                continue
            vn = np.linalg.norm(v)
            if vn == 0:
                continue
            score = float(np.dot(qv, v / vn))
            results.append({"node_id": node_id, "handle": handle, "score": score})

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
