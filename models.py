"""
models.py — Core data structures: Node, Edge, EdgeChannels
═══════════════════════════════════════════════════════════════════════════════
PROJECT MAP  (masmid/)
─────────────────────────────────────────────────
  main.py            Entry point
  config.py          Config constants
  llm.py             LLM abstraction
  models.py        ◄ YOU ARE HERE — Node, Edge, EdgeChannels dataclasses
  vector_store.py    VectorStore
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
  EdgeChannels  — 6-channel edge weight container
  Edge          — directed/undirected graph edge
  Node          — KG node with content, handles, salience, flags

USED BY:
  knowledge_graph.py, rabbi_init.py, agents.py, vector_store.py
═══════════════════════════════════════════════════════════════════════════════
"""

from dataclasses import dataclass, field
from datetime import datetime


# ─── Edge Channels ────────────────────────────────────────────────────────────

@dataclass
class EdgeChannels:
    semantic:    float = 0.0
    causal:      float = 0.0
    temporal:    float = 0.0
    emotional:   float = 0.0
    conflict:    float = 0.0
    reinforcing: float = 0.0

    NAMES = ["semantic", "causal", "temporal", "emotional", "conflict", "reinforcing"]

    def to_dict(self):
        return {k: getattr(self, k) for k in self.NAMES}

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: float(d.get(k, 0.0)) for k in cls.NAMES})

    def total(self):
        return sum(getattr(self, k) for k in self.NAMES) / len(self.NAMES)


# ─── Edge ─────────────────────────────────────────────────────────────────────

@dataclass
class Edge:
    id: str
    from_id: str
    to_id: str
    directed: bool = True
    channels: EdgeChannels = field(default_factory=EdgeChannels)
    created: str = field(default_factory=lambda: datetime.now().isoformat())
    updated: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self):
        return {
            "id": self.id, "from_id": self.from_id, "to_id": self.to_id,
            "directed": self.directed, "channels": self.channels.to_dict(),
            "created": self.created, "updated": self.updated,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            id=d["id"], from_id=d["from_id"], to_id=d["to_id"],
            directed=d.get("directed", True),
            channels=EdgeChannels.from_dict(d.get("channels", {})),
            created=d.get("created", ""), updated=d.get("updated", ""),
        )


# ─── Node ─────────────────────────────────────────────────────────────────────

@dataclass
class Node:
    id: str
    type: str
    name: str
    content: dict
    handles: dict
    salience: float = 0.5
    flags: list = field(default_factory=list)
    created: str = field(default_factory=lambda: datetime.now().isoformat())
    last_accessed: str = field(default_factory=lambda: datetime.now().isoformat())
    last_modified: str = field(default_factory=lambda: datetime.now().isoformat())
    tags: list = field(default_factory=list)

    _FIELDS = [
        "id", "type", "name", "content", "handles", "salience", "flags",
        "created", "last_accessed", "last_modified", "tags",
    ]

    def to_dict(self):
        return {k: getattr(self, k) for k in self._FIELDS}

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: d[k] for k in cls._FIELDS if k in d})

    def touch(self):
        self.last_accessed = datetime.now().isoformat()
        self.salience = min(1.0, self.salience + 0.03)

    def summary(self) -> str:
        c = self.content
        h = self.handles or {}
        if self.type == "soul":
            return (f"{c.get('name')}: {c.get('essence', '')} | "
                    f"values={c.get('values', [])} | voice={c.get('voice', '')}")
        if self.type == "drive":
            desc = c.get('description', '') or h.get('surface', '')
            return (f"{self.name} intensity={c.get('intensity', 0):.2f} "
                    f"({c.get('valence', 'approach')}): {desc}")
        if self.type == "habit":
            # Use the rich handle descriptions when available
            surface = h.get('surface', '') or h.get('behavior', '')
            trigger = c.get('trigger', '') or h.get('trigger', '')
            if surface:
                return f"When {trigger}: {surface} [str={c.get('strength', 0.5):.2f}]"
            return (f"trigger={trigger} → {c.get('behavior', '')} "
                    f"[str={c.get('strength', 0.5):.2f}]")
        if self.type == "goal":
            stmt = c.get('statement', '') or h.get('surface', '')
            return f"{self.name}: {stmt} [{c.get('status', 'active')}]"
        if self.type == "skill":
            desc = c.get('description', '') or h.get('surface', '')
            usage = h.get('usage', '')
            extra = f" — {usage}" if usage else ""
            return (f"{self.name}: {desc}{extra} "
                    f"[prof={c.get('proficiency', 0.5):.2f}]")
        if self.type == "memory":
            return c.get('summary', '')[:150]
        if self.type == "encounter":
            name = c.get('name', '?')
            sessions = c.get('sessions', 0)
            topics = ', '.join(c.get('topics', [])[-4:])
            return f"Student {name}: {sessions} visits, topics: {topics}"
        return str(c)[:200]
