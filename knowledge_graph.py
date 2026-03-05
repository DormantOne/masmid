"""
knowledge_graph.py — KnowledgeGraph: nodes, edges, queries, dream ops
===============================================================================
PROJECT MAP  (masmid/)
-----------------------------------------------
  main.py            Entry point
  config.py          Config constants
  llm.py             LLM abstraction
  models.py          Node, Edge, EdgeChannels
  vector_store.py    VectorStore
  knowledge_graph.py YOU ARE HERE — KnowledgeGraph class
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
  KnowledgeGraph — add_node, get_node, query, query_with_lens, summary,
                   full_export, buildup_drives, decay_salience,
                   do_reweight, do_fade, do_strengthen, do_spawn, do_flag

DEPENDS ON:
  config.DATA_DIR, config.TOP_K, config.DRIVE_BUILDUP_RATE, config.SALIENCE_DECAY
  models.Node, models.Edge, models.EdgeChannels
  vector_store.VectorStore
===============================================================================
"""

import json
import random
import threading
import uuid
from datetime import datetime

import config
from models import Node, Edge, EdgeChannels
from vector_store import VectorStore


class KnowledgeGraph:
    def __init__(self, name, vs):
        self.name  = name
        self.vs    = vs
        self.nodes = {}   # dict[str, Node]
        self.edges = []   # list[Edge]
        self._lock = threading.Lock()
        self._file = config.DATA_DIR / f"kg_{name}.json"
        self._load()

    # -- Persistence -----------------------------------------------------------

    def _load(self):
        if not self._file.exists():
            return
        try:
            d = json.loads(self._file.read_text())
            self.nodes = {k: Node.from_dict(v) for k, v in d.get("nodes", {}).items()}
            self.edges = [Edge.from_dict(e) for e in d.get("edges", [])]
        except Exception as e:
            print(f"[KG LOAD {self.name}] {e}", flush=True)

    def save(self):
        with self._lock:
            self._file.write_text(json.dumps({
                "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
                "edges": [e.to_dict() for e in self.edges],
                "saved": datetime.now().isoformat(),
            }, indent=2))

    # -- Node CRUD -------------------------------------------------------------

    def add_node(self, n, embed=True):
        with self._lock:
            self.nodes[n.id] = n
        if embed:
            for handle, text in n.handles.items():
                self.vs.upsert(n.id, handle, text)
        self.save()

    def get_node(self, nid):
        return self.nodes.get(nid)

    def get_soul(self):
        return next((n for n in self.nodes.values() if n.type == "soul"), None)

    def get_by_type(self, t):
        return [n for n in self.nodes.values() if n.type == t]

    def remove_node(self, nid):
        with self._lock:
            self.nodes.pop(nid, None)
            self.edges = [e for e in self.edges
                          if e.from_id != nid and e.to_id != nid]
        self.vs.remove(nid)
        self.save()

    # -- Edge CRUD -------------------------------------------------------------

    def add_edge(self, from_id, to_id, channels):
        e = Edge(id=str(uuid.uuid4())[:8], from_id=from_id,
                 to_id=to_id, channels=channels)
        with self._lock:
            self.edges.append(e)
        self.save()
        return e

    def get_edge(self, eid):
        return next((e for e in self.edges if e.id == eid), None)

    def get_node_edges(self, nid):
        return [e for e in self.edges if e.from_id == nid or e.to_id == nid]

    def get_neighbors(self, nid):
        out = []
        for e in self.edges:
            if e.channels.total() < 0.1:
                continue
            other = (e.to_id if e.from_id == nid
                     else (e.from_id if not e.directed and e.to_id == nid
                           else None))
            if other:
                n = self.get_node(other)
                if n:
                    out.append(n)
        return out

    # -- Queries ---------------------------------------------------------------

    def query(self, text, top_k=None):
        top_k = top_k or config.TOP_K
        hits = self.vs.search(text, top_k=top_k + 2, kg=self,
                              exclude_types=["soul"])
        result = []
        for h in hits:
            n = self.get_node(h["node_id"])
            if n:
                h["node"] = n
                n.touch()
                result.append(h)
        return result[:top_k]

    def query_with_lens(self, lens_text, top_k=8):
        hits = self.vs.search(lens_text, top_k=top_k, kg=self)
        result = []
        for h in hits:
            n = self.get_node(h["node_id"])
            if n:
                h["node"] = n
                result.append(h)
        return result

    def sample_nodes(self, n=5):
        nodes = list(self.nodes.values())
        if not nodes:
            return []
        weights = [max(0.01, nd.salience) for nd in nodes]
        total = sum(weights)
        probs = [w / total for w in weights]
        k = min(n, len(nodes))
        idxs = set(random.choices(range(len(nodes)), weights=probs, k=k))
        return [nodes[i] for i in idxs]

    # -- Drive / salience dynamics ---------------------------------------------

    def buildup_drives(self):
        for n in self.nodes.values():
            if n.type == "drive":
                n.content["intensity"] = min(
                    1.0, n.content.get("intensity", 0.0) + config.DRIVE_BUILDUP_RATE)
        self.save()

    def discharge_drive(self, nid, amount=0.12):
        n = self.get_node(nid)
        if n and n.type == "drive":
            n.content["intensity"] = max(
                0.0, n.content.get("intensity", 0.0) - amount)
            self.save()

    def decay_salience(self):
        for n in self.nodes.values():
            if n.type != "soul":
                n.salience = max(0.05, n.salience * config.SALIENCE_DECAY)
        self.save()

    # -- Dream-cycle operations ------------------------------------------------

    def do_reweight(self, eid, channel, val):
        e = self.get_edge(eid)
        if e and hasattr(e.channels, channel):
            setattr(e.channels, channel, max(0.0, min(1.0, float(val))))
            e.updated = datetime.now().isoformat()
            self.save()
            return True
        return False

    def do_fade(self, nid, amount=0.08):
        n = self.get_node(nid)
        if n and n.type != "soul":
            n.salience = max(0.0, n.salience - amount)
            self.save()
            return True
        return False

    def do_strengthen(self, nid, amount=0.1):
        n = self.get_node(nid)
        if n:
            n.salience = min(1.0, n.salience + amount)
            self.save()
            return True
        return False

    def do_spawn(self, ntype, content, handles, reason=""):
        prefix = {"habit": "h_", "drive": "d_", "skill": "sk_",
                  "goal": "g_"}.get(ntype, "n_")
        n = Node(
            id=f"{prefix}{uuid.uuid4().hex[:6]}",
            type=ntype,
            name=content.get("name", f"{ntype}_{uuid.uuid4().hex[:4]}"),
            content=content, handles=handles, salience=0.4,
        )
        self.add_node(n)
        return n

    def do_flag(self, nid, message):
        n = self.get_node(nid)
        if n:
            n.flags.append({"message": message, "ts": datetime.now().isoformat()})
            self.save()

    # -- Summaries / exports ---------------------------------------------------

    def summary(self):
        return {
            "name":        self.name,
            "nodes":       len(self.nodes),
            "edges":       len(self.edges),
            "by_type":     {t: len(self.get_by_type(t))
                            for t in ["soul", "goal", "habit", "skill", "drive"]},
            "drives":      [
                {"id": n.id, "name": n.name,
                 "intensity": round(n.content.get("intensity", 0), 3)}
                for n in self.get_by_type("drive")],
            "top_salience": [
                {"id": n.id, "name": n.name, "type": n.type,
                 "salience": round(n.salience, 3)}
                for n in sorted(self.nodes.values(),
                                key=lambda x: x.salience, reverse=True)[:8]],
            "flagged":     [
                {"id": n.id, "name": n.name, "msg": n.flags[-1]["message"]}
                for n in self.nodes.values() if n.flags],
        }

    def full_export(self):
        nodes_out = []
        for n in self.nodes.values():
            node_edges = self.get_node_edges(n.id)
            edge_list = []
            for e in node_edges:
                other_id = e.to_id if e.from_id == n.id else e.from_id
                other = self.get_node(other_id)
                dom_ch = max(e.channels.NAMES,
                             key=lambda k: getattr(e.channels, k))
                edge_list.append({
                    "edge_id":    e.id,
                    "direction":  "out" if e.from_id == n.id else "in",
                    "other_id":   other_id,
                    "other_name": other.name if other else "?",
                    "other_type": other.type if other else "?",
                    "dominant_channel": dom_ch,
                    "channel_val": round(getattr(e.channels, dom_ch), 3),
                    "total":      round(e.channels.total(), 3),
                })
            nodes_out.append({
                "id":       n.id, "type": n.type, "name": n.name,
                "salience": round(n.salience, 3),
                "content":  n.content,
                "handles":  n.handles,
                "tags":     n.tags,
                "flags":    n.flags,
                "edges":    edge_list,
                "summary":  n.summary(),
            })
        nodes_out.sort(key=lambda x: -x["salience"])
        return {"nodes": nodes_out, "total_edges": len(self.edges)}
