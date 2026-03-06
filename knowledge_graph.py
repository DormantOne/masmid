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
  NODE_TYPE_LIMITS — dict of per-type max node counts
  KnowledgeGraph — add_node, get_node, query, query_with_lens, summary,
                   full_export, buildup_drives, decay_salience,
                   do_reweight, do_fade, do_strengthen, do_spawn, do_flag,
                   consolidate, do_merge, node_census

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
from collections import Counter, defaultdict
from datetime import datetime

import config
from models import Node, Edge, EdgeChannels
from vector_store import VectorStore


# Hard limits per node type (memory/encounter managed by episodic.py)
NODE_TYPE_LIMITS = {
    "goal": 3,
    "habit": 8,
    "skill": 6,
    "drive": 4,
    "definition": 3,
}


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
        """Graph-aware retrieval: vector seeds → 1-hop edge walk → merge."""
        top_k = top_k or config.TOP_K

        seed_count = min(top_k + 4, top_k * 2)
        seeds = self.vs.search(text, top_k=seed_count, kg=self,
                               exclude_types=["soul"])

        best = {}

        def _update_best(node_id, handle, score):
            if node_id not in best or score > best[node_id]["score"]:
                best[node_id] = {
                    "node_id": node_id, "handle": handle, "score": score
                }

        seed_ids = set()
        for h in seeds:
            _update_best(h["node_id"], h["handle"], h["score"])
            seed_ids.add(h["node_id"])

        walk_from = list(seed_ids)[:top_k]
        for src_id in walk_from:
            src_score = best[src_id]["score"]
            for edge in self.edges:
                if edge.from_id == src_id:
                    neighbor_id = edge.to_id
                elif (not edge.directed) and edge.to_id == src_id:
                    neighbor_id = edge.from_id
                else:
                    continue

                neighbor = self.get_node(neighbor_id)
                if not neighbor or neighbor.type == "soul":
                    continue

                ch = edge.channels
                edge_strength = (
                    ch.semantic * 1.0 +
                    ch.causal * 1.2 +
                    ch.reinforcing * 1.0 +
                    ch.emotional * 0.8 +
                    ch.conflict * 0.9 +
                    ch.temporal * 0.5
                )

                if edge_strength < 0.1:
                    continue

                decay = 0.6
                neighbor_score = src_score * edge_strength * neighbor.salience * decay

                handle = "surface" if "surface" in neighbor.handles else (
                    list(neighbor.handles.keys())[0] if neighbor.handles else "?")

                _update_best(neighbor_id, handle, neighbor_score)

        result = []
        for node_id, hit in best.items():
            n = self.get_node(node_id)
            if n:
                hit["node"] = n
                n.touch()
                result.append(hit)

        result.sort(key=lambda x: x["score"], reverse=True)
        return result[:top_k]

    def query_with_lens(self, lens_text, top_k=8):
        """Lens query — same graph walk but includes soul nodes."""
        hits = self.vs.search(lens_text, top_k=top_k, kg=self)

        best = {}
        seed_ids = set()
        for h in hits:
            nid = h["node_id"]
            if nid not in best or h["score"] > best[nid]["score"]:
                best[nid] = h
            seed_ids.add(nid)

        for src_id in list(seed_ids)[:top_k]:
            src_score = best[src_id]["score"]
            for edge in self.edges:
                if edge.from_id == src_id:
                    nb_id = edge.to_id
                elif (not edge.directed) and edge.to_id == src_id:
                    nb_id = edge.from_id
                else:
                    continue
                nb = self.get_node(nb_id)
                if not nb:
                    continue
                edge_str = edge.channels.total()
                if edge_str < 0.05:
                    continue
                nb_score = src_score * edge_str * nb.salience * 0.5
                handle = "surface" if "surface" in nb.handles else (
                    list(nb.handles.keys())[0] if nb.handles else "?")
                if nb_id not in best or nb_score > best[nb_id]["score"]:
                    best[nb_id] = {"node_id": nb_id, "handle": handle, "score": nb_score}

        result = []
        for nid, h in best.items():
            n = self.get_node(nid)
            if n:
                h["node"] = n
                result.append(h)

        result.sort(key=lambda x: x["score"], reverse=True)
        return result[:top_k]

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

    # -- Consolidation ---------------------------------------------------------

    def consolidate(self):
        """Merge near-duplicate nodes and enforce per-type limits.

        Called before each dream cycle to keep the KG lean.
        Two nodes are candidates for merging when they share a type
        and their names overlap by > 50% of meaningful words.
        The higher-salience node survives; edges are redirected.
        """
        by_type = defaultdict(list)
        for n in list(self.nodes.values()):
            if n.type in ("soul", "memory", "encounter"):
                continue
            by_type[n.type].append(n)

        merged_count = 0
        for ntype, nodes in by_type.items():
            if len(nodes) <= 1:
                continue
            # Best nodes first — they survive
            nodes.sort(key=lambda x: x.salience, reverse=True)
            survivors = []
            for node in nodes:
                merged = False
                for surv in survivors:
                    if self._name_similarity(surv.name, node.name) > 0.5:
                        self._merge_into(surv, node)
                        merged_count += 1
                        merged = True
                        break
                if not merged:
                    survivors.append(node)

        if merged_count:
            print(f"[KG {self.name}] Consolidated {merged_count} duplicate nodes",
                  flush=True)

        removed = self._enforce_limits()
        if merged_count or removed:
            self.save()
        return merged_count

    @staticmethod
    def _name_similarity(a, b):
        """Word-overlap Jaccard on meaningful words (min-denominator)."""
        stop = {
            "the", "a", "an", "in", "of", "and", "to", "for", "with",
            "on", "at", "by", "is", "rabbi", "shammai", "hillel",
            "halachic", "halacha", "practice", "daily", "immediate",
        }
        def words(s):
            return {w.lower() for w in s.split()
                    if w.lower() not in stop and len(w) > 2}
        wa, wb = words(a), words(b)
        if not wa or not wb:
            return 0.0
        return len(wa & wb) / min(len(wa), len(wb))

    def _merge_into(self, survivor, absorbed):
        """Fold absorbed node into survivor: best salience wins,
        handles/content merge, edges redirect, absorbed is deleted."""
        survivor.salience = max(survivor.salience, absorbed.salience)
        # Merge handles (survivor's take precedence)
        for k, v in absorbed.handles.items():
            if k not in survivor.handles:
                survivor.handles[k] = v
        # Merge content keys (survivor's take precedence)
        for k, v in absorbed.content.items():
            if k not in survivor.content:
                survivor.content[k] = v
        # Redirect edges
        for e in self.edges:
            if e.from_id == absorbed.id:
                e.from_id = survivor.id
            if e.to_id == absorbed.id:
                e.to_id = survivor.id
        # Deduplicate edges (same from+to after redirect) and remove self-loops
        seen = set()
        deduped = []
        for e in self.edges:
            if e.from_id == e.to_id:
                continue
            key = (e.from_id, e.to_id)
            if key not in seen:
                seen.add(key)
                deduped.append(e)
        self.edges = deduped
        # Re-embed survivor with potentially new handles
        for handle, text in survivor.handles.items():
            self.vs.upsert(survivor.id, handle, text)
        # Remove absorbed
        with self._lock:
            self.nodes.pop(absorbed.id, None)
        self.vs.remove(absorbed.id)

    def _enforce_limits(self):
        """Prune lowest-salience nodes when a type exceeds its limit."""
        total_removed = 0
        for ntype, limit in NODE_TYPE_LIMITS.items():
            typed = [n for n in self.nodes.values() if n.type == ntype]
            if len(typed) <= limit:
                continue
            typed.sort(key=lambda x: x.salience, reverse=True)
            for n in typed[limit:]:
                self.remove_node(n.id)
                total_removed += 1
            print(f"[KG {self.name}] Pruned {ntype}: kept top {limit}, "
                  f"removed {len(typed) - limit}", flush=True)
        return total_removed

    def do_merge(self, keep_id, remove_id):
        """Dream action: explicitly merge two nodes."""
        keep = self.get_node(keep_id)
        remove = self.get_node(remove_id)
        if not keep or not remove:
            return "node not found"
        if keep.type != remove.type:
            return "type mismatch"
        self._merge_into(keep, remove)
        self.save()
        return f"merged {remove_id} into {keep_id}"

    def node_census(self):
        """Return a compact string showing node counts by type vs limits."""
        counts = Counter(n.type for n in self.nodes.values())
        parts = []
        for ntype in ["soul", "goal", "habit", "skill", "drive",
                       "memory", "encounter", "definition"]:
            c = counts.get(ntype, 0)
            limit = NODE_TYPE_LIMITS.get(ntype)
            if limit:
                parts.append(f"{ntype}:{c}/{limit}")
            elif c > 0:
                parts.append(f"{ntype}:{c}")
        return "  ".join(parts)

    # -- Summaries / exports ---------------------------------------------------

    def summary(self):
        return {
            "name":        self.name,
            "nodes":       len(self.nodes),
            "edges":       len(self.edges),
            "by_type":     {t: len(self.get_by_type(t))
                            for t in ["soul", "goal", "habit", "skill", "drive",
                                      "memory", "encounter"]},
            "census":      self.node_census(),
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