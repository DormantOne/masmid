"""
agents.py — RabbiAgent and DreamAgent
===============================================================================
PROJECT MAP  (masmid/)
-----------------------------------------------
  main.py            Entry point
  config.py          Config constants
  llm.py             LLM abstraction
  models.py          Node, Edge, EdgeChannels
  vector_store.py    VectorStore
  knowledge_graph.py KnowledgeGraph
  rabbi_init.py      KG seed data
  agents.py          YOU ARE HERE — RabbiAgent, DreamAgent
  sefaria.py         Sefaria API
  orchestrator.py    DebateOrchestrator
  log_system.py      LogSystem
  auth.py            UserStore
  prompts.py         Prompt templates
  app.py             Flask app factory + routes
  templates.py       HTML templates
  daf_ingest.py      Daf segmentation + exchange tagging
  context_manager.py Context budget assembly

EXPORTS:
  RabbiAgent  — respond(), respond_to_human()
  DreamAgent  — start(), stop(), cycle(), status()

DEPENDS ON:
  config.TOP_K, config.DREAM_INTERVAL
  llm.llm_chat, llm.parse_json, llm.LLMError
  prompts.DEBATE_SYSTEM, prompts.HUMAN_RESPONSE_SYSTEM, prompts.DREAM_SYSTEM
  context_manager.assemble_debate_context, assemble_human_context
  daf_ingest.tag_exchange
  knowledge_graph.KnowledgeGraph, NODE_TYPE_LIMITS
  log_system.LogSystem
===============================================================================
"""

import json
import threading
from datetime import datetime
from typing import Optional

import config
from llm import llm_chat, parse_json, LLMError
from prompts import DEBATE_SYSTEM, HUMAN_RESPONSE_SYSTEM, DREAM_SYSTEM
from context_manager import assemble_debate_context, assemble_human_context
from daf_ingest import tag_exchange


class RabbiAgent:
    def __init__(self, kg, logger):
        self.kg   = kg
        self.log  = logger
        soul      = kg.get_soul()
        self.name = soul.content["name"] if soul else kg.name
        self.last_activated = []   # node names activated in last response

    def respond(self, daf_ref, segments, exchanges, opponent_last,
                opponent_name, convergence_note=""):
        """Generate a debate response using context-managed assembly.

        If convergence_note is set, the debate has been detected as circular
        and the model is instructed to find new ground or yield.

        Returns: (reply_text, segment_ranges) tuple.
        """
        ctx = assemble_debate_context(
            kg=self.kg,
            daf_ref=daf_ref,
            segments=segments,
            exchanges=exchanges,
            opponent_last=opponent_last,
            opponent_name=opponent_name,
            speaker_name=self.name,
        )
        if not ctx:
            return "[No soul node]", []

        system = DEBATE_SYSTEM.format(
            name           = ctx["name"],
            essence        = ctx["essence"],
            values         = ctx["values"],
            voice          = ctx["voice"],
            daf_ref        = ctx["daf_ref"],
            segment_index  = ctx["segment_index"],
            daf_excerpt    = ctx["daf_excerpt"],
            drives         = ctx["drives"],
            kg_context     = ctx["kg_context"],
            episodic       = ctx.get("episodic", ""),
            recent         = ctx["recent"],
            opponent       = ctx["opponent"],
            last_stmt      = ctx["last_stmt"],
            used_arguments = ctx["used_arguments"],
        )

        # Convergence injection — appended last for maximum prompt weight
        if convergence_note:
            system += "\n" + convergence_note

        seg_ranges = ctx.get("_segment_ranges", [])
        self.last_activated = ctx.get("_activated", [])
        try:
            reply = llm_chat(
                messages=[{"role": "user", "content": "Respond now."}],
                system=system,
            ).strip()
            self.log.log("debate", "rabbi_response", {
                "rabbi": self.name,
                "text": reply[:200],
                "activated": ctx.get("_activated", []),
                "segments": ctx.get("_segments_used", []),
                "converged": bool(convergence_note),
            }, tags=[self.name.lower().replace(" ", "_")])
            return reply, seg_ranges
        except LLMError as e:
            self.log.log("debate", "rabbi_error",
                         {"rabbi": self.name, "error": str(e)}, level="ERROR")
            print(f"[AGENT ERROR] {self.name} respond failed: {e}", flush=True)
            return f"[{self.name} pauses — LLM error: {e}]", seg_ranges

    def respond_to_human(self, daf_ref, segments, exchanges,
                         human_text, human_name):
        """Respond to a human student using context-managed assembly.

        Returns: (reply_text, segment_ranges) tuple.
        """
        ctx = assemble_human_context(
            kg=self.kg,
            daf_ref=daf_ref,
            segments=segments,
            exchanges=exchanges,
            human_text=human_text,
            human_name=human_name,
        )
        if not ctx:
            return "[No soul node]", []

        system = HUMAN_RESPONSE_SYSTEM.format(
            name           = ctx["name"],
            essence        = ctx["essence"],
            values         = ctx["values"],
            voice          = ctx["voice"],
            daf_ref        = ctx["daf_ref"],
            segment_index  = ctx["segment_index"],
            daf_excerpt    = ctx["daf_excerpt"],
            drives         = ctx["drives"],
            kg_context     = ctx["kg_context"],
            episodic       = ctx.get("episodic", ""),
            recent         = ctx["recent"],
            human_name     = ctx["human_name"],
            human_stmt     = ctx["human_stmt"],
        )
        seg_ranges = ctx.get("_segment_ranges", [])
        self.last_activated = ctx.get("_activated", [])
        try:
            reply = llm_chat(
                messages=[{"role": "user", "content": "Respond to the student now."}],
                system=system,
            ).strip()
            self.log.log("debate", "rabbi_to_human", {
                "rabbi": self.name,
                "human": human_name,
                "text": reply[:200],
            }, tags=[self.name.lower().replace(" ", "_"), "human_interaction"])
            return reply, seg_ranges
        except LLMError as e:
            self.log.log("debate", "rabbi_error",
                         {"rabbi": self.name, "error": str(e)}, level="ERROR")
            print(f"[AGENT ERROR] {self.name} respond_to_human failed: {e}", flush=True)
            return f"[{self.name} turns to you — LLM error: {e}]", seg_ranges


# ==============================================================================
#  DREAM AGENT
# ==============================================================================

class DreamAgent:
    def __init__(self, kg, logger):
        self.kg    = kg
        self.log   = logger
        soul       = kg.get_soul()
        self.name  = soul.content["name"] if soul else kg.name
        self._timer: Optional[threading.Timer] = None
        self._running = False
        self.cycles   = 0
        self.last_ran: Optional[str] = None
        self.last_summary = ""

    def start(self):
        self._running = True
        self._schedule()

    def stop(self):
        self._running = False
        if self._timer:
            self._timer.cancel()

    def _schedule(self):
        if self._running:
            self._timer = threading.Timer(config.DREAM_INTERVAL, self._run)
            self._timer.daemon = True
            self._timer.start()

    def _run(self):
        try:
            self.cycle()
        except Exception as e:
            self.log.log("dream", "error",
                         {"rabbi": self.name, "error": str(e)}, level="ERROR")
        finally:
            self._schedule()

    def cycle(self, manual=False):
        self.cycles += 1
        self.last_ran = datetime.now().isoformat()
        self.kg.decay_salience()
        self.kg.buildup_drives()

        # Consolidate before dreaming — merges duplicates, enforces limits
        self.kg.consolidate()

        soul   = self.kg.get_soul()
        sample = self.kg.sample_nodes(5)
        edges  = [e.to_dict() for e in self.kg.edges
                  if e.from_id in {n.id for n in sample}
                  or e.to_id in {n.id for n in sample}]
        if not soul or not sample:
            return {}

        # Include census so the LLM sees current counts vs limits
        payload = {
            "soul":   soul.summary(),
            "census": self.kg.node_census(),
            "goals":  [g.summary() for g in self.kg.get_by_type("goal")],
            "nodes":  [{"id": n.id,
                        "brief": f"[{n.type}] {n.name} sal={n.salience:.2f}",
                        "summary": n.summary()} for n in sample],
            "edges":  edges[:8],
        }
        try:
            raw = llm_chat(
                messages=[{"role": "user", "content": json.dumps(payload)}],
                system=DREAM_SYSTEM.format(name=self.name))
            result = parse_json(raw)
        except LLMError as e:
            self.log.log("dream", "llm_error",
                         {"rabbi": self.name, "error": str(e)}, level="ERROR")
            return {}

        actions_done = []
        for a in result.get("actions", []):
            out = self._exec(a)
            if out:
                actions_done.append({"type": a.get("type"), "out": out})

        ts      = datetime.now().strftime("%Y-%m-%d %H:%M")
        summary = result.get("meta_summary", "")
        obs     = result.get("observations", [])
        self.last_summary = summary

        block = (f"\n#### {self.name} Dream #{self.cycles} [{ts}]\n\n"
                 + (f"**Obs**: {'; '.join(obs)}\n\n" if obs else "")
                 + (f"**Summary**: {summary}\n\n" if summary else "")
                 + (f"**Actions**: {len(actions_done)}\n\n"))
        self.log.append_meta(block)
        self.log.log("dream", "cycle_done", {
            "rabbi": self.name, "cycle": self.cycles,
            "actions": len(actions_done), "summary": summary[:120],
        })
        return {"summary": summary, "observations": obs,
                "actions": actions_done}

    def _exec(self, a):
        t = a.get("type", "")
        try:
            if t == "merge":
                return str(self.kg.do_merge(a["keep_id"], a["remove_id"]))
            if t == "reweight":
                return str(self.kg.do_reweight(
                    a["edge_id"], a["channel"], a["new_value"]))
            if t == "fade":
                return str(self.kg.do_fade(a["node_id"]))
            if t == "strengthen":
                return str(self.kg.do_strengthen(a["node_id"]))
            if t == "spawn":
                # Block spawn if type is already at its limit
                ntype = a.get("node_type", "")
                from knowledge_graph import NODE_TYPE_LIMITS
                limit = NODE_TYPE_LIMITS.get(ntype, 999)
                current = len(self.kg.get_by_type(ntype))
                if current >= limit:
                    return f"blocked: {ntype} at limit ({current}/{limit})"
                n = self.kg.do_spawn(
                    ntype, a.get("content", {}),
                    a.get("handles", {"surface":
                        a.get("content", {}).get("name", "new")}))
                return f"spawned {n.id}"
            if t == "flag":
                self.kg.do_flag(a["node_id"], a["message"])
                return "flagged"
        except Exception as e:
            return f"err:{e}"
        return ""

    def status(self):
        return {
            "name": self.name, "cycles": self.cycles,
            "last_ran": self.last_ran, "running": self._running,
            "last_summary": self.last_summary[:120],
            "census": self.kg.node_census(),
        }