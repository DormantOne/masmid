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
  knowledge_graph.KnowledgeGraph
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

    def respond(self, daf_ref, segments, exchanges, opponent_last, opponent_name):
        """Generate a debate response using context-managed assembly.

        Returns: (reply_text, segment_ranges) tuple.
            segment_ranges is a list of {index, title, line_start, line_end}
            for the daf segments that were loaded into this response's context.
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
            recent         = ctx["recent"],
            opponent       = ctx["opponent"],
            last_stmt      = ctx["last_stmt"],
            used_arguments = ctx["used_arguments"],
        )
        seg_ranges = ctx.get("_segment_ranges", [])
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
            }, tags=[self.name.lower().replace(" ", "_")])
            return reply, seg_ranges
        except LLMError as e:
            self.log.log("debate", "rabbi_error",
                         {"rabbi": self.name, "error": str(e)}, level="ERROR")
            return f"[{self.name} pauses in contemplation...]", seg_ranges

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
            recent         = ctx["recent"],
            human_name     = ctx["human_name"],
            human_stmt     = ctx["human_stmt"],
        )
        seg_ranges = ctx.get("_segment_ranges", [])
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
            return f"[{self.name} turns to you thoughtfully...]", seg_ranges


# ==============================================================================
#  DREAM AGENT  (unchanged except for updated project map)
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

        soul   = self.kg.get_soul()
        sample = self.kg.sample_nodes(5)
        edges  = [e.to_dict() for e in self.kg.edges
                  if e.from_id in {n.id for n in sample}
                  or e.to_id in {n.id for n in sample}]
        if not soul or not sample:
            return {}

        payload = {
            "soul":  soul.summary(),
            "goals": [g.summary() for g in self.kg.get_by_type("goal")],
            "nodes": [{"id": n.id,
                       "brief": f"[{n.type}] {n.name} sal={n.salience:.2f}",
                       "summary": n.summary()} for n in sample],
            "edges": edges[:8],
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
        return {"summary": summary, "observations": obs, "actions": actions_done}

    def _exec(self, a):
        t = a.get("type", "")
        try:
            if t == "reweight":
                return str(self.kg.do_reweight(
                    a["edge_id"], a["channel"], a["new_value"]))
            if t == "fade":
                return str(self.kg.do_fade(a["node_id"]))
            if t == "strengthen":
                return str(self.kg.do_strengthen(a["node_id"]))
            if t == "spawn":
                n = self.kg.do_spawn(
                    a["node_type"], a.get("content", {}),
                    a.get("handles", {"surface": a.get("content", {}).get("name", "new")}))
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
        }
