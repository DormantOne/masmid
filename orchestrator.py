"""
orchestrator.py — DebateOrchestrator: manages debate flow
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
  agents.py          RabbiAgent, DreamAgent
  sefaria.py         Sefaria API
  orchestrator.py    YOU ARE HERE — DebateOrchestrator
  log_system.py      LogSystem
  auth.py            UserStore
  prompts.py         Prompt templates
  app.py             Flask app factory + routes
  templates.py       HTML templates
  daf_ingest.py      Daf segmentation + exchange tagging
  context_manager.py Context budget assembly

EXPORTS:
  DebateOrchestrator — load_daf, step, human_say, start_auto, stop_auto,
                       export, status

CHANGES FROM v4 MONOLITH:
  - Daf is segmented on load (daf_ingest.segment_daf)
  - Each exchange gets a 1-2 word tag (daf_ingest.tag_exchange)
  - Turn balance uses explicit next_speaker tracking instead of modulo
  - After human_say, next step() picks up the correct speaker
  - Agents receive segments instead of raw daf_text

DEPENDS ON:
  config.DEBATE_DELAY
  agents.RabbiAgent
  sefaria.fetch_daf_yomi, sefaria.fetch_daf_text
  daf_ingest.segment_daf, tag_exchange
  log_system.LogSystem
===============================================================================
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Optional

import config
from sefaria import fetch_daf_yomi, fetch_daf_text
from daf_ingest import segment_daf, tag_exchange


class DebateOrchestrator:
    def __init__(self, hillel, shammai, logger):
        self.hillel    = hillel
        self.shammai   = shammai
        self.log       = logger
        self.daf: Optional[dict]      = None
        self.daf_text: Optional[dict] = None
        self.segments: list           = []     # list[DafSegment]
        self.exchanges: list[dict]    = []
        self.turn      = 0
        # Explicit speaker tracking: always alternate, survives human interrupts
        self._next_speaker = "hillel"
        self._auto     = False
        self._thread: Optional[threading.Thread] = None

    def _flip_speaker(self):
        """Toggle next speaker. Called after every rabbi response."""
        self._next_speaker = (
            "shammai" if self._next_speaker == "hillel" else "hillel"
        )

    def _get_speaker_and_opponent(self):
        """Return (speaker_agent, opponent_agent) based on current turn."""
        if self._next_speaker == "hillel":
            return self.hillel, self.shammai
        return self.shammai, self.hillel

    # -- Daf loading -----------------------------------------------------------

    def load_daf(self, ref=None):
        self.daf = fetch_daf_yomi() if not ref else {"ref": ref, "he_ref": "", "url": ""}
        self.daf_text = fetch_daf_text(self.daf["ref"])
        self.exchanges = []
        self.turn = 0
        self._next_speaker = "hillel"

        # Segment the daf (LLM-powered, runs once)
        print(f"[INGEST] Segmenting daf: {self.daf['ref']}...", flush=True)
        self.segments = segment_daf(self.daf_text)
        print(f"[INGEST] Created {len(self.segments)} segments: "
              f"{[s.title for s in self.segments]}", flush=True)

        self.log.log("debate", "daf_loaded", {
            "ref": self.daf["ref"],
            "length": self.daf_text.get("length", 0),
            "segments": len(self.segments),
            "segment_titles": [s.title for s in self.segments],
        })
        return {
            "daf": self.daf,
            "text": self.daf_text,
            "segments": [
                {"index": s.index, "title": s.title, "summary": s.summary,
                 "line_start": s.line_start, "line_end": s.line_end}
                for s in self.segments
            ],
        }

    # -- Debate step -----------------------------------------------------------

    def step(self):
        """One debate turn. Speaker alternates strictly."""
        if not self.daf_text:
            return None

        speaker, opponent = self._get_speaker_and_opponent()
        last = self.exchanges[-1]["text"] if self.exchanges else None

        reply, seg_ranges = speaker.respond(
            daf_ref=self.daf["ref"],
            segments=self.segments,
            exchanges=self.exchanges[-12:],   # generous history, context_manager compresses
            opponent_last=last,
            opponent_name=opponent.name,
        )

        # Tag the exchange and attach segment ranges for UI highlighting
        ex = {
            "rabbi": speaker.name,
            "text": reply,
            "turn": self.turn,
            "ts": datetime.now().isoformat(),
            "daf_ref": self.daf["ref"],
            "seg_ranges": seg_ranges,
        }
        ex["tag"] = tag_exchange(ex)

        self.exchanges.append(ex)
        self.turn += 1
        self._flip_speaker()
        return ex

    # -- Human interaction -----------------------------------------------------

    def human_say(self, text, display_name="A Student"):
        """Human enters the debate. Both rabbis respond, then debate resumes
        with the correct next speaker."""
        if not self.daf_text:
            return {"error": "No Daf loaded."}

        # Record human entry
        human_ex = {
            "rabbi": display_name, "text": text, "turn": self.turn,
            "ts": datetime.now().isoformat(), "daf_ref": self.daf["ref"],
            "is_human": True,
        }
        human_ex["tag"] = tag_exchange(human_ex)
        self.exchanges.append(human_ex)
        self.turn += 1

        self.log.log("debate", "human_entry",
                     {"name": display_name, "text": text[:200]},
                     tags=["human_interaction"])

        recent_exchanges = list(self.exchanges[-12:])

        # Both rabbis respond in parallel
        def hillel_task():
            return self.hillel.respond_to_human(
                daf_ref=self.daf["ref"],
                segments=self.segments,
                exchanges=recent_exchanges,
                human_text=text,
                human_name=display_name,
            )

        def shammai_task():
            return self.shammai.respond_to_human(
                daf_ref=self.daf["ref"],
                segments=self.segments,
                exchanges=recent_exchanges,
                human_text=text,
                human_name=display_name,
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            h_future = pool.submit(hillel_task)
            s_future = pool.submit(shammai_task)
            h_reply, h_seg_ranges = h_future.result()
            s_reply, s_seg_ranges = s_future.result()

        h_ex = {
            "rabbi": self.hillel.name, "text": h_reply, "turn": self.turn,
            "ts": datetime.now().isoformat(), "daf_ref": self.daf["ref"],
            "responding_to_human": True,
            "seg_ranges": h_seg_ranges,
        }
        h_ex["tag"] = tag_exchange(h_ex)
        self.exchanges.append(h_ex)
        self.turn += 1

        s_ex = {
            "rabbi": self.shammai.name, "text": s_reply, "turn": self.turn,
            "ts": datetime.now().isoformat(), "daf_ref": self.daf["ref"],
            "responding_to_human": True,
            "seg_ranges": s_seg_ranges,
        }
        s_ex["tag"] = tag_exchange(s_ex)
        self.exchanges.append(s_ex)
        self.turn += 1

        # After both respond to human, next step() resumes with whoever was
        # supposed to go next BEFORE the human interrupted.
        # (We don't call _flip_speaker here — that's only for debate steps.)

        return {
            "human": human_ex, "hillel": h_ex, "shammai": s_ex,
            "turn": self.turn,
        }

    # -- Auto-debate -----------------------------------------------------------

    def start_auto(self):
        if self._auto:
            return
        self._auto = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop_auto(self):
        self._auto = False

    def _loop(self):
        while self._auto:
            if self.daf_text:
                self.step()
                time.sleep(config.DEBATE_DELAY)
            else:
                time.sleep(2.0)

    # -- Export / status -------------------------------------------------------

    def export(self):
        daf_ref = self.daf["ref"] if self.daf else ""
        return self.log.export_debate(self.exchanges, daf_ref)

    def status(self):
        return {
            "daf": self.daf,
            "turn": self.turn,
            "exchanges": len(self.exchanges),
            "auto": self._auto,
            "next_speaker": self._next_speaker,
            "segments": len(self.segments),
            "last": self.exchanges[-1] if self.exchanges else None,
        }
