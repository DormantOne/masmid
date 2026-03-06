"""
context_manager.py — Context budget assembly with progressive compression
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
  orchestrator.py    DebateOrchestrator
  log_system.py      LogSystem
  auth.py            UserStore
  prompts.py         Prompt templates
  app.py             Flask app factory + routes
  templates.py       HTML templates
  daf_ingest.py      Daf segmentation + exchange tagging
  context_manager.py YOU ARE HERE — assemble_context + compress_to_fit

DESIGN PHILOSOPHY:
  Push context close to the edge, then compress progressively if it doesn't fit.
  Serial compression passes — each pass removes the cheapest information first:

  COMPRESSION CASCADE:
    Pass 0: Full context (3 segments, 6 KG, 4 recent)
    Pass 1: Drop to 2 active segments
    Pass 2: Trim KG activations 6→4
    Pass 3: Trim recent exchanges 4→3
    Pass 4: Drop to 1 active segment
    Pass 5: Trim KG activations 4→2
    Pass 6: Trim recent exchanges 3→2
    Pass 7: Truncate active segment text to 200 chars + "..."
    Pass 8: Truncate KG entries to 80 chars each
    Pass 9: Drop segment index (just keep active)
    Pass 10: Nuclear — minimal context, 1 segment summary only

  Each pass re-estimates tokens. Stop as soon as we fit.

EXPORTS:
  assemble_debate_context(...)  -> dict
  assemble_human_context(...)   -> dict
  select_segments(...)          -> list[DafSegment]

DEPENDS ON:
  daf_ingest.DafSegment, compress_exchanges, tag_exchange
  config.CONTEXT_LIMIT
===============================================================================
"""

import re
from daf_ingest import DafSegment, compress_exchanges, tag_exchange
from episodic import get_episodic_context
import config


# ==============================================================================
#  TOKEN ESTIMATION
# ==============================================================================

def _estimate_tokens(text):
    """Rough token estimate: ~4 chars per token for English, ~3 for Hebrew mix."""
    if not text:
        return 0
    return len(text) // 4 + 1


def _estimate_context_tokens(ctx):
    """Estimate total tokens of all context fields that go into the prompt."""
    total = 0
    for key, val in ctx.items():
        if key.startswith("_"):
            continue
        if isinstance(val, str):
            total += _estimate_tokens(val)
    # Add ~80 tokens for the prompt template chrome (instructions, labels, etc.)
    return total + 80


# ==============================================================================
#  SEGMENT SELECTION  (citation-aware)
# ==============================================================================

def _extract_cited_indices(text):
    """Find [0], [1], etc. references in exchange text."""
    return set(int(m) for m in re.findall(r"\[(\d+)\]", text))


def select_segments(segments, query, exchanges=None, top_k=3):
    """Select the most relevant daf segments for a query.

    Scoring: keyword overlap + citation boost + position bias.
    No embedding calls — stays fast.
    """
    if not segments:
        return []
    if not query and not exchanges:
        result = [segments[0]]
        if len(segments) > 1:
            result.append(segments[-1])
        return result[:top_k]

    query_words = set()
    if query:
        query_words = set(w.lower() for w in query.split() if len(w) > 2)

    cite_counts = {}
    if exchanges:
        for ex in exchanges[-8:]:
            for idx in _extract_cited_indices(ex.get("text", "")):
                cite_counts[idx] = cite_counts.get(idx, 0) + 1

    scored = []
    for seg in segments:
        seg_text = (seg.title + " " + seg.summary + " " +
                    seg.en_text[:200]).lower()
        seg_words = set(w for w in seg_text.split() if len(w) > 2)
        overlap = len(query_words & seg_words) if query_words else 0
        cite_boost = cite_counts.get(seg.index, 0) * 3.0
        position_bonus = 0.1 * (1.0 - seg.index / max(len(segments), 1))
        scored.append((overlap + cite_boost + position_bonus, seg))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [seg for _, seg in scored[:top_k]]


# ==============================================================================
#  FORMATTING HELPERS
# ==============================================================================

def _format_segment_index(segments):
    if not segments:
        return "  (no segments)"
    return "\n".join(f"  {seg.header()}" for seg in segments)


def _format_active_segments(selected_segments, max_chars=None):
    """Full English text of selected segments, optionally truncated."""
    if not selected_segments:
        return "  (no segment selected)"
    parts = []
    for seg in selected_segments:
        text = seg.en_text
        if max_chars and len(text) > max_chars:
            text = text[:max_chars] + "..."
        parts.append(f"--- [{seg.index}] {seg.title} ---\n{text}")
    return "\n\n".join(parts)


def _format_compressed_history(compressed_tags, recent_exchanges, max_recent_chars=300):
    lines = []
    if compressed_tags:
        lines.append("Earlier: " + " | ".join(compressed_tags))
    for ex in recent_exchanges:
        speaker = ex.get("rabbi", "?")
        text = ex.get("text", "")[:max_recent_chars]
        is_human = ex.get("is_human", False)
        prefix = "Student" if is_human else speaker
        lines.append(f"  {prefix}: {text}")
    return "\n".join(lines) if lines else "  (Opening statement)"


def _format_kg_context(hits, kg, max_chars=180):
    kg_lines = []
    for h in hits:
        n = h["node"]
        kg_lines.append(
            f"  [{n.type}:{n.name}] {n.summary()[:max_chars]}"
        )
        if n.type == "drive":
            kg.discharge_drive(n.id, 0.10)
        if n.type in ("habit", "skill"):
            for nb in kg.get_neighbors(n.id):
                if nb.type == "drive":
                    kg.discharge_drive(nb.id, 0.05)
    return "\n".join(kg_lines) if kg_lines else "  (no relevant nodes activated)"


def _format_drives(kg):
    drives = kg.get_by_type("drive")
    if not drives:
        return "(no drives)"
    return " | ".join(
        f"{d.name}={d.content.get('intensity', 0):.2f}"
        for d in drives
    )


def _find_opponent_last(exchanges, opponent_name):
    for ex in reversed(exchanges):
        if ex.get("rabbi", "") == opponent_name and not ex.get("is_human"):
            return ex.get("text", "")
    if exchanges:
        return exchanges[-1].get("text", "")
    return ""


def _collect_speaker_tags(exchanges, speaker_name):
    tags = []
    for ex in exchanges:
        if ex.get("rabbi", "") == speaker_name and not ex.get("is_human"):
            tag = ex.get("tag", "")
            if tag and tag != "remark":
                tags.append(tag)
    return tags


# ==============================================================================
#  PROGRESSIVE COMPRESSION CASCADE
# ==============================================================================

def _compress_to_fit(
    ctx,
    kg,
    segments,
    selected,
    all_hits,
    compressed_tags,
    recent,
    exchanges,
    real_opponent_last,
    limit,
):
    """Progressively compress context until it fits within the token limit.

    Each pass strips the cheapest information first. Returns the modified ctx dict.
    """
    passes_applied = []

    def _measure():
        return _estimate_context_tokens(ctx)

    def _rebuild_segments(segs, max_chars=None):
        ctx["daf_excerpt"] = _format_active_segments(segs, max_chars=max_chars)
        ctx["_segments_used"] = [s.index for s in segs]
        ctx["_segment_ranges"] = [
            {"index": s.index, "title": s.title,
             "line_start": s.line_start, "line_end": s.line_end}
            for s in segs
        ]

    def _rebuild_kg(hits, max_chars=180):
        ctx["kg_context"] = _format_kg_context(hits, kg, max_chars=max_chars)
        ctx["_activated"] = [h["node"].name for h in hits]

    def _rebuild_history(tags, rec, max_chars=300):
        ctx["recent"] = _format_compressed_history(tags, rec, max_recent_chars=max_chars)

    # Check if we already fit
    est = _measure()
    if est <= limit:
        return ctx, passes_applied

    # Pass 0: Drop episodic memories (cheapest to lose)
    if ctx.get("episodic"):
        ctx["episodic"] = ""
        passes_applied.append("episodic:dropped")
        if _measure() <= limit:
            return ctx, passes_applied

    # Pass 1: Drop to 2 active segments
    if len(selected) > 2:
        selected = selected[:2]
        _rebuild_segments(selected)
        passes_applied.append("segments:3→2")
        if _measure() <= limit:
            return ctx, passes_applied

    # Pass 2: Trim KG activations 6→4
    if len(all_hits) > 4:
        all_hits = all_hits[:4]
        _rebuild_kg(all_hits)
        passes_applied.append("kg:6→4")
        if _measure() <= limit:
            return ctx, passes_applied

    # Pass 3: Trim recent exchanges 4→3
    if len(recent) > 3:
        recent = recent[-3:]
        _rebuild_history(compressed_tags, recent)
        passes_applied.append("recent:4→3")
        if _measure() <= limit:
            return ctx, passes_applied

    # Pass 4: Drop to 1 active segment
    if len(selected) > 1:
        selected = selected[:1]
        _rebuild_segments(selected)
        passes_applied.append("segments:2→1")
        if _measure() <= limit:
            return ctx, passes_applied

    # Pass 5: Trim KG activations 4→2
    if len(all_hits) > 2:
        all_hits = all_hits[:2]
        _rebuild_kg(all_hits)
        passes_applied.append("kg:4→2")
        if _measure() <= limit:
            return ctx, passes_applied

    # Pass 6: Trim recent exchanges 3→2
    if len(recent) > 2:
        recent = recent[-2:]
        _rebuild_history(compressed_tags, recent)
        passes_applied.append("recent:3→2")
        if _measure() <= limit:
            return ctx, passes_applied

    # Pass 7: Truncate active segment text to 200 chars
    _rebuild_segments(selected, max_chars=200)
    passes_applied.append("seg_text:truncated")
    if _measure() <= limit:
        return ctx, passes_applied

    # Pass 8: Truncate KG entries to 80 chars
    _rebuild_kg(all_hits, max_chars=80)
    passes_applied.append("kg_text:truncated")
    if _measure() <= limit:
        return ctx, passes_applied

    # Pass 9: Drop segment index entirely
    ctx["segment_index"] = "(see active passage below)"
    passes_applied.append("seg_index:dropped")
    if _measure() <= limit:
        return ctx, passes_applied

    # Pass 10: Trim recent to 1, truncate opponent statement
    recent = recent[-1:] if recent else []
    _rebuild_history([], recent, max_chars=150)
    ctx["last_stmt"] = ctx["last_stmt"][:150] + ("..." if len(ctx["last_stmt"]) > 150 else "")
    ctx["used_arguments"] = ""
    passes_applied.append("nuclear:minimal")

    return ctx, passes_applied


# ==============================================================================
#  CONTEXT ASSEMBLY
# ==============================================================================

def assemble_debate_context(
    kg,
    daf_ref,
    segments,
    exchanges,
    opponent_last,
    opponent_name,
    speaker_name=None,
    top_k_segments=3,
    top_k_kg=6,
    full_recent=4,
):
    """Assemble context with progressive compression to fit CONTEXT_LIMIT."""
    soul = kg.get_soul()
    if not soul:
        return None

    if not speaker_name:
        speaker_name = soul.content.get("name", "")

    # 1. Find actual opponent statement
    real_opponent_last = _find_opponent_last(exchanges, opponent_name)
    if not real_opponent_last:
        real_opponent_last = opponent_last or ""

    # 2. Select segments (citation-aware)
    selected = select_segments(
        segments, real_opponent_last, exchanges, top_k=top_k_segments
    )

    # 3. Compress exchange history
    for ex in exchanges:
        if "tag" not in ex:
            ex["tag"] = tag_exchange(ex)
    compressed_tags, recent = compress_exchanges(exchanges, full_recent=full_recent)

    # 4. Query KG
    kg_query = real_opponent_last
    if selected:
        kg_query += " " + " ".join(s.title for s in selected)
    hits = kg.query(kg_query, top_k=top_k_kg)

    # 5. Anti-repetition
    my_prior_tags = _collect_speaker_tags(exchanges, speaker_name)
    if my_prior_tags:
        used_args = ("Arguments you have already made (DO NOT repeat these — "
                     "advance the discussion): " + ", ".join(my_prior_tags[-8:]))
    else:
        used_args = ""

    # 5.5. Episodic memories (past exchanges, student encounters)
    ep_query = real_opponent_last
    if selected:
        ep_query += " " + " ".join(s.title for s in selected)
    episodic_ctx = get_episodic_context(kg, ep_query, top_k=3)

    # 6. Assemble full context
    ctx = {
        "name":           soul.content["name"],
        "essence":        soul.content["essence"],
        "values":         "; ".join(soul.content.get("values", [])),
        "voice":          soul.content.get("voice", ""),
        "daf_ref":        daf_ref,
        "segment_index":  _format_segment_index(segments),
        "daf_excerpt":    _format_active_segments(selected),
        "drives":         _format_drives(kg),
        "kg_context":     _format_kg_context(hits, kg),
        "episodic":       episodic_ctx,
        "recent":         _format_compressed_history(compressed_tags, recent),
        "opponent":       opponent_name or "your colleague",
        "last_stmt":      real_opponent_last[:400] or "(Begin the debate.)",
        "used_arguments": used_args,
        "_activated":     [h["node"].name for h in hits],
        "_segments_used": [s.index for s in selected],
        "_segment_ranges": [
            {"index": s.index, "title": s.title,
             "line_start": s.line_start, "line_end": s.line_end}
            for s in selected
        ],
    }

    # 7. Progressive compression if over budget
    limit = config.CONTEXT_LIMIT
    est = _estimate_context_tokens(ctx)

    if est > limit:
        ctx, passes = _compress_to_fit(
            ctx, kg, segments, selected, hits,
            compressed_tags, recent, exchanges,
            real_opponent_last, limit,
        )
        if passes:
            print(f"[CONTEXT] Compressed to fit {limit}t: {' → '.join(passes)} "
                  f"({est}t → {_estimate_context_tokens(ctx)}t)", flush=True)
    else:
        print(f"[CONTEXT] {speaker_name}: {est}t / {limit}t limit "
              f"(segs={ctx['_segments_used']}, kg={len(ctx['_activated'])})", flush=True)

    return ctx


def assemble_human_context(
    kg,
    daf_ref,
    segments,
    exchanges,
    human_text,
    human_name,
    top_k_segments=3,
    top_k_kg=6,
    full_recent=4,
):
    """Assemble context for human response with progressive compression."""
    soul = kg.get_soul()
    if not soul:
        return None

    selected = select_segments(
        segments, human_text, exchanges, top_k=top_k_segments
    )

    for ex in exchanges:
        if "tag" not in ex:
            ex["tag"] = tag_exchange(ex)
    compressed_tags, recent = compress_exchanges(exchanges, full_recent=full_recent)

    kg_query = human_text
    if selected:
        kg_query += " " + " ".join(s.title for s in selected)
    hits = kg.query(kg_query, top_k=top_k_kg)

    # Episodic: especially valuable for human interactions — recognize the student
    ep_query = human_text + " " + human_name
    if selected:
        ep_query += " " + " ".join(s.title for s in selected)
    episodic_ctx = get_episodic_context(kg, ep_query, top_k=3)

    ctx = {
        "name":           soul.content["name"],
        "essence":        soul.content["essence"],
        "values":         "; ".join(soul.content.get("values", [])),
        "voice":          soul.content.get("voice", ""),
        "daf_ref":        daf_ref,
        "segment_index":  _format_segment_index(segments),
        "daf_excerpt":    _format_active_segments(selected),
        "drives":         _format_drives(kg),
        "kg_context":     _format_kg_context(hits, kg),
        "episodic":       episodic_ctx,
        "recent":         _format_compressed_history(compressed_tags, recent),
        "human_name":     human_name,
        "human_stmt":     human_text,
        "used_arguments": "",
        "_activated":     [h["node"].name for h in hits],
        "_segments_used": [s.index for s in selected],
        "_segment_ranges": [
            {"index": s.index, "title": s.title,
             "line_start": s.line_start, "line_end": s.line_end}
            for s in selected
        ],
    }

    # Progressive compression
    limit = config.CONTEXT_LIMIT
    est = _estimate_context_tokens(ctx)

    if est > limit:
        ctx, passes = _compress_to_fit(
            ctx, kg, segments, selected, hits,
            compressed_tags, recent, exchanges,
            human_text, limit,
        )
        if passes:
            print(f"[CONTEXT] Human response compressed: {' → '.join(passes)} "
                  f"({est}t → {_estimate_context_tokens(ctx)}t)", flush=True)

    return ctx
