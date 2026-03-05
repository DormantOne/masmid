"""
context_manager.py — Context budget assembly for rabbi agents
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
  context_manager.py YOU ARE HERE — assemble_context()

DESIGN PHILOSOPHY:
  Small working memory (context window) + big hippocampus (KG + segment index).
  Push context close to the edge — optimize for a small model so the
  scaffolding scales up effortlessly to larger ones.

    ┌───────────────────────────────────────────────────┐
    │  CONTEXT WINDOW  (push toward model limit)        │
    │                                                   │
    │  [Segment Index]     all titles, 1 line each      │
    │  [Active Segments]   2-3 relevant full chunks     │
    │  [Compressed History] older = tags only            │
    │  [Recent Exchanges]  last 4 full text              │
    │  [KG Activations]    top-6 relevant nodes (rich)   │
    │  [Drives]            current intensity levels      │
    │  [Soul]              identity + values (always)    │
    │  [Anti-Repetition]   tags of THIS speaker's prior  │
    └───────────────────────────────────────────────────┘

EXPORTS:
  assemble_debate_context(...)  -> dict with all context strings
  assemble_human_context(...)   -> dict for responding to human
  select_segments(segments, query, exchanges, top_k) -> list[DafSegment]

DEPENDS ON:
  daf_ingest.DafSegment, compress_exchanges, tag_exchange
===============================================================================
"""

import re
from daf_ingest import DafSegment, compress_exchanges, tag_exchange


# ==============================================================================
#  SEGMENT SELECTION  (citation-aware)
# ==============================================================================

def _extract_cited_indices(text):
    """Find [0], [1], etc. references in exchange text."""
    return set(int(m) for m in re.findall(r"\[(\d+)\]", text))


def select_segments(segments, query, exchanges=None, top_k=3):
    """Select the most relevant daf segments for a query.

    Scoring combines:
      1. Keyword overlap with query (title + summary + en_text first 200 chars)
      2. Citation boost: segments referenced as [N] in recent exchanges
      3. Small recency bias toward earlier segments (opening context)

    No embedding calls — stays fast.
    """
    if not segments:
        return []
    if not query and not exchanges:
        result = [segments[0]]
        if len(segments) > 1:
            result.append(segments[-1])
        return result[:top_k]

    # Build query word set from the query
    query_words = set()
    if query:
        query_words = set(w.lower() for w in query.split() if len(w) > 2)

    # Count citation frequency across recent exchanges
    cite_counts = {}   # seg_index -> count
    if exchanges:
        for ex in exchanges[-8:]:
            for idx in _extract_cited_indices(ex.get("text", "")):
                cite_counts[idx] = cite_counts.get(idx, 0) + 1

    scored = []
    for seg in segments:
        # 1. Keyword overlap: title + summary + first 200 chars of en_text
        seg_text = (seg.title + " " + seg.summary + " " +
                    seg.en_text[:200]).lower()
        seg_words = set(w for w in seg_text.split() if len(w) > 2)
        overlap = len(query_words & seg_words) if query_words else 0

        # 2. Citation boost: heavily reward segments that rabbis already cited
        cite_boost = cite_counts.get(seg.index, 0) * 3.0

        # 3. Small position bias (prefer earlier for opening, but weak)
        position_bonus = 0.1 * (1.0 - seg.index / max(len(segments), 1))

        scored.append((overlap + cite_boost + position_bonus, seg))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [seg for _, seg in scored[:top_k]]


# ==============================================================================
#  FORMATTING HELPERS
# ==============================================================================

def _format_segment_index(segments):
    """One-line-per-segment overview. Cheap tokens, high orientation value."""
    if not segments:
        return "  (no segments)"
    return "\n".join(f"  {seg.header()}" for seg in segments)


def _format_active_segments(selected_segments):
    """Full English text of selected segments."""
    if not selected_segments:
        return "  (no segment selected)"
    parts = []
    for seg in selected_segments:
        parts.append(f"--- [{seg.index}] {seg.title} ---\n{seg.en_text}")
    return "\n\n".join(parts)


def _format_compressed_history(compressed_tags, recent_exchanges):
    """Compressed older history + full recent exchanges."""
    lines = []

    if compressed_tags:
        lines.append("Earlier: " + " | ".join(compressed_tags))

    for ex in recent_exchanges:
        speaker = ex.get("rabbi", "?")
        text = ex.get("text", "")[:300]   # slightly more generous
        is_human = ex.get("is_human", False)
        prefix = "Student" if is_human else speaker
        lines.append(f"  {prefix}: {text}")

    return "\n".join(lines) if lines else "  (Opening statement)"


def _format_kg_context(hits, kg):
    """Format KG query hits with drive discharge side-effects.

    Uses the improved Node.summary() which pulls from handles for richer text.
    """
    kg_lines = []
    for h in hits:
        n = h["node"]
        # summary() now includes handle descriptions — much richer
        kg_lines.append(
            f"  [{n.type}:{n.name}] {n.summary()[:180]}"
        )
        # Side-effect: discharge activated drives
        if n.type == "drive":
            kg.discharge_drive(n.id, 0.10)
        if n.type in ("habit", "skill"):
            for nb in kg.get_neighbors(n.id):
                if nb.type == "drive":
                    kg.discharge_drive(nb.id, 0.05)
    return "\n".join(kg_lines) if kg_lines else "  (no relevant nodes activated)"


def _format_drives(kg):
    """Current drive intensities as a compact string."""
    drives = kg.get_by_type("drive")
    if not drives:
        return "(no drives)"
    return " | ".join(
        f"{d.name}={d.content.get('intensity', 0):.2f}"
        for d in drives
    )


def _find_opponent_last(exchanges, opponent_name):
    """Find the last exchange actually said by the opponent.

    Prevents the prompt from attributing a human's or this speaker's
    own words to the opponent.
    """
    for ex in reversed(exchanges):
        if ex.get("rabbi", "") == opponent_name and not ex.get("is_human"):
            return ex.get("text", "")
    # Fallback: last exchange text regardless
    if exchanges:
        return exchanges[-1].get("text", "")
    return ""


def _collect_speaker_tags(exchanges, speaker_name):
    """Collect tags of arguments this speaker has already made.

    Used for anti-repetition: the model sees what it already said
    and is instructed to advance the discussion beyond these.
    """
    tags = []
    for ex in exchanges:
        if ex.get("rabbi", "") == speaker_name and not ex.get("is_human"):
            tag = ex.get("tag", "")
            if tag and tag != "remark":
                tags.append(tag)
    return tags


# ==============================================================================
#  CONTEXT ASSEMBLY
# ==============================================================================

def assemble_debate_context(
    kg,
    daf_ref,
    segments,
    exchanges,
    opponent_last,      # kept for backward compat, but we also search
    opponent_name,
    speaker_name=None,  # for anti-repetition
    top_k_segments=3,
    top_k_kg=6,
    full_recent=4,
):
    """Assemble the full context dict for a debate turn.

    Pushes context close to the model's limit for maximum depth.
    """
    soul = kg.get_soul()
    if not soul:
        return None

    # Speaker name from soul if not provided
    if not speaker_name:
        speaker_name = soul.content.get("name", "")

    # 1. Find the actual last opponent statement
    real_opponent_last = _find_opponent_last(exchanges, opponent_name)
    if not real_opponent_last:
        real_opponent_last = opponent_last or ""

    # 2. Select relevant daf segments (citation-aware)
    selected = select_segments(
        segments, real_opponent_last, exchanges, top_k=top_k_segments
    )

    # 3. Compress exchange history
    for ex in exchanges:
        if "tag" not in ex:
            ex["tag"] = tag_exchange(ex)
    compressed_tags, recent = compress_exchanges(exchanges, full_recent=full_recent)

    # 4. Query KG with combined signal
    kg_query = real_opponent_last
    if selected:
        kg_query += " " + " ".join(s.title for s in selected)
    hits = kg.query(kg_query, top_k=top_k_kg)

    # 5. Anti-repetition: what has this speaker already argued?
    my_prior_tags = _collect_speaker_tags(exchanges, speaker_name)
    if my_prior_tags:
        used_args = ("Arguments you have already made (DO NOT repeat these — "
                     "advance the discussion): " + ", ".join(my_prior_tags[-8:]))
    else:
        used_args = ""

    # 6. Assemble
    return {
        "name":           soul.content["name"],
        "essence":        soul.content["essence"],
        "values":         "; ".join(soul.content.get("values", [])),
        "voice":          soul.content.get("voice", ""),
        "daf_ref":        daf_ref,
        "segment_index":  _format_segment_index(segments),
        "daf_excerpt":    _format_active_segments(selected),
        "drives":         _format_drives(kg),
        "kg_context":     _format_kg_context(hits, kg),
        "recent":         _format_compressed_history(compressed_tags, recent),
        "opponent":       opponent_name or "your colleague",
        "last_stmt":      real_opponent_last[:400] or "(Begin the debate.)",
        "used_arguments": used_args,
        # Metadata for logging and UI highlighting
        "_activated":     [h["node"].name for h in hits],
        "_segments_used": [s.index for s in selected],
        "_segment_ranges": [
            {"index": s.index, "title": s.title,
             "line_start": s.line_start, "line_end": s.line_end}
            for s in selected
        ],
    }


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
    """Assemble context for responding to a human student."""
    soul = kg.get_soul()
    if not soul:
        return None

    # Select segments relevant to the human's question (citation-aware)
    selected = select_segments(
        segments, human_text, exchanges, top_k=top_k_segments
    )

    # Tag and compress
    for ex in exchanges:
        if "tag" not in ex:
            ex["tag"] = tag_exchange(ex)
    compressed_tags, recent = compress_exchanges(exchanges, full_recent=full_recent)

    # KG query
    kg_query = human_text
    if selected:
        kg_query += " " + " ".join(s.title for s in selected)
    hits = kg.query(kg_query, top_k=top_k_kg)

    return {
        "name":           soul.content["name"],
        "essence":        soul.content["essence"],
        "values":         "; ".join(soul.content.get("values", [])),
        "voice":          soul.content.get("voice", ""),
        "daf_ref":        daf_ref,
        "segment_index":  _format_segment_index(segments),
        "daf_excerpt":    _format_active_segments(selected),
        "drives":         _format_drives(kg),
        "kg_context":     _format_kg_context(hits, kg),
        "recent":         _format_compressed_history(compressed_tags, recent),
        "human_name":     human_name,
        "human_stmt":     human_text,
        "used_arguments": "",   # no anti-repetition needed for human responses
        "_activated":     [h["node"].name for h in hits],
        "_segments_used": [s.index for s in selected],
        "_segment_ranges": [
            {"index": s.index, "title": s.title,
             "line_start": s.line_start, "line_end": s.line_end}
            for s in selected
        ],
    }
