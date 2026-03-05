"""
daf_ingest.py — Daf segmentation + exchange compression
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
  daf_ingest.py      YOU ARE HERE — segment_daf(), tag_exchange()
  context_manager.py Context budget assembly

EXPORTS:
  DafSegment         — dataclass: index, title, summary, he_text, en_text
  segment_daf(daf_text) -> list[DafSegment]   (LLM-powered, called once per daf)
  tag_exchange(exchange) -> str                (fast extractive, called per turn)

DEPENDS ON:
  llm.llm_chat, llm.parse_json, llm.LLMError
===============================================================================
"""

import re
from dataclasses import dataclass, field

from llm import llm_chat, parse_json, LLMError


# ==============================================================================
#  DATA
# ==============================================================================

@dataclass
class DafSegment:
    """One semantic chunk of the Daf with its index metadata."""
    index: int              # 0-based segment number
    title: str              # 2-4 word title  (e.g. "Shema Timing Dispute")
    summary: str            # One sentence summary
    he_text: str            # Hebrew/Aramaic lines for this segment
    en_text: str            # English lines for this segment
    line_start: int = 0     # first line index in the original daf
    line_end: int = 0       # last line index (exclusive)

    def header(self) -> str:
        """Short header for index/overview display (cheap tokens)."""
        return f"[{self.index}] {self.title}: {self.summary}"

    def full_text(self, lang="en") -> str:
        """Full text for when this segment is selected for context."""
        if lang == "he":
            return self.he_text
        if lang == "both":
            return f"{self.he_text}\n---\n{self.en_text}"
        return self.en_text


# ==============================================================================
#  DAF SEGMENTATION
# ==============================================================================

SEGMENT_PROMPT = """\
You are a Talmud study assistant. Given the English translation of a Daf Yomi page,
break it into {target_segments} logical segments (sugyot or topic shifts).

For each segment, provide:
- title: 2-4 word title capturing the topic
- summary: One sentence (max 20 words) summarizing the content
- start_line: first line number (0-indexed)
- end_line: last line number (exclusive)

The text has {num_lines} lines total (0 to {max_line}).

Return ONLY valid JSON, no markdown:
{{
  "segments": [
    {{"title": "...", "summary": "...", "start_line": 0, "end_line": 5}},
    ...
  ]
}}
"""


def _chunk_lines(lines, target_segments=6):
    """Simple fallback: evenly divide lines into segments."""
    n = len(lines)
    if n == 0:
        return []
    chunk_size = max(1, n // target_segments)
    chunks = []
    for i in range(0, n, chunk_size):
        chunks.append((i, min(i + chunk_size, n)))
    # Merge last tiny chunk
    if len(chunks) > target_segments and chunks:
        last = chunks.pop()
        chunks[-1] = (chunks[-1][0], last[1])
    return chunks


def segment_daf(daf_text, target_segments=6) -> list:
    """Break a daf into titled, summarized segments.

    Uses LLM to find natural topic breaks. Falls back to even splitting
    if the LLM fails or the daf is short.

    Args:
        daf_text: dict from sefaria.fetch_daf_text()
        target_segments: approximate number of segments to create

    Returns:
        list of DafSegment
    """
    he_lines = daf_text.get("he_lines", [])
    en_lines = daf_text.get("en_lines", [])

    if not en_lines and not he_lines:
        return []

    # For very short daf pages, don't over-segment
    num_lines = max(len(he_lines), len(en_lines))
    if num_lines <= 4:
        target_segments = 1
    elif num_lines <= 8:
        target_segments = min(3, target_segments)

    # Try LLM segmentation
    en_block = "\n".join(f"[{i}] {line}" for i, line in enumerate(en_lines))

    try:
        system = SEGMENT_PROMPT.format(
            target_segments=target_segments,
            num_lines=len(en_lines),
            max_line=len(en_lines) - 1,
        )
        raw = llm_chat(
            messages=[{"role": "user", "content": en_block[:6000]}],
            system=system,
        )
        result = parse_json(raw)
        seg_specs = result.get("segments", [])

        if seg_specs and len(seg_specs) >= 1:
            return _build_segments(seg_specs, he_lines, en_lines)

    except (LLMError, Exception) as e:
        print(f"[DAF INGEST] LLM segmentation failed, using fallback: {e}",
              flush=True)

    # Fallback: even split with extractive titles
    return _fallback_segments(he_lines, en_lines, target_segments)


def _build_segments(seg_specs, he_lines, en_lines) -> list:
    """Build DafSegment list from LLM-produced specs."""
    segments = []
    for i, spec in enumerate(seg_specs):
        start = int(spec.get("start_line", 0))
        end = int(spec.get("end_line", start + 1))

        # Clamp to valid range
        start = max(0, min(start, len(en_lines)))
        end = max(start + 1, min(end, len(en_lines)))

        he_chunk = "\n".join(he_lines[start:end]) if start < len(he_lines) else ""
        en_chunk = "\n".join(en_lines[start:end]) if start < len(en_lines) else ""

        segments.append(DafSegment(
            index=i,
            title=spec.get("title", f"Section {i+1}")[:50],
            summary=spec.get("summary", "")[:120],
            he_text=he_chunk,
            en_text=en_chunk,
            line_start=start,
            line_end=end,
        ))
    return segments


def _fallback_segments(he_lines, en_lines, target) -> list:
    """Even-split fallback with extractive titles."""
    max_lines = max(len(he_lines), len(en_lines))
    chunks = _chunk_lines(list(range(max_lines)), target)
    segments = []

    for i, (start, end) in enumerate(chunks):
        he_chunk = "\n".join(he_lines[start:end]) if start < len(he_lines) else ""
        en_chunk = "\n".join(en_lines[start:end]) if start < len(en_lines) else ""

        # Extractive title: first few meaningful words
        title = _extract_title(en_chunk) if en_chunk else f"Section {i+1}"
        summary = en_chunk[:100].replace("\n", " ").strip()
        if len(summary) > 80:
            summary = summary[:77] + "..."

        segments.append(DafSegment(
            index=i,
            title=title,
            summary=summary,
            he_text=he_chunk,
            en_text=en_chunk,
            line_start=start,
            line_end=end,
        ))
    return segments


def _extract_title(text, max_words=4):
    """Pull a rough title from the first meaningful phrase."""
    # Strip common Talmud prefixes
    cleaned = re.sub(
        r"^(The Gemara|The Mishnah|We learned|It was taught|Come and hear|"
        r"The Sages taught|Rabbi \w+ said|Rav \w+ said)\s*[:,.]?\s*",
        "", text.strip(), flags=re.IGNORECASE,
    )
    words = cleaned.split()[:max_words]
    title = " ".join(words).rstrip(".,;:")
    return title[:50] if title else "Passage"


# ==============================================================================
#  EXCHANGE COMPRESSION
# ==============================================================================

# Common Talmudic stopwords to skip when extracting tags
_STOPWORDS = {
    "the", "a", "an", "is", "was", "are", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "this", "that",
    "these", "those", "i", "you", "he", "she", "it", "we", "they",
    "what", "which", "who", "whom", "where", "when", "how", "why",
    "not", "no", "nor", "but", "and", "or", "if", "then", "than",
    "so", "as", "of", "in", "on", "at", "to", "for", "with", "by",
    "from", "about", "into", "through", "during", "before", "after",
    "above", "below", "between", "under", "again", "further", "once",
    "here", "there", "all", "each", "every", "both", "few", "more",
    "most", "other", "some", "such", "only", "own", "same", "very",
    "just", "now", "also", "well", "even", "still", "already", "yet",
    "too", "quite", "rather", "much", "many", "us", "our", "your",
    "his", "her", "its", "their", "my", "me", "him",
    "said", "says", "say", "indeed", "thus", "therefore", "however",
    "yet", "one", "like", "see", "let",
}


def tag_exchange(exchange) -> str:
    """Generate a 1-3 word tag for an exchange. Fast, no LLM call.

    Examples:
        "When we consider the timing of Shema..." → "shema-timing"
        "The boundary between sacred and profane..." → "sacred-profane"
    """
    text = exchange.get("text", "")
    if not text:
        return "?"

    # Strip bracketed stage directions
    text = re.sub(r"\[.*?\]", "", text)

    # Extract meaningful words
    words = re.findall(r"[A-Za-z\u0590-\u05FF]{3,}", text.lower())
    keywords = [w for w in words if w not in _STOPWORDS]

    if not keywords:
        return "remark"

    # Take first 2 unique meaningful words
    seen = set()
    tags = []
    for w in keywords:
        if w not in seen:
            seen.add(w)
            tags.append(w)
            if len(tags) >= 2:
                break

    return "-".join(tags) if tags else "remark"


def compress_exchanges(exchanges, full_recent=3):
    """Compress a list of exchanges for context display.

    Returns two parts:
        compressed: list of one-line summaries for older exchanges
        recent:     list of full exchange dicts for the last N

    This gives the model a "scroll-back" view of the whole conversation
    in very few tokens, plus full detail for the active thread.
    """
    if len(exchanges) <= full_recent:
        return [], exchanges

    older = exchanges[:-full_recent]
    recent = exchanges[-full_recent:]

    compressed = []
    for ex in older:
        tag = ex.get("tag") or tag_exchange(ex)
        speaker = ex.get("rabbi", "?")
        # Ultra-compressed: "Hillel:shema-timing" or "Student:why-boundary"
        short_name = speaker.split()[-1] if " " in speaker else speaker
        compressed.append(f"{short_name}:{tag}")

    return compressed, recent
