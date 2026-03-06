"""
episodic.py — Episodic memory layer for the KG
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
  context_manager.py Context budget assembly
  episodic.py        YOU ARE HERE — memory + encounter nodes

DESIGN:
  The character layer (soul, drives, habits, skills, goals) is static personality.
  The episodic layer adds what *happened*:

    ┌─ Character Layer ─────────────────────────────┐
    │  soul ── drives ── habits ── skills ── goals  │
    │                     │           │              │
    │              ┌──────┴───────────┘              │
    │              ▼                                 │
    │  ┌─ Episodic Layer ────────────────────────┐  │
    │  │  memory: "conceded watch-over for baked" │  │
    │  │  memory: "used reductio on boiled dough" │  │
    │  │  encounter: "Adam — asks deep questions" │  │
    │  └──────────────────────────────────────────┘  │
    └────────────────────────────────────────────────┘

  memory nodes:
    - Created every N exchanges (not every turn — too noisy)
    - Compressed 1-line summary + topic handle
    - Low initial salience (0.3), builds if re-activated
    - Edges to character nodes whose handles were activated

  encounter nodes:
    - One per human name, updated across sessions
    - Tracks: topics discussed, sessions count, last interaction
    - Edges to memory nodes from their conversations

EXPORTS:
  record_memory(kg, exchange, activated_nodes, daf_ref) -> Node or None
  record_encounter(kg, human_name, topic_tags, daf_ref) -> Node
  get_episodic_context(kg, query, top_k) -> str

DEPENDS ON:
  models.Node, models.EdgeChannels
  knowledge_graph.KnowledgeGraph
  daf_ingest.tag_exchange
===============================================================================
"""

import uuid
from datetime import datetime

from models import Node, EdgeChannels
from daf_ingest import tag_exchange


# ==============================================================================
#  MEMORY NODES
# ==============================================================================

# How often to create a memory node (every N exchanges per speaker)
MEMORY_INTERVAL = 3


def _make_memory_id():
    return f"mem_{uuid.uuid4().hex[:8]}"


def _make_encounter_id(name):
    """Deterministic ID from human name — so we find it again."""
    clean = name.strip().lower().replace(" ", "_")
    return f"enc_{clean}"


def _summarize_exchange(exchange):
    """Create a 1-line summary from an exchange. No LLM — extractive."""
    rabbi = exchange.get("rabbi", "?")
    tag = exchange.get("tag", "")
    text = exchange.get("text", "")

    # Take first sentence (up to first period, question mark, or 120 chars)
    first_sentence = text[:120]
    for punct in [".", "?", "!"]:
        idx = first_sentence.find(punct)
        if idx > 20:  # don't cut too short
            first_sentence = first_sentence[:idx + 1]
            break

    return f"{rabbi}: {first_sentence}"


def record_memory(kg, exchange, activated_nodes=None, daf_ref=""):
    """Create a memory node from a debate exchange.

    Only creates one every MEMORY_INTERVAL exchanges per speaker
    to avoid flooding the KG with noise.

    Args:
        kg: the rabbi's KnowledgeGraph
        exchange: the exchange dict
        activated_nodes: list of node names that were activated for this turn
        daf_ref: current daf reference

    Returns:
        The new memory Node, or None if skipped.
    """
    rabbi = exchange.get("rabbi", "")
    turn = exchange.get("turn", 0)
    is_human = exchange.get("is_human", False)

    # Count existing memories to decide if we should create one
    existing_memories = [n for n in kg.nodes.values() if n.type == "memory"]
    speaker_memories = [m for m in existing_memories
                        if m.content.get("rabbi") == rabbi]

    # Only record every MEMORY_INTERVAL turns per speaker (or always for humans)
    if not is_human and len(speaker_memories) > 0:
        last_turn = max(m.content.get("turn", 0) for m in speaker_memories)
        if turn - last_turn < MEMORY_INTERVAL:
            return None

    # Build the memory node
    tag = exchange.get("tag", "") or tag_exchange(exchange)
    summary = _summarize_exchange(exchange)

    content = {
        "rabbi": rabbi,
        "turn": turn,
        "tag": tag,
        "summary": summary,
        "daf_ref": daf_ref,
        "timestamp": exchange.get("ts", datetime.now().isoformat()),
        "is_human": is_human,
        "activated": activated_nodes or [],
    }

    # Handles for embedding — topic and summary give different retrieval angles
    handles = {
        "surface": summary[:150],
        "topic": f"{daf_ref} {tag} {' '.join(activated_nodes or [])}",
    }

    mem_node = Node(
        id=_make_memory_id(),
        type="memory",
        name=f"Turn {turn}: {tag}",
        content=content,
        handles=handles,
        salience=0.35,  # low initial salience — builds if re-accessed
        tags=["episodic", tag],
    )

    kg.add_node(mem_node)

    # Create edges to activated character nodes
    if activated_nodes:
        for act_name in activated_nodes[:3]:  # limit edge creation
            # Find the node by name
            target = next((n for n in kg.nodes.values()
                          if n.name == act_name and n.type != "memory"), None)
            if target:
                kg.add_edge(
                    mem_node.id, target.id,
                    EdgeChannels(
                        semantic=0.4,
                        temporal=0.6,
                        causal=0.2 if target.type in ("habit", "skill") else 0.0,
                        reinforcing=0.3,
                    ),
                )

    return mem_node


# ==============================================================================
#  ENCOUNTER NODES
# ==============================================================================

def record_encounter(kg, human_name, topic_tags=None, daf_ref="",
                     memory_node=None):
    """Create or update an encounter node for a human visitor.

    Called when a human enters the debate. The encounter node persists
    across sessions, accumulating topics and interaction count.

    Args:
        kg: the rabbi's KnowledgeGraph
        human_name: display name of the human
        topic_tags: list of topic tags from the conversation
        daf_ref: current daf reference
        memory_node: the memory node created for this interaction (for edge)

    Returns:
        The encounter Node (new or updated).
    """
    enc_id = _make_encounter_id(human_name)
    existing = kg.get_node(enc_id)

    now = datetime.now().isoformat()
    topic_tags = topic_tags or []

    if existing:
        # Update existing encounter
        c = existing.content
        c["sessions"] = c.get("sessions", 0) + 1
        c["last_seen"] = now
        c["last_daf"] = daf_ref

        # Accumulate topics (keep last 20 unique)
        old_topics = c.get("topics", [])
        all_topics = old_topics + topic_tags
        seen = set()
        unique = []
        for t in reversed(all_topics):
            if t not in seen:
                seen.add(t)
                unique.append(t)
        c["topics"] = list(reversed(unique))[:20]

        # Update handles for re-embedding
        topics_str = ", ".join(c["topics"][-8:])
        existing.handles["surface"] = (
            f"{human_name} — sessions: {c['sessions']}, "
            f"topics: {topics_str}, last: {daf_ref}"
        )
        existing.handles["topics"] = topics_str
        existing.salience = min(1.0, existing.salience + 0.05)
        existing.touch()

        # Re-embed with updated handles
        for handle, text in existing.handles.items():
            kg.vs.upsert(enc_id, handle, text)
        kg.save()

        enc_node = existing
    else:
        # Create new encounter
        topics_str = ", ".join(topic_tags[:8])
        content = {
            "name": human_name,
            "sessions": 1,
            "first_seen": now,
            "last_seen": now,
            "first_daf": daf_ref,
            "last_daf": daf_ref,
            "topics": topic_tags[:20],
        }
        handles = {
            "surface": f"{human_name} — first visit, discussing {daf_ref}, topics: {topics_str}",
            "topics": topics_str if topics_str else daf_ref,
        }
        enc_node = Node(
            id=enc_id,
            type="encounter",
            name=f"Student: {human_name}",
            content=content,
            handles=handles,
            salience=0.45,
            tags=["episodic", "encounter"],
        )
        kg.add_node(enc_node)

    # Edge from encounter to the memory node from this conversation
    if memory_node:
        kg.add_edge(
            enc_node.id, memory_node.id,
            EdgeChannels(
                semantic=0.5,
                temporal=0.8,
                emotional=0.3,
            ),
        )

    return enc_node


# ==============================================================================
#  EPISODIC CONTEXT FORMATTING
# ==============================================================================

def get_episodic_context(kg, query_text="", top_k=3):
    """Retrieve relevant episodic memories for inclusion in context.

    Returns a formatted string of memory summaries, cheap on tokens.
    Uses the same graph-walk-aware query as character nodes.
    """
    # Get all memory and encounter nodes
    episodic = [n for n in kg.nodes.values()
                if n.type in ("memory", "encounter")]

    if not episodic:
        return ""

    # Use the KG query to find relevant episodic nodes
    # (they participate in the same vector search + graph walk)
    hits = kg.query(query_text, top_k=top_k + 4) if query_text else []

    # Filter to just episodic hits
    ep_hits = [h for h in hits if h["node"].type in ("memory", "encounter")][:top_k]

    if not ep_hits:
        # Fallback: show most recent memories by salience
        recent = sorted(episodic, key=lambda n: n.salience, reverse=True)[:top_k]
        if not recent:
            return ""
        lines = []
        for n in recent:
            if n.type == "memory":
                lines.append(f"  [memory] {n.content.get('summary', '')[:100]}")
            elif n.type == "encounter":
                c = n.content
                lines.append(f"  [student:{c.get('name','')}] "
                            f"sessions={c.get('sessions',0)}, "
                            f"topics: {', '.join(c.get('topics',[])[-4:])}")
        return "\n".join(lines)

    lines = []
    for h in ep_hits:
        n = h["node"]
        if n.type == "memory":
            lines.append(f"  [memory] {n.content.get('summary', '')[:100]}")
        elif n.type == "encounter":
            c = n.content
            lines.append(f"  [student:{c.get('name','')}] "
                        f"sessions={c.get('sessions',0)}, "
                        f"topics: {', '.join(c.get('topics',[])[-4:])}")
    return "\n".join(lines)
