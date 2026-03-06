#!/usr/bin/env python3
"""
consolidate_kgs.py — One-time cleanup to collapse duplicate KG nodes.

Run this ONCE before going live to fix the existing bloated KGs.
After this, the dream agent's pre-cycle consolidation will keep them lean.

Usage:
    # With OpenAI (default):
    export OPENAI_API_KEY=sk-...
    python consolidate_kgs.py

    # With local Ollama:
    python consolidate_kgs.py --ollama

    # With Ollama + custom model/URL:
    python consolidate_kgs.py --ollama --ollama-url http://localhost:11434 \
                              --ollama-embed-model nomic-embed-text

    # Custom data dir:
    python consolidate_kgs.py --ollama --data-dir ./masmid_data

    # Skip re-embedding entirely (fastest, just structural cleanup):
    python consolidate_kgs.py --skip-embed
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Consolidate Masmid KGs")
    parser.add_argument("--data-dir", default="./masmid_data",
                        help="Path to masmid_data directory")
    parser.add_argument("--ollama", action="store_true",
                        help="Use local Ollama for embeddings")
    parser.add_argument("--ollama-url", default="http://localhost:11434",
                        help="Ollama server URL (default: http://localhost:11434)")
    parser.add_argument("--ollama-embed-model", default="nomic-embed-text",
                        help="Ollama embedding model (default: nomic-embed-text)")
    parser.add_argument("--skip-embed", action="store_true",
                        help="Skip re-embedding during merge (fastest, structural only)")
    args = parser.parse_args()

    # ── Configure config module BEFORE importing anything else ──
    import config
    config.DATA_DIR = Path(args.data_dir)

    if args.ollama:
        config.LLM_BACKEND  = "ollama"
        config.LLM_BASE_URL = args.ollama_url.rstrip("/") + "/v1"
        config.LLM_API_KEY  = "ollama"
        config.EMBED_MODEL  = args.ollama_embed_model
        config.LLM_MODEL    = "unused-for-consolidation"
        print(f"Using Ollama embeddings: {config.EMBED_MODEL} "
              f"at {config.LLM_BASE_URL}")
    elif args.skip_embed:
        print("Skipping re-embedding (structural cleanup only)")
    else:
        if not config.LLM_API_KEY:
            print("WARNING: No OPENAI_API_KEY set and --ollama not specified.")
            print("Re-embedding will fail. Use --ollama or --skip-embed.\n")

    # Now safe to import modules that read config
    from vector_store import VectorStore
    from knowledge_graph import KnowledgeGraph, NODE_TYPE_LIMITS

    # ── If --skip-embed, monkey-patch VectorStore.upsert to be a no-op ──
    if args.skip_embed:
        _original_upsert = VectorStore.upsert
        VectorStore.upsert = lambda self, *a, **kw: None
        print("(VectorStore.upsert disabled — no embedding calls will be made)\n")

    print(f"Data dir: {config.DATA_DIR.resolve()}")
    print(f"Node limits: {NODE_TYPE_LIMITS}\n")

    for name in ["hillel", "shammai"]:
        kg_file = config.DATA_DIR / f"kg_{name}.json"
        if not kg_file.exists():
            print(f"  {name}: kg file not found, skipping")
            continue

        vs = VectorStore(name)
        kg = KnowledgeGraph(name, vs)

        before_nodes = len(kg.nodes)
        before_edges = len(kg.edges)

        # Show before state
        from collections import Counter
        before_counts = Counter(n.type for n in kg.nodes.values())
        print(f"  {name.upper()} BEFORE: {before_nodes} nodes, {before_edges} edges")
        for t in ["soul", "goal", "habit", "skill", "drive",
                   "memory", "encounter", "definition", "edge"]:
            c = before_counts.get(t, 0)
            if c > 0:
                limit = NODE_TYPE_LIMITS.get(t)
                lim_str = f"/{limit}" if limit else ""
                print(f"    {t}: {c}{lim_str}")

        # Consolidate
        merged = kg.consolidate()

        after_nodes = len(kg.nodes)
        after_edges = len(kg.edges)
        after_counts = Counter(n.type for n in kg.nodes.values())

        print(f"\n  {name.upper()} AFTER:  {after_nodes} nodes, {after_edges} edges")
        for t in ["soul", "goal", "habit", "skill", "drive",
                   "memory", "encounter", "definition", "edge"]:
            c = after_counts.get(t, 0)
            if c > 0:
                limit = NODE_TYPE_LIMITS.get(t)
                lim_str = f"/{limit}" if limit else ""
                print(f"    {t}: {c}{lim_str}")

        removed = before_nodes - after_nodes
        print(f"\n  {name}: {merged} merged, {removed} total removed "
              f"({before_nodes} → {after_nodes})")

        # Show surviving nodes
        print(f"\n  Surviving character nodes for {name}:")
        for n in sorted(kg.nodes.values(),
                        key=lambda x: (-{"soul":0,"goal":1,"drive":2,
                                         "habit":3,"skill":4}.get(x.type, 9),
                                        -x.salience)):
            if n.type in ("memory", "encounter"):
                continue
            print(f"    [{n.type:6s}] {n.name:45s} sal={n.salience:.3f}")

        print("\n" + "="*60 + "\n")

    print("Done. The KGs have been saved with consolidated nodes.")
    print("You can now start the server normally.")


if __name__ == "__main__":
    main()