#!/usr/bin/env python3
"""
main.py — Entry point for Masmid
===============================================================================
PROJECT MAP  (masmid/)
-----------------------------------------------
  main.py            YOU ARE HERE — CLI parsing, config apply, launch
  config.py          Config constants, CLI arg parsing
  llm.py             LLM abstraction (OpenAI / Ollama)
  models.py          Node, Edge, EdgeChannels
  vector_store.py    VectorStore
  knowledge_graph.py KnowledgeGraph
  rabbi_init.py      KG seed data (Hillel, Shammai)
  agents.py          RabbiAgent, DreamAgent
  sefaria.py         Sefaria API
  orchestrator.py    DebateOrchestrator
  log_system.py      LogSystem
  auth.py            UserStore
  prompts.py         Prompt templates
  app.py             Flask app factory + routes
  templates.py       HTML templates
  templates_main.html  Main UI HTML (loaded by templates.py)
  requirements.txt   Dependencies

USAGE:
  # OpenAI (default):
  export OPENAI_API_KEY=sk-...
  python main.py

  # Local Ollama with default model (gpt-oss:20b):
  python main.py --ollama

  # Local Ollama with specific model:
  python main.py --ollama llama3:8b

  # Override port, data dir, etc:
  python main.py --ollama --port 8080 --data-dir ./my_data

  # Custom Ollama URL + embedding model:
  python main.py --ollama mistral:7b --ollama-url http://192.168.1.5:11434 \
                 --ollama-embed-model mxbai-embed-large
===============================================================================
"""

from config import parse_cli_args, apply_cli_args
import config  # noqa: need the module reference for apply_cli_args


def main():
    # 1. Parse CLI args
    args = parse_cli_args()

    # 2. Apply to config module (must happen before any other import reads config)
    apply_cli_args(args)

    # 3. Now safe to import app (which imports everything else)
    from app import create_app

    # 4. Build and run
    app = create_app()
    app.run(debug=False, host="0.0.0.0", port=config.PORT, threaded=True)


if __name__ == "__main__":
    main()
