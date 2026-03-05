"""
sefaria.py — Sefaria API client: fetch Daf Yomi + full text
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
  sefaria.py         YOU ARE HERE — fetch_daf_yomi(), fetch_daf_text()
  orchestrator.py    DebateOrchestrator
  log_system.py      LogSystem
  auth.py            UserStore
  prompts.py         Prompt templates
  app.py             Flask app factory + routes
  templates.py       HTML templates

EXPORTS:
  fetch_daf_yomi()     -> dict {ref, he_ref, url}
  fetch_daf_text(ref)  -> dict {ref, title, he_title, he, en, he_lines, en_lines, ...}

USED BY:
  orchestrator.py (DebateOrchestrator.load_daf)
===============================================================================
"""

import re
import requests


def fetch_daf_yomi():
    """Get today's Daf Yomi reference from Sefaria calendar API."""
    try:
        r = requests.get("https://www.sefaria.org/api/calendars", timeout=10)
        r.raise_for_status()
        for item in r.json().get("calendar_items", []):
            if item.get("title", {}).get("en", "") == "Daf Yomi":
                return {
                    "ref":    item["displayValue"]["en"],
                    "he_ref": item["displayValue"].get("he", ""),
                    "url":    item.get("url", ""),
                }
    except Exception as e:
        print(f"[SEFARIA calendar] {e}", flush=True)
    return {"ref": "Berakhot 2a", "he_ref": "\u05d1\u05e8\u05db\u05d5\u05ea \u05d1\u05f3", "url": ""}


def _strip_html(text):
    return re.sub(r"<[^>]+>", "", str(text)).strip()


def _flatten_text(v, sep="\n"):
    if isinstance(v, list):
        parts = [_flatten_text(x, sep) for x in v if x]
        return sep.join(p for p in parts if p.strip())
    return _strip_html(v)


def fetch_daf_text(ref):
    """Fetch full Daf text (Hebrew + English) from Sefaria."""
    try:
        safe = ref.replace(" ", "_").replace("'", "").replace(":", ".")
        url = f"https://www.sefaria.org/api/texts/{safe}?context=0&pad=0&multiple=1"
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        data = r.json()

        if isinstance(data, list):
            data = data[0] if data else {}

        he_raw = data.get("he", "")
        en_raw = data.get("text", "")

        he = _flatten_text(he_raw, "\n")
        en = _flatten_text(en_raw, "\n")

        he_lines = [l for l in he.split("\n") if l.strip()]
        en_lines = [l for l in en.split("\n") if l.strip()]

        return {
            "ref":      ref,
            "title":    data.get("ref", ref),
            "he_title": data.get("heRef", ""),
            "he":       he,
            "en":       en,
            "he_lines": he_lines,
            "en_lines": en_lines,
            "sections": data.get("sections", []),
            "length":   max(len(he_lines), len(en_lines)),
        }
    except Exception as e:
        print(f"[SEFARIA text] {e}", flush=True)
        return {
            "ref": ref, "title": ref, "he_title": "", "he": "",
            "en": f"[Sefaria error: {e}]",
            "he_lines": [], "en_lines": [], "sections": [], "length": 0,
        }
