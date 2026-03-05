"""
app.py — Flask application factory and all routes
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
  app.py             YOU ARE HERE — create_app() factory + all Flask routes
  templates.py       HTML templates

EXPORTS:
  create_app() -> Flask   (called by main.py)

DEPENDS ON:
  config.*
  llm.check_connection
  vector_store.VectorStore
  knowledge_graph.KnowledgeGraph
  rabbi_init.init_hillel, init_shammai
  agents.RabbiAgent, DreamAgent
  orchestrator.DebateOrchestrator
  log_system.LogSystem
  auth.UserStore
  templates.LOGIN_HTML, MAIN_HTML
===============================================================================
"""

import json
from functools import wraps

from flask import Flask, jsonify, render_template_string, request, session, redirect

import config
from llm import check_connection
from vector_store import VectorStore
from knowledge_graph import KnowledgeGraph
from rabbi_init import init_hillel, init_shammai
from agents import RabbiAgent, DreamAgent
from orchestrator import DebateOrchestrator
from log_system import LogSystem
from auth import UserStore
from templates import LOGIN_HTML, MAIN_HTML


def create_app():
    """Build and return the configured Flask app with all routes."""

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[STARTUP] Masmid v4 (modular) -- data={config.DATA_DIR.resolve()}", flush=True)
    print(f"[STARTUP] Backend={config.LLM_BACKEND}  Model={config.LLM_MODEL}  "
          f"Embed={config.EMBED_MODEL}", flush=True)
    print(f"[STARTUP] URL prefix: '{config.URL_PREFIX}'", flush=True)

    check_connection()

    # -- Build KGs -------------------------------------------------------------
    print("[STARTUP] Building Hillel KG...", flush=True)
    vs_h = VectorStore("hillel")
    kg_h = KnowledgeGraph("hillel", vs_h)
    init_hillel(kg_h)

    print("[STARTUP] Building Shammai KG...", flush=True)
    vs_s = VectorStore("shammai")
    kg_s = KnowledgeGraph("shammai", vs_s)
    init_shammai(kg_s)

    # Re-embed any missing vectors
    for label, kg_obj, vs_obj in [("Hillel", kg_h, vs_h), ("Shammai", kg_s, vs_s)]:
        vs_obj.purge_stale()
        missing = 0
        for n in kg_obj.nodes.values():
            for handle, text in n.handles.items():
                key = f"{n.id}:{handle}"
                if key not in vs_obj._store:
                    vs_obj.upsert(n.id, handle, text)
                    missing += 1
        if missing:
            print(f"[STARTUP] Re-embedded {missing} missing vectors for {label}",
                  flush=True)

    # -- Build agents / orchestrator / dreams ----------------------------------
    logger  = LogSystem()
    logger.new_session()
    hillel  = RabbiAgent(kg_h, logger)
    shammai = RabbiAgent(kg_s, logger)
    debate  = DebateOrchestrator(hillel, shammai, logger)
    dream_h = DreamAgent(kg_h, logger)
    dream_h.start()
    dream_s = DreamAgent(kg_s, logger)
    dream_s.start()
    users   = UserStore()

    print(f"[STARTUP] Ready -> http://localhost:{config.PORT}", flush=True)
    if config.URL_PREFIX:
        print(f"[STARTUP] Behind proxy: ...{config.URL_PREFIX}/", flush=True)

    # -- Flask app -------------------------------------------------------------
    app = Flask(__name__)
    app.secret_key = config.SECRET_KEY

    if config.URL_PREFIX:
        app.config["SESSION_COOKIE_PATH"] = config.URL_PREFIX + "/"

    lens_keys_js = json.dumps(list(config.LENSES.keys()))
    lens_map_js  = json.dumps(config.LENSES)

    # ── Use a Blueprint so all routes live under URL_PREFIX ────────────────
    # This means /masmid/auth/login, /masmid/api/step, etc. are real Flask
    # routes — works both locally and behind nginx (with or without stripping).
    from flask import Blueprint
    bp = Blueprint("masmid", __name__)

    # -- Auth decorators -------------------------------------------------------

    def require_auth(f):
        @wraps(f)
        def wrapped(*a, **kw):
            if "username" not in session:
                return redirect(f"{config.URL_PREFIX}/auth/login")
            return f(*a, **kw)
        return wrapped

    def require_auth_api(f):
        @wraps(f)
        def wrapped(*a, **kw):
            if "username" not in session:
                return jsonify({"error": "Not authenticated"}), 401
            return f(*a, **kw)
        return wrapped

    # -- Auth routes -----------------------------------------------------------

    @bp.route("/auth/login", methods=["GET"])
    def login_page():
        if "username" in session:
            return redirect(f"{config.URL_PREFIX}/")
        html = LOGIN_HTML.replace("{{URL_PREFIX}}", config.URL_PREFIX)
        return render_template_string(html)

    @bp.route("/auth/login", methods=["POST"])
    def login_submit():
        data = request.json or {}
        ok, msg = users.authenticate(data.get("username", ""), data.get("password", ""))
        if ok:
            session["username"]     = data["username"].strip().lower()
            session["display_name"] = msg
            logger.log("auth", "login", {"user": session["username"]})
            return jsonify({"ok": True})
        return jsonify({"ok": False, "error": msg})

    @bp.route("/auth/register", methods=["POST"])
    def register_submit():
        data = request.json or {}
        ok, msg = users.register(
            data.get("username", ""), data.get("password", ""),
            data.get("display_name", ""))
        if ok:
            logger.log("auth", "register", {"user": data.get("username", "")})
        return jsonify({"ok": ok, "error": "" if ok else msg})

    @bp.route("/auth/logout")
    def logout_route():
        session.clear()
        return redirect(f"{config.URL_PREFIX}/auth/login")

    # -- Main page -------------------------------------------------------------

    def _render_main():
        display_name = session.get("display_name", "Guest")
        return (MAIN_HTML
                .replace("{{lens_keys}}", lens_keys_js)
                .replace("{{lens_map}}",  lens_map_js)
                .replace("{{DISPLAY_NAME}}", display_name)
                .replace("{{URL_PREFIX}}", config.URL_PREFIX))

    @bp.route("/")
    @require_auth
    def index():
        return render_template_string(_render_main())

    @bp.route("/favicon.ico")
    def favicon():
        return "", 204

    # -- API routes ------------------------------------------------------------

    @bp.route("/api/health")
    def health():
        return jsonify({
            "status":        "ok",
            "backend":       config.LLM_BACKEND,
            "model":         config.LLM_MODEL,
            "embed":         config.EMBED_MODEL,
            "api_key":       "set" if config.LLM_API_KEY else "MISSING",
            "daf":           debate.daf["ref"] if debate.daf else None,
            "turn":          debate.turn,
            "exchanges":     len(debate.exchanges),
            "hillel_nodes":  len(kg_h.nodes),
            "shammai_nodes": len(kg_s.nodes),
            "hillel_embeds": len(vs_h._store),
            "shammai_embeds":len(vs_s._store),
        })

    @bp.route("/api/load_daf", methods=["POST"])
    @require_auth_api
    def api_load_daf():
        try:
            data = request.json or {}
            result = debate.load_daf(data.get("ref"))
            return jsonify(result)
        except Exception as e:
            print(f"[API ERROR /load_daf] {e}", flush=True)
            return jsonify({"error": str(e)}), 500

    @bp.route("/api/step", methods=["POST"])
    @require_auth_api
    def api_step():
        try:
            ex = debate.step()
            return jsonify({"exchange": ex, "turn": debate.turn})
        except Exception as e:
            print(f"[API ERROR /step] {e}", flush=True)
            return jsonify({"error": str(e), "exchange": None,
                            "turn": debate.turn}), 500

    @bp.route("/api/auto", methods=["POST"])
    @require_auth_api
    def api_auto():
        data = request.json or {}
        if data.get("enabled"):
            debate.start_auto()
        else:
            debate.stop_auto()
        return jsonify({"auto": debate._auto})

    @bp.route("/api/human_say", methods=["POST"])
    @require_auth_api
    def api_human_say():
        try:
            data = request.json or {}
            text = (data.get("text", "") or "").strip()
            if not text:
                return jsonify({"error": "Empty message."})
            display_name = session.get("display_name", "A Student")
            result = debate.human_say(text, display_name)
            return jsonify(result)
        except Exception as e:
            print(f"[API ERROR /human_say] {e}", flush=True)
            return jsonify({"error": str(e)}), 500

    @bp.route("/api/debate_status")
    @require_auth_api
    def api_debate_status():
        s = debate.status()
        s["exchanges"] = debate.exchanges
        s["segments"] = [
            {"index": seg.index, "title": seg.title, "summary": seg.summary,
             "line_start": seg.line_start, "line_end": seg.line_end}
            for seg in debate.segments
        ]
        return jsonify(s)

    @bp.route("/api/export_debate")
    @require_auth_api
    def api_export_debate():
        return jsonify({"text": debate.export()})

    @bp.route("/api/kg_summary")
    @require_auth_api
    def api_kg_summary():
        return jsonify({"hillel": kg_h.summary(), "shammai": kg_s.summary()})

    @bp.route("/api/kg_full/<who>")
    @require_auth_api
    def api_kg_full(who):
        kg = kg_h if who == "hillel" else kg_s
        return jsonify(kg.full_export())

    @bp.route("/api/kg_survey")
    @require_auth_api
    def api_kg_survey():
        lens = request.args.get("lens", "")
        query_text = config.LENSES.get(lens.lower(), lens)
        def hits_for(kg):
            raw = kg.query_with_lens(query_text, top_k=7)
            return [{
                "id": h["node"].id, "name": h["node"].name,
                "type": h["node"].type, "handle": h["handle"],
                "score": round(h["score"], 3),
                "summary": h["node"].summary()[:120],
            } for h in raw]
        return jsonify({
            "lens": lens, "hillel": hits_for(kg_h), "shammai": hits_for(kg_s),
        })

    @bp.route("/api/dream/<who>", methods=["POST"])
    @require_auth_api
    def api_trigger_dream(who):
        d = dream_h if who == "hillel" else dream_s
        return jsonify(d.cycle(manual=True))

    @bp.route("/api/dream_status")
    @require_auth_api
    def api_dream_status():
        return jsonify({"hillel": dream_h.status(), "shammai": dream_s.status()})

    @bp.route("/api/log")
    @require_auth_api
    def api_get_log():
        n = int(request.args.get("n", 80))
        return jsonify({"entries": logger.read_full(n=n)})

    @bp.route("/api/meta_log")
    @require_auth_api
    def api_get_meta_log():
        return jsonify({"content": logger.read_meta(chars=12000)})

    @bp.route("/api/daf_text")
    @require_auth_api
    def api_get_daf_text():
        return jsonify(debate.daf_text or {})

    @bp.route("/api/new_session", methods=["POST"])
    @require_auth_api
    def api_new_session():
        logger.new_session()
        debate.exchanges = []
        debate.turn = 0
        debate.stop_auto()
        return jsonify({"ok": True})

    # ── Register blueprint under the prefix ───────────────────────────────
    prefix = config.URL_PREFIX or ""
    app.register_blueprint(bp, url_prefix=prefix)

    # ── Bare-root redirect: / → /masmid/ when prefix is set ──────────────
    if prefix:
        @app.route("/")
        def root_redirect():
            return redirect(f"{prefix}/")

    return app
