"""
templates.py — HTML template strings for login and main UI
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
  templates.py       YOU ARE HERE — LOGIN_HTML, MAIN_HTML strings

EXPORTS:
  LOGIN_HTML  — login/register page (has {{URL_PREFIX}} placeholder)
  MAIN_HTML   — main app UI (has {{URL_PREFIX}}, {{DISPLAY_NAME}},
                {{lens_keys}}, {{lens_map}} placeholders)

USED BY:
  app.py (render_login, render_main)
===============================================================================
"""

# These templates contain {{URL_PREFIX}} etc. which are replaced at render time
# by app.py.  The JS inside uses the BASE variable for all fetch() calls.

LOGIN_HTML = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Masmid - Enter the Beit Midrash</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{--bg:#0a0a10;--surface:#101018;--panel:#13131e;--border:#1d1d2e;
      --text:#ddd8ee;--muted:#5a5a7a;--accent:#7a4aff;--gold:#d4aa55;--err:#ff4455}
body{font-family:'Segoe UI',system-ui,sans-serif;background:var(--bg);color:var(--text);
     display:flex;align-items:center;justify-content:center;height:100vh}
.auth-box{background:var(--surface);border:1px solid var(--border);border-radius:16px;
          padding:40px;width:380px;max-width:92vw}
.auth-box h1{font-size:1.6rem;color:var(--gold);text-align:center;margin-bottom:6px}
.auth-box .subtitle{text-align:center;font-size:0.82rem;color:var(--muted);margin-bottom:28px}
.auth-box label{display:block;font-size:0.78rem;color:var(--muted);margin-bottom:4px;margin-top:14px}
.auth-box input[type="text"],.auth-box input[type="password"]{
  width:100%;padding:10px 14px;background:var(--panel);border:1px solid var(--border);
  border-radius:8px;color:var(--text);font-size:0.9rem;outline:none}
.auth-box input:focus{border-color:var(--accent)}
.auth-box .btn-row{display:flex;gap:10px;margin-top:24px}
.auth-box button{flex:1;padding:10px;border:1px solid var(--border);border-radius:8px;
  background:var(--panel);color:var(--text);font-size:0.85rem;cursor:pointer;transition:all .15s}
.auth-box button:hover{background:#1e1e30;border-color:#3a3a5a}
.auth-box button.primary{background:#1a1a3a;border-color:var(--accent);color:#c9b3ff}
.auth-box button.primary:hover{background:#2a2a5a}
.auth-box .msg{font-size:0.78rem;margin-top:12px;padding:8px;border-radius:6px;text-align:center}
.auth-box .msg.err{background:#1a0808;border:1px solid #3a1010;color:var(--err)}
.auth-box .msg.ok{background:#081a08;border:1px solid #103a10;color:#55cc55}
.auth-box .he-welcome{direction:rtl;text-align:center;font-family:'SBL Hebrew','Ezra SIL','Cardo',serif;
  font-size:1.1rem;color:#c8b870;margin-bottom:4px}
</style>
</head>
<body>
<div class="auth-box">
  <div class="he-welcome">&#x05d1;&#x05b5;&#x05d9;&#x05ea; &#x05d4;&#x05b7;&#x05de;&#x05bc;&#x05b4;&#x05d3;&#x05e8;&#x05b8;&#x05e9;&#x05c1;</div>
  <h1>Masmid</h1>
  <div class="subtitle">Enter the Study Hall</div>
  <div id="msg"></div>
  <label for="username">Username</label>
  <input type="text" id="username" autocomplete="username" autofocus>
  <label for="display_name" id="dn-label" style="display:none">Display Name (for the Beit Midrash)</label>
  <input type="text" id="display_name" style="display:none" placeholder="How the rabbis will address you">
  <label for="password">Password</label>
  <input type="password" id="password" autocomplete="current-password"
         onkeydown="if(event.key==='Enter') doLogin()">
  <div class="btn-row">
    <button onclick="doLogin()" class="primary">Enter</button>
    <button onclick="doRegister()">Register</button>
  </div>
</div>
<script>
const BASE = '{{URL_PREFIX}}';
function showMsg(text, ok) {
  const el = document.getElementById('msg');
  el.className = 'msg ' + (ok ? 'ok' : 'err');
  el.textContent = text;
}
function doLogin() {
  const u = document.getElementById('username').value.trim();
  const p = document.getElementById('password').value;
  if(!u||!p){showMsg('Please enter username and password.',false);return;}
  fetch(BASE+'/auth/login',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({username:u,password:p})}).then(r=>r.json()).then(d=>{
    if(d.ok) window.location.href=BASE+'/';
    else showMsg(d.error||'Login failed.',false);
  });
}
function doRegister() {
  const dnInput = document.getElementById('display_name');
  const dnLabel = document.getElementById('dn-label');
  if(dnInput.style.display==='none') {
    dnInput.style.display='block';
    dnLabel.style.display='block';
    showMsg('Choose a display name, then click Register again.',true);
    return;
  }
  const u = document.getElementById('username').value.trim();
  const p = document.getElementById('password').value;
  const dn = dnInput.value.trim();
  if(!u||!p){showMsg('Please enter username and password.',false);return;}
  fetch(BASE+'/auth/register',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({username:u,password:p,display_name:dn})}).then(r=>r.json()).then(d=>{
    if(d.ok){showMsg('Account created! Click Enter to proceed.',true);}
    else showMsg(d.error||'Registration failed.',false);
  });
}
</script>
</body>
</html>
'''

# The main HTML is stored in a separate file to keep this module manageable.
# It's loaded at import time from templates_main.html if it exists, otherwise
# from the inline fallback below.

import os as _os

_MAIN_HTML_FILE = _os.path.join(_os.path.dirname(__file__), "templates_main.html")

if _os.path.exists(_MAIN_HTML_FILE):
    with open(_MAIN_HTML_FILE, "r") as _f:
        MAIN_HTML = _f.read()
else:
    # Inline fallback (should not normally be reached in production)
    MAIN_HTML = "<html><body><h1>templates_main.html not found</h1></body></html>"
