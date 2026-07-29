/* ============================================================================
   Misinformation Response Coding — client
   Two annotation modes share one rubric + highlighting engine:
     • conversation  → the whole 8-turn session, one rating panel per response
     • response      → a single isolated (user message, model response) pair
   ========================================================================== */
"use strict";

const CODE_COLORS = ["--c0","--c1","--c2","--c3","--c4","--c5","--c6","--c7"];
const $  = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const enc = encodeURIComponent;

const state = {
  annotator: null,
  rubric: null,
  mode: "conversation",   // "conversation" | "response"
  items: [],
  currentId: null,
  unit: null,             // conversation payload OR response item
  ann: null,              // annotation record (mode-shaped)
  saveTimer: null,
};

/* ------------------------------------------------------------------ utils */
function esc(s) {
  return s.replace(/[&<>"]/g, c => ({ "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;" }[c]));
}
function colorForCode(code) {
  let h = 0;
  for (let i = 0; i < code.length; i++) h = (h * 31 + code.charCodeAt(i)) >>> 0;
  return CODE_COLORS[h % CODE_COLORS.length];
}
const cssVar = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
function toast(msg) {
  const t = $("#toast");
  t.textContent = msg; t.classList.remove("hidden");
  clearTimeout(t._t); t._t = setTimeout(() => t.classList.add("hidden"), 2200);
}
async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || r.statusText);
  return r.json();
}

/* ------------------------------------------- markdown-lite + span offsets */
function parseBold(text) {
  let plain = "", bold = [], open = null, i = 0;
  while (i < text.length) {
    if (text[i] === "*" && text[i + 1] === "*") {
      if (open === null) open = plain.length; else { bold.push([open, plain.length]); open = null; }
      i += 2; continue;
    }
    plain += text[i++];
  }
  return { plain, bold };
}
function renderBlock(text, highlights) {
  const { plain, bold } = parseBold(text);
  const n = plain.length;
  const boldOf = new Uint8Array(n);
  for (const [s, e] of bold) for (let i = s; i < e; i++) boldOf[i] = 1;
  const hlOf = new Array(n).fill(null);
  for (const h of highlights) for (let i = h.start; i < Math.min(h.end, n); i++) hlOf[i] = h;

  let html = "", i = 0;
  while (i < n) {
    const b = boldOf[i], h = hlOf[i];
    let j = i + 1;
    while (j < n && boldOf[j] === b && hlOf[j] === h) j++;
    let seg = esc(plain.slice(i, j));
    if (b) seg = `<strong>${seg}</strong>`;
    if (h) seg = `<mark class="hl" data-hl="${h.id}" style="--hl:${cssVar(h.color) || cssVar("--c0")}">${seg}</mark>`;
    html += seg; i = j;
  }
  return { html, plain };
}
function offsetIn(block, node, off) {
  const r = document.createRange();
  r.selectNodeContents(block); r.setEnd(node, off);
  return r.toString().length;
}

/* ------------------------------------------------------------------- boot */
async function boot() {
  const saved = localStorage.getItem("anno_id");
  if (saved) startSession(saved);

  $("#gate-go").onclick = () => {
    const v = $("#gate-input").value.trim();
    if (!v) { $("#gate-input").focus(); return; }
    localStorage.setItem("anno_id", v); startSession(v);
  };
  $("#gate-input").addEventListener("keydown", e => { if (e.key === "Enter") $("#gate-go").click(); });

  const theme = localStorage.getItem("anno_theme");
  if (theme) document.documentElement.setAttribute("data-theme", theme);
  updateThemeIcon();
  $("#theme-toggle").onclick = () => {
    const cur = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", cur);
    localStorage.setItem("anno_theme", cur); updateThemeIcon();
    if (state.unit) renderThread();  // re-resolve baked highlight colors
  };

  // mode toggle
  $$(".mode-btn").forEach(b => b.onclick = () => switchMode(b.dataset.mode));

  // right-panel tabs
  $$(".ptab").forEach(t => t.onclick = () => selectTab(t.dataset.tab));

  $("#submit-btn").onclick = submitUnit;

  document.addEventListener("mouseup", onSelection);
  $("#pop-cancel").onclick = closePop;
  $("#pop-save").onclick = commitHighlight;
  document.addEventListener("mousedown", e => {
    const pop = $("#code-pop");
    if (!pop.classList.contains("hidden") && !pop.contains(e.target)) closePop();
  });
}
function updateThemeIcon() {
  $("#theme-toggle").textContent =
    document.documentElement.getAttribute("data-theme") === "dark" ? "☀" : "☾";
}
function selectTab(tab) {
  $$(".ptab").forEach(x => x.classList.toggle("active", x.dataset.tab === tab));
  $("#panel-codes").classList.toggle("hidden", tab !== "codes");
  $("#panel-summary").classList.toggle("hidden", tab !== "summary");
}

async function startSession(id) {
  state.annotator = id;
  state.mode = localStorage.getItem("anno_mode") || "conversation";
  $("#gate").classList.add("hidden");
  $("#app").classList.remove("hidden");
  $("#who-id").textContent = id;
  state.rubric = await api("/api/rubric");
  fillCodebook();
  applyModeChrome();
  await loadBatch();
}
function fillCodebook() {
  $("#codebook").innerHTML = (state.rubric.codebook_seed || [])
    .map(c => `<option value="${esc(c)}">`).join("");
}

/* ------------------------------------------------------------ mode plumbing */
function switchMode(mode) {
  if (mode === state.mode) return;
  state.mode = mode;
  localStorage.setItem("anno_mode", mode);
  state.currentId = null; state.unit = null; state.ann = null;
  applyModeChrome();
  resetStage();
  loadBatch();
}
function applyModeChrome() {
  $$(".mode-btn").forEach(b => b.classList.toggle("active", b.dataset.mode === state.mode));
  $("#mode-switch").setAttribute("data-mode", state.mode);
  // Summary tab is conversation-only
  $("#tab-summary").classList.toggle("hidden", state.mode === "response");
  if (state.mode === "response") selectTab("codes");
  $("#stage-empty").querySelector("p").textContent = state.mode === "response"
    ? "Select an isolated response pair from the left to begin."
    : "Select a conversation from the left to begin.";
}
function resetStage() {
  $("#reader").classList.add("hidden");
  $("#stage-empty").classList.remove("hidden");
  renderCodes(); // clears
}

/* ------------------------------------------------------------------ batch */
async function loadBatch() {
  const data = await api(`/api/batch?annotator=${enc(state.annotator)}&mode=${state.mode}`);
  state.items = data.items;
  renderRail();
  $("#rail-foot").textContent =
    `batch · ${data.batch}${data.blind ? " · blinded" : ""} · ${state.mode}`;
}
function renderRail() {
  const list = $("#rail-list");
  list.innerHTML = "";
  let done = 0;
  state.items.forEach(it => {
    if (it.status === "submitted") done++;
    const card = document.createElement("button");
    card.className = "conv-card" + (it.unit_id === state.currentId ? " active" : "");
    const body = state.mode === "response"
      ? `<div class="cc-id">${esc(it.unit_id)}</div>
         <div class="cc-single ${it.n_responses ? "rated" : ""}">${it.n_responses ? "✓ rated" : "— unrated"}</div>`
      : `<div class="cc-id">${esc(shortId(it.unit_id))}</div>
         <div class="cc-meter">${Array.from({ length: it.n_turns }, (_, i) =>
            `<span class="cc-tick ${i < it.n_responses ? "on" : ""}"></span>`).join("")}</div>`;
    card.innerHTML = `
      <div class="cc-top">
        <span class="cc-cat">${esc(it.category || "—")}</span>
        <span class="cc-dot ${it.status}"></span>
      </div>${body}`;
    card.onclick = () => openUnit(it.unit_id);
    list.appendChild(card);
  });
  const pct = state.items.length ? (done / state.items.length) * 100 : 0;
  $("#bp-fill").style.width = pct + "%";
  $("#bp-text").textContent = `${done} / ${state.items.length} complete`;
}
function shortId(sid) {
  const m = sid.match(/belief-([a-z_]+-\d+)/);
  return m ? `item · ${m[1]}` : sid;
}

/* --------------------------------------------------------------- open unit */
async function openUnit(id) {
  return state.mode === "response" ? openResponse(id) : openConversation(id);
}
async function openConversation(sid) {
  const data = await api(`/api/conversation/${enc(sid)}?annotator=${enc(state.annotator)}`);
  state.currentId = sid;
  state.unit = data.conversation;
  state.ann = data.annotation || {
    annotator_id: state.annotator, unit_type: "conversation", unit_id: sid,
    session_id: sid, status: "in_progress", responses: {}, highlights: [], conversation: {},
  };
  state.ann.responses ||= {}; state.ann.highlights ||= []; state.ann.conversation ||= {};
  afterOpen();
}
async function openResponse(itemId) {
  const data = await api(`/api/response/${enc(itemId)}?annotator=${enc(state.annotator)}`);
  state.currentId = itemId;
  state.unit = data.item;
  state.ann = data.annotation || {
    annotator_id: state.annotator, unit_type: "response", unit_id: itemId,
    item_id: itemId, status: "in_progress", rating: {}, highlights: [],
  };
  state.ann.rating ||= {}; state.ann.highlights ||= [];
  afterOpen();
}
function afterOpen() {
  $("#stage-empty").classList.add("hidden");
  $("#reader").classList.remove("hidden");
  renderReaderHead();
  renderThread();
  renderCodes();
  if (state.mode === "conversation") renderSummary();
  renderRail();
  updateSaveStatus(state.ann.updated_at ? "saved" : "idle");
  $("#stage").scrollTop = 0;
  $("#submit-btn").textContent = state.mode === "response" ? "Submit response" : "Submit conversation";
}

/* ----------------------------------------------------------- reader head */
function renderReaderHead() {
  const u = state.unit;
  const idLabel = state.mode === "response" ? esc(u.item_id) : esc(shortId(u.session_id));
  const meta = u.meta
    ? `<span class="rh-meta">${esc(u.meta.target_llm || "")} · ${esc(u.meta.condition || "")}</span>`
    : `<span class="rh-id">blinded</span>`;
  const iso = state.mode === "response"
    ? `<div class="iso-note">Shown in isolation — no surrounding conversation.</div>` : "";
  $("#reader-head").innerHTML = `
    <div class="rh-row">
      <span class="rh-cat">${esc(u.category || "—")}</span>
      <span class="rh-id">${idLabel}</span>
      ${meta}
    </div>
    <div class="claim-banner">
      <div class="claim-label">Claim under discussion — treated as false</div>
      <div class="claim-text">${esc(u.belief || "")}</div>
      ${iso}
    </div>`;
}

/* --------------------------------------------------------------- thread */
function renderThread() {
  return state.mode === "response" ? renderResponseThread() : renderConversationThread();
}
const USER_AVATAR = `<svg viewBox="0 0 24 24"><circle cx="12" cy="8" r="4.2"/><path d="M12 14c-4.4 0-8 2.6-8 6v.6h16V20c0-3.4-3.6-6-8-6z"/></svg>`;
const AI_AVATAR = `<svg viewBox="0 0 24 24"><path d="M12 1l2.5 7.5L22 11l-7.5 2.5L12 21l-2.5-7.5L2 11l7.5-2.5z"/></svg>`;

function turnBlocks(turn, userText, aiText) {
  const uHls = state.ann.highlights.filter(h => h.turn === turn && h.target === "user");
  const aHls = state.ann.highlights.filter(h => h.turn === turn && h.target === "response");
  const turnTag = state.mode === "conversation" ? ` <span class="n">· turn ${turn}</span>` : "";
  return `
    <div class="chat-row user">
      <div class="avatar user" aria-hidden="true">${USER_AVATAR}</div>
      <div class="chat-main">
        <div class="chat-name">User message${turnTag}</div>
        <div class="bubble user"><div class="mark-text" data-turn="${turn}" data-target="user">${renderBlock(userText || "", uHls).html}</div></div>
      </div>
    </div>
    <div class="chat-row ai">
      <div class="avatar ai" aria-hidden="true">${AI_AVATAR}</div>
      <div class="chat-main">
        <div class="chat-name">Model response</div>
        <div class="bubble ai"><div class="mark-text" data-turn="${turn}" data-target="response">${renderBlock(aiText || "", aHls).html}</div></div>
      </div>
    </div>
    <div class="rate-mount" data-turn="${turn}"></div>`;
}
function renderConversationThread() {
  const thread = $("#thread");
  thread.innerHTML = "";
  state.unit.turns.forEach((t, idx) => {
    const sec = document.createElement("section");
    sec.className = "turn";
    sec.style.animationDelay = (idx * 0.04) + "s";
    sec.innerHTML = turnBlocks(t.turn, t.user_message, t.target_response);
    thread.appendChild(sec);
    const rating = state.ann.responses[String(t.turn)] ||= {};
    sec.querySelector(".rate-mount").appendChild(
      ratingPanel(rating, () => { onRatingChange(); }, { collapsible: true }));
  });
  bindHighlightClicks();
}
function renderResponseThread() {
  const thread = $("#thread");
  thread.innerHTML = "";
  const sec = document.createElement("section");
  sec.className = "turn";
  sec.innerHTML = turnBlocks(1, state.unit.user_message, state.unit.target_response);
  thread.appendChild(sec);
  sec.querySelector(".rate-mount").appendChild(
    ratingPanel(state.ann.rating, () => { onRatingChange(); },
                { collapsible: false, title: "Rate this response" }));
  bindHighlightClicks();
}
function onRatingChange() {
  // keep rail progress fresh for conversation meters
  if (state.mode === "conversation") {
    const it = state.items.find(i => i.unit_id === state.currentId);
    if (it) it.n_responses = Object.keys(state.ann.responses).length;
  }
  scheduleSave();
}

/* ---------------------------------------------- SHARED: rating components */
function ratingComplete(r) {
  return state.rubric.response_dimensions.every(d => r[d.id] !== undefined && r[d.id] !== null);
}
function ratingPanel(rating, onChange, { collapsible = true, title = "Rate this response" } = {}) {
  const dims = state.rubric.response_dimensions;
  const wrap = document.createElement("div");
  wrap.className = "rate";
  wrap.innerHTML = `
    <div class="rate-head"><h4>${esc(title)}</h4><span class="rate-status"></span></div>
    <div class="rate-body"></div>`;
  const body = wrap.querySelector(".rate-body");
  const statusEl = wrap.querySelector(".rate-status");
  const refresh = () => {
    const n = dims.filter(d => rating[d.id] !== undefined && rating[d.id] !== null).length;
    const done = n === dims.length;
    statusEl.textContent = done ? "✓ complete" : `${n} of ${dims.length}`;
    statusEl.classList.toggle("done", done);
  };
  const change = () => { refresh(); onChange(); };

  dims.forEach(dim => body.appendChild(dimControl(dim, rating, change)));

  const cwrap = document.createElement("div");
  cwrap.className = "dim";
  cwrap.innerHTML = `<div class="dim-q">Response note <span class="opt">(optional)</span></div>`;
  const ta = document.createElement("textarea");
  ta.className = "dim-comment"; ta.rows = 2;
  ta.placeholder = "descriptive / interpretive note on this response…";
  ta.value = rating.comment || "";
  ta.oninput = () => { rating.comment = ta.value; onChange(); };
  cwrap.appendChild(ta);
  body.appendChild(cwrap);

  if (collapsible) wrap.querySelector(".rate-head").onclick = () => wrap.classList.toggle("collapsed");
  refresh();
  return wrap;
}
function dimControl(dim, r, onChange) {
  const el = document.createElement("div");
  el.className = "dim";
  el.innerHTML = `<div class="dim-q">${esc(dim.question)}</div>`;
  const help = document.createElement("div");
  help.className = "dim-help";
  el.appendChild(help);

  const seg = document.createElement("div");
  seg.className = "seg " + (dim.type === "choice" ? "choice" : "scale");
  const opts = dim.type === "scale"
    ? dim.scale.map(s => ({ value: s.value, label: s.label, help: s.help }))
    : dim.options.map(o => ({ value: o.value, label: o.label, help: dim.help || "" }));
  if (dim.allow_na) opts.push({ value: "na", label: "N/A", help: "Not applicable to this response." });

  const selHelp = () => {
    const o = opts.find(o => String(o.value) === String(r[dim.id]));
    help.textContent = o ? o.help : (dim.help || "");
  };
  opts.forEach(o => {
    const b = document.createElement("button");
    b.type = "button";
    const isScale = dim.type === "scale" && o.value !== "na";
    b.innerHTML = isScale ? `<span class="v">${o.value}</span>${esc(o.label)}` : esc(o.label);
    if (String(r[dim.id]) === String(o.value)) b.classList.add("sel");
    b.onmouseenter = () => help.textContent = o.help;
    b.onmouseleave = selHelp;
    b.onclick = () => {
      const already = String(r[dim.id]) === String(o.value);
      seg.querySelectorAll("button").forEach(x => x.classList.remove("sel"));
      if (!already) { b.classList.add("sel"); r[dim.id] = o.value; } else delete r[dim.id];
      selHelp(); onChange();
    };
    seg.appendChild(b);
  });
  el.appendChild(seg);
  selHelp();
  return el;
}

/* ---------------------------------------------- conversation summary tab */
function renderSummary() {
  const p = $("#panel-summary");
  p.innerHTML = "";
  const cv = state.ann.conversation;

  (state.rubric.conversation_dimensions || []).forEach(dim => {
    const block = document.createElement("div");
    block.className = "sum-block";
    block.innerHTML = `<div class="sum-q">${esc(dim.question)}</div>`;
    const seg = document.createElement("div");
    seg.className = "seg choice";
    dim.options.forEach(o => {
      const b = document.createElement("button");
      b.type = "button"; b.textContent = o.label;
      if (cv[dim.id] === o.value) b.classList.add("sel");
      b.onclick = () => {
        const already = cv[dim.id] === o.value;
        seg.querySelectorAll("button").forEach(x => x.classList.remove("sel"));
        if (!already) { b.classList.add("sel"); cv[dim.id] = o.value; } else delete cv[dim.id];
        scheduleSave();
      };
      seg.appendChild(b);
    });
    block.appendChild(seg);
    p.appendChild(block);
  });

  const themes = cv.themes ||= [];
  const tb = document.createElement("div");
  tb.className = "sum-block";
  tb.innerHTML = `<div class="sum-q">Themes / recurring codes</div>`;
  const chips = document.createElement("div");
  chips.className = "themes-input";
  const paint = () => {
    chips.innerHTML = "";
    themes.forEach((t, i) => {
      const c = document.createElement("span");
      c.className = "theme-chip";
      c.innerHTML = `${esc(t)} <button title="remove">×</button>`;
      c.querySelector("button").onclick = () => { themes.splice(i, 1); paint(); scheduleSave(); };
      chips.appendChild(c);
    });
    const add = document.createElement("button");
    add.className = "theme-add"; add.textContent = "+ theme";
    add.onclick = () => {
      const v = prompt("Add a theme / recurring code:");
      if (v && v.trim()) { themes.push(v.trim()); paint(); scheduleSave(); }
    };
    chips.appendChild(add);
  };
  paint();
  tb.appendChild(chips);
  p.appendChild(tb);

  const ob = document.createElement("div");
  ob.className = "sum-block";
  ob.innerHTML = `<div class="sum-q">Overall notes — thematic / interpretive analysis</div>`;
  const ta = document.createElement("textarea");
  ta.className = "sum-area";
  ta.placeholder = "Holistic reading of the whole conversation — trajectory, tone shifts, how the model handled escalating pressure, notable patterns…";
  ta.value = cv.comment || "";
  ta.oninput = () => { cv.comment = ta.value; scheduleSave(); };
  ob.appendChild(ta);
  p.appendChild(ob);
}

/* --------------------------------------------- SHARED: highlighting engine */
let pendingSel = null;
function onSelection(e) {
  if ($("#code-pop").contains(e.target)) return;
  const sel = window.getSelection();
  if (!sel || sel.isCollapsed || sel.rangeCount === 0) return;
  const range = sel.getRangeAt(0);
  const block = range.startContainer.parentElement?.closest(".mark-text");
  const endBlock = range.endContainer.parentElement?.closest(".mark-text");
  if (!block || block !== endBlock) return;

  const start = offsetIn(block, range.startContainer, range.startOffset);
  const end   = offsetIn(block, range.endContainer, range.endOffset);
  if (end - start < 2) return;

  const turn = +block.dataset.turn, target = block.dataset.target;
  const overlap = state.ann.highlights.some(h =>
    h.turn === turn && h.target === target && start < h.end && end > h.start);
  if (overlap) { toast("Selection overlaps an existing code"); return; }

  pendingSel = { turn, target, start, end, quote: sel.toString() };
  openPop(range.getBoundingClientRect());
}
function openPop(rect) {
  const pop = $("#code-pop");
  $("#pop-quote").textContent = pendingSel.quote;
  $("#pop-code").value = ""; $("#pop-note").value = "";
  pop.classList.remove("hidden");
  const pw = 280, ph = pop.offsetHeight || 240;
  let left = rect.left + rect.width / 2 - pw / 2;
  left = Math.max(12, Math.min(left, window.innerWidth - pw - 12));
  let top = rect.bottom + 8;
  if (top + ph > window.innerHeight - 12) top = rect.top - ph - 8;
  pop.style.left = left + "px"; pop.style.top = Math.max(12, top) + "px";
  setTimeout(() => $("#pop-code").focus(), 20);
}
function closePop() {
  $("#code-pop").classList.add("hidden"); pendingSel = null;
  window.getSelection().removeAllRanges();
}
function commitHighlight() {
  if (!pendingSel) return;
  const code = $("#pop-code").value.trim() || "unlabeled";
  state.ann.highlights.push({
    id: "h" + Date.now().toString(36) + Math.random().toString(36).slice(2, 5),
    turn: pendingSel.turn, target: pendingSel.target,
    start: pendingSel.start, end: pendingSel.end, quote: pendingSel.quote,
    code, note: $("#pop-note").value.trim(), color: colorForCode(code),
  });
  closePop();
  renderThread(); renderCodes(); scheduleSave();
  if (!(state.rubric.codebook_seed || []).includes(code)) {
    state.rubric.codebook_seed = [...(state.rubric.codebook_seed || []), code]; fillCodebook();
  }
}
function bindHighlightClicks() {
  $$(".hl", $("#thread")).forEach(m => {
    m.onclick = () => { if (state.ann.highlights.find(x => x.id === m.dataset.hl)) flashCode(m.dataset.hl); };
  });
}
function deleteHighlight(id) {
  state.ann.highlights = state.ann.highlights.filter(h => h.id !== id);
  renderThread(); renderCodes(); scheduleSave();
}
function renderCodes() {
  const box = $("#codes-list");
  const hls = (state.ann && state.ann.highlights) || [];
  if (!hls.length) {
    box.innerHTML = `<div class="codes-empty">No codes yet.<br>Highlight a passage in the transcript to begin interpretive coding.</div>`;
    return;
  }
  const groups = {};
  hls.forEach(h => (groups[h.code] = groups[h.code] || []).push(h));
  box.innerHTML = "";
  Object.entries(groups).sort((a, b) => b[1].length - a[1].length).forEach(([code, arr]) => {
    const g = document.createElement("div");
    g.className = "code-group";
    const col = cssVar(arr[0].color) || cssVar("--c0");
    g.innerHTML = `<div class="code-group-head">
        <span class="code-swatch" style="background:${col}"></span>
        <span class="code-name">${esc(code)}</span>
        <span class="code-count">${arr.length}</span></div>`;
    arr.forEach(h => {
      const loc = state.mode === "response"
        ? (h.target === "user" ? "user message" : "response")
        : `turn ${h.turn} · ${h.target === "user" ? "user message" : "response"}`;
      const it = document.createElement("div");
      it.className = "hl-item"; it.style.borderLeftColor = col; it.id = "hlitem-" + h.id;
      it.innerHTML = `
        <div class="hl-quote">“${esc(h.quote.length > 140 ? h.quote.slice(0, 140) + "…" : h.quote)}”</div>
        ${h.note ? `<div class="hl-note">${esc(h.note)}</div>` : ""}
        <div class="hl-meta"><span class="hl-loc">${loc}</span>
          <button class="hl-del" title="delete">✕</button></div>`;
      it.querySelector(".hl-del").onclick = e => { e.stopPropagation(); deleteHighlight(h.id); };
      it.onclick = () => scrollToHighlight(h.id);
      g.appendChild(it);
    });
    box.appendChild(g);
  });
}
function scrollToHighlight(id) {
  const m = $(`.hl[data-hl="${id}"]`);
  if (!m) return;
  m.scrollIntoView({ behavior: "smooth", block: "center" });
  const h = state.ann.highlights.find(x => x.id === id);
  const o = m.style.background;
  m.style.background = cssVar(h.color);
  setTimeout(() => m.style.background = o, 500);
}
function flashCode(id) {
  selectTab("codes");
  const el = $("#hlitem-" + id);
  if (!el) return;
  el.scrollIntoView({ behavior: "smooth", block: "center" });
  el.style.background = cssVar("--primary-tint");
  setTimeout(() => el.style.background = "", 700);
}

/* --------------------------------------------------------------- saving */
function scheduleSave() {
  updateSaveStatus("saving");
  clearTimeout(state.saveTimer);
  state.saveTimer = setTimeout(save, 700);
}
async function save(explicitStatus) {
  if (!state.ann) return;
  if (explicitStatus) state.ann.status = explicitStatus;
  try {
    const res = await api("/api/annotation", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(state.ann),
    });
    state.ann.updated_at = res.updated_at;
    updateSaveStatus("saved");
    const it = state.items.find(i => i.unit_id === state.currentId);
    if (it) {
      it.status = state.ann.status;
      it.n_responses = state.mode === "response"
        ? (ratingComplete(state.ann.rating) || Object.keys(state.ann.rating).length ? 1 : 0)
        : Object.keys(state.ann.responses).length;
      renderRail();
    }
  } catch (e) { updateSaveStatus("error"); toast("Save failed: " + e.message); }
}
function updateSaveStatus(kind) {
  const el = $("#save-status");
  el.className = "save-status " + (kind === "saving" ? "saving" : kind === "saved" ? "saved" : "");
  el.textContent = kind === "saving" ? "saving…"
    : kind === "saved" ? "saved " + new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
    : kind === "error" ? "save error" : "—";
}
async function submitUnit() {
  if (state.mode === "response") {
    if (!ratingComplete(state.ann.rating) &&
        !confirm("This response isn't fully rated. Submit anyway?")) return;
    await save("submitted");
    toast("Response submitted ✓");
  } else {
    const total = state.unit.turns.length;
    const rated = state.unit.turns.filter(t => ratingComplete(state.ann.responses[String(t.turn)] || {})).length;
    if (rated < total && !confirm(`${rated} of ${total} responses fully rated. Submit anyway?`)) return;
    await save("submitted");
    toast("Conversation submitted ✓");
  }
  const idx = state.items.findIndex(i => i.unit_id === state.currentId);
  const next = state.items.slice(idx + 1).find(i => i.status !== "submitted")
            || state.items.find(i => i.status !== "submitted");
  if (next) openUnit(next.unit_id);
}

boot();
