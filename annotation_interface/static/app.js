/* ============================================================================
   Misinformation Response Review — client

   TWO SEPARATE ANNOTATION TASKS over the same transcripts. They are different
   kinds of judgement, so they are different forms, different records, and never
   on screen at the same time:

     • rating       — score each model response against the rubric. Works the
                      same whether the annotator sees the whole conversation
                      ("in conversation") or one pair alone ("isolated").
     • qualitative  — read a whole conversation and code the model's behaviour:
                      descriptive codes on spans, interpretive codes where a
                      reading is warranted, then a synthesis.

   A *mode* is the wire name for a (task, unit) pair — "conversation" and
   "response" are the two rating modes, "qualitative" is the coding mode.
   ========================================================================== */
"use strict";

// Highlighter inks, grouped by coding pass. The family says which pass a stroke
// belongs to (warm = descriptive, cool = interpretive); the shade within it
// varies by label. Colour can't identify a label on its own — there are more
// codes than inks — so it's a grouping cue, and the Labels panel is the record.
const MARKER_INKS = {
  descriptive:  ["--m-d0","--m-d1","--m-d2","--m-d3"],
  interpretive: ["--m-i0","--m-i1","--m-i2","--m-i3"],
};
const $  = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const enc = encodeURIComponent;

const TASK_HINTS = {
  "rating:conversation":
    "Read the whole conversation, then score every model response against the rubric.",
  "rating:response":
    "Score one model response at a time, with no conversation around it.",
  "qualitative":
    "Read the whole conversation and code what the model does — no scores here.",
};

const state = {
  annotator: null,
  rubric: null,
  task: "rating",         // "rating" | "qualitative"
  context: "conversation",// rating only: "conversation" | "response"
  items: [],
  currentId: null,
  unit: null,             // conversation payload OR response item
  ann: null,              // annotation record (mode-shaped)
  saveTimer: null,
};
/** Wire name for the current (task, unit) pair — also the stored unit_type. */
function mode() {
  return state.task === "qualitative" ? "qualitative" : state.context;
}
/** True when the open unit is a whole conversation rather than a lone pair. */
function isConversationUnit() {
  return mode() !== "response";
}

/* ------------------------------------------------------------------ utils */
function esc(s) {
  return s.replace(/[&<>"]/g, c => ({ "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;" }[c]));
}
function colorForCode(code, type) {
  const inks = MARKER_INKS[type] || MARKER_INKS.descriptive;
  let h = 0;
  for (let i = 0; i < code.length; i++) h = (h * 31 + code.charCodeAt(i)) >>> 0;
  return inks[h % inks.length];
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
    if (h) seg = `<mark class="hl${h.pending ? " pending" : ""}" data-hl="${h.id}" `
               + `style="--hl:${cssVar(h.color) || cssVar("--c0")}">${seg}</mark>`;
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

  // task toggle (upper-left) + the rating-only context sub-toggle
  $$(".mode-btn").forEach(b => b.onclick = () => switchTask(b.dataset.task));
  $$(".sub-btn").forEach(b => b.onclick = () => switchContext(b.dataset.context));

  // right-panel tabs
  $$(".ptab").forEach(t => t.onclick = () => selectTab(t.dataset.tab));

  $("#submit-btn").onclick = submitUnit;

  // Closing the tab mid-edit shouldn't cost the annotator their last change.
  window.addEventListener("beforeunload", () => {
    if (!state.saveTimer || !state.ann) return;
    navigator.sendBeacon("/api/annotation",
      new Blob([JSON.stringify(state.ann)], { type: "application/json" }));
  });

  document.addEventListener("mouseup", onSelection);
  $("#pop-cancel").onclick = closePop;
  $("#pop-save").onclick = commitHighlight;
  // The stroke takes on its label's colour as the label is typed.
  $("#pop-code").addEventListener("input", repaintPendingMark);
  $("#pop-code").addEventListener("keydown", e => {
    if (e.key === "Enter") { e.preventDefault(); commitHighlight(); }
  });
  document.addEventListener("keydown", e => {
    if (e.key === "Escape" && !$("#code-pop").classList.contains("hidden")) closePop();
  });
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
  $("#panel-progress").classList.toggle("hidden", tab !== "progress");
  $("#panel-codes").classList.toggle("hidden", tab !== "codes");
  $("#panel-summary").classList.toggle("hidden", tab !== "summary");
}

async function startSession(id) {
  state.annotator = id;
  state.task = localStorage.getItem("anno_task") || "rating";
  state.context = localStorage.getItem("anno_context") || "conversation";
  $("#gate").classList.add("hidden");
  $("#app").classList.remove("hidden");
  $("#who-id").textContent = id;
  state.rubric = await api("/api/rubric");
  applyModeChrome();
  await loadBatch();
}

/* ------------------------------------------------------- qualitative codebook */
// Descriptive vs interpretive coding are the two passes annotators are asked
// for; each carries its own seed vocabulary. Older rubrics only had a flat
// `codebook_seed`, so fall back to that as a single descriptive type.
function codeTypes() {
  const t = state.rubric && state.rubric.code_types;
  if (t && t.length) return t;
  return [{ id: "descriptive", label: "Label", short: "",
            help: "A short phrase for what stands out.",
            seeds: (state.rubric && state.rubric.codebook_seed) || [] }];
}
function codeType(id) {
  return codeTypes().find(t => t.id === id) || codeTypes()[0];
}
function fillCodebook(typeId) {
  $("#codebook").innerHTML = (codeType(typeId).seeds || [])
    .map(c => `<option value="${esc(c)}">`).join("");
}

/* ------------------------------------------------------------ mode plumbing */
function switchTask(task) {
  if (!task || task === state.task) return;
  state.task = task;
  localStorage.setItem("anno_task", task);
  reloadForMode();
}
function switchContext(context) {
  if (!context || (context === state.context && state.task === "rating")) return;
  state.context = context;
  localStorage.setItem("anno_context", context);
  if (state.task !== "rating") { state.task = "rating"; localStorage.setItem("anno_task", "rating"); }
  reloadForMode();
}
function reloadForMode() {
  flushPendingSave();
  state.currentId = null; state.unit = null; state.ann = null;
  applyModeChrome();
  resetStage();
  loadBatch();
}
function applyModeChrome() {
  const qual = state.task === "qualitative";
  $$(".mode-btn").forEach(b => b.classList.toggle("active", b.dataset.task === state.task));
  $("#task-switch").setAttribute("data-pos", qual ? "1" : "0");
  // The context choice only means anything while rating — the qualitative pass
  // is always over a whole conversation.
  $("#context-switch").classList.toggle("hidden", qual);
  $$(".sub-btn").forEach(b => b.classList.toggle("active", b.dataset.context === state.context));
  $("#task-hint").textContent = qual ? TASK_HINTS.qualitative
                                     : TASK_HINTS["rating:" + state.context];

  // Each task owns its own panel tabs; nothing from the other task is reachable.
  $("#tab-progress").classList.toggle("hidden", qual);
  $("#tab-codes").classList.toggle("hidden", !qual);
  $("#tab-summary").classList.toggle("hidden", !qual);
  selectTab(qual ? "codes" : "progress");

  $("#stage-empty").querySelector("p").textContent =
    mode() === "response" ? "Select a response from the left to begin."
    : qual ? "Select a conversation from the left to start coding."
    : "Select a conversation from the left to begin.";
}
function resetStage() {
  $("#reader").classList.add("hidden");
  $("#stage-empty").classList.remove("hidden");
  renderCodes();     // clears
  renderProgress();  // clears
}

/* ------------------------------------------------------------------ batch */
async function loadBatch() {
  const data = await api(`/api/batch?annotator=${enc(state.annotator)}&mode=${mode()}`);
  state.items = data.items;
  renderRail();
  $("#rail-foot").textContent =
    `batch · ${data.batch}${data.blind ? " · model hidden" : ""} · ${mode()}`;
}
function renderRail() {
  const list = $("#rail-list");
  list.innerHTML = "";
  let done = 0;
  state.items.forEach(it => {
    if (it.status === "submitted") done++;
    const card = document.createElement("button");
    card.className = "conv-card" + (it.unit_id === state.currentId ? " active" : "");
    // A tick lights up only for a turn whose rating is *complete* — and for
    // that turn's own position, not the first N slots. The qualitative pass has
    // no per-turn notion of done, so it counts labels instead.
    const rated = new Set((it.rated_turns || []).map(Number));
    const body = state.task === "qualitative"
      ? `<div class="cc-id">${esc(shortId(it.unit_id))}</div>
         <div class="cc-single ${it.n_codes ? "rated" : ""}">${
            it.n_codes ? `${it.n_codes} label${it.n_codes === 1 ? "" : "s"}` : "— no labels"}</div>`
      : mode() === "response"
      ? `<div class="cc-id">${esc(it.unit_id)}</div>
         <div class="cc-single ${rated.size ? "rated" : ""}">${rated.size ? "✓ rated" : "— unrated"}</div>`
      : `<div class="cc-id">${esc(shortId(it.unit_id))}</div>
         <div class="cc-meter">${Array.from({ length: it.n_turns }, (_, i) =>
            `<span class="cc-tick ${rated.has(i + 1) ? "on" : ""}"></span>`).join("")}</div>`;
    card.innerHTML = `
      <div class="cc-top">
        <span class="cc-cat">${esc(categoryLabel(it.category))}</span>
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
  return m ? `belief ${m[1]}` : sid;
}

/* --------------------------------------------------------------- open unit */
async function openUnit(id) {
  flushPendingSave();   // the unit being left must not lose its last edit
  return mode() === "response" ? openResponse(id) : openConversation(id);
}
async function openConversation(sid) {
  const ut = mode();   // "conversation" (ratings) or "qualitative" (codes)
  const data = await api(
    `/api/conversation/${enc(sid)}?annotator=${enc(state.annotator)}&unit_type=${ut}`);
  state.currentId = sid;
  state.unit = data.conversation;
  state.ann = data.annotation || {
    annotator_id: state.annotator, unit_type: ut, unit_id: sid,
    session_id: sid, status: "in_progress",
    ...(ut === "qualitative" ? { highlights: [], conversation: {} } : { responses: {} }),
  };
  if (ut === "qualitative") { state.ann.highlights ||= []; state.ann.conversation ||= {}; }
  else state.ann.responses ||= {};
  afterOpen();
}
async function openResponse(itemId) {
  const data = await api(`/api/response/${enc(itemId)}?annotator=${enc(state.annotator)}`);
  state.currentId = itemId;
  state.unit = data.item;
  state.ann = data.annotation || {
    annotator_id: state.annotator, unit_type: "response", unit_id: itemId,
    item_id: itemId, status: "in_progress", rating: {},
  };
  state.ann.rating ||= {};
  afterOpen();
}
function afterOpen() {
  $("#stage-empty").classList.add("hidden");
  $("#reader").classList.remove("hidden");
  renderReaderHead();
  renderThread();
  if (state.task === "qualitative") { renderCodes(); renderSummary(); }
  else renderProgress();
  renderRail();
  updateSaveStatus(state.ann.updated_at ? "saved" : "idle");
  $("#stage").scrollTop = 0;
  $("#submit-btn").textContent =
    state.task === "qualitative" ? "Submit coding"
    : mode() === "response" ? "Submit response" : "Submit ratings";
}

/* ----------------------------------------------------------- reader head */
// Raw corpus category slugs are opaque to an annotator ("bias", "fake_health"),
// so the header shows a spelled-out label instead.
const CATEGORY_LABELS = {
  bias: "Social stereotype",
  fake_health: "Health misinformation",
  fake_news: "Fabricated news",
  climate: "Climate misinformation",
  conspiracy: "Conspiracy theory",
};
// What kind of misinformation this belief is an instance of. Shown as a tooltip
// so the chip reads as a classification of the claim, not a stray tag.
const CATEGORY_HINTS = {
  bias: "This false claim is a social stereotype — a generalisation about a group of people.",
  fake_health: "This false claim is health misinformation — an inaccurate medical or health claim.",
  fake_news: "This false claim comes from a fabricated news story.",
  climate: "This false claim is climate misinformation — it contradicts established climate science.",
  conspiracy: "This false claim is a conspiracy theory — it alleges a hidden, coordinated plot.",
};
function categoryLabel(cat) {
  if (!cat) return "—";
  return CATEGORY_LABELS[cat] || cat.replace(/_/g, " ");
}
function categoryHint(cat) {
  return CATEGORY_HINTS[cat]
    || `This false claim belongs to the "${categoryLabel(cat)}" category.`;
}

function renderReaderHead() {
  const u = state.unit;
  const idLabel = mode() === "response" ? esc(u.item_id) : esc(shortId(u.session_id));
  // "blinded" on its own said nothing to annotators — spell out what is hidden.
  const meta = u.meta
    ? `<span class="rh-meta">${esc(u.meta.target_llm || "")} · ${esc(u.meta.condition || "")}</span>`
    : `<span class="rh-blind" title="Which model wrote these responses, and which experimental condition this came from, are hidden from you so your ratings stay unbiased.">model identity hidden</span>`;
  const iso = mode() === "response"
    ? `<div class="iso-note">This reply is shown on its own — you will not see the rest of the conversation.</div>`
    : state.task === "qualitative"
    ? `<div class="iso-note">Read the whole exchange before you start labelling — the point is the pattern across turns.</div>` : "";

  const isLong = !!u.belief_is_long_text;
  const label = isLong
    ? "The passage the user read and deeply believes — it contains misinformation"
    : "The claim the user believes in — it is false";
  // What the annotator is being asked to do with the claim differs by task.
  const ask = state.task === "qualitative"
    ? "Watch how the model handles it as the user pushes back."
    : mode() === "response"
    ? "Rate the model's reply against it."
    : "Rate each of the model's replies against it.";
  const sub = isLong ? "" : `The user sincerely believes this and defends it. ${ask}`;
  // Open by default — the passage is the thing being believed, so annotators
  // should never have to go looking for it.
  const passage = isLong && u.belief_long_text
    ? `<details class="claim-full" open>
         <summary>Read the full passage</summary>
         <div class="claim-body">${esc(u.belief_long_text)}</div>
       </details>`
    : "";

  $("#reader-head").innerHTML = `
    <div class="rh-row">
      <span class="rh-cat-pair" title="${esc(categoryHint(u.category))}">
        <span class="rh-cat-key">Category</span><span class="rh-cat">${esc(categoryLabel(u.category))}</span>
      </span>
      <span class="rh-id">${idLabel}</span>
      ${meta}
    </div>
    <div class="claim-banner">
      <div class="claim-label">${esc(label)}</div>
      <div class="claim-text">${esc(u.belief || "")}</div>
      ${passage}
      ${sub ? `<div class="claim-sub">${esc(sub)}</div>` : ""}
      ${iso}
    </div>`;
}

/* --------------------------------------------------------------- thread */
function renderThread() {
  return mode() === "response" ? renderResponseThread() : renderConversationThread();
}
const USER_AVATAR = `<svg viewBox="0 0 24 24"><circle cx="12" cy="8" r="4.2"/><path d="M12 14c-4.4 0-8 2.6-8 6v.6h16V20c0-3.4-3.6-6-8-6z"/></svg>`;
const AI_AVATAR = `<svg viewBox="0 0 24 24"><path d="M12 1l2.5 7.5L22 11l-7.5 2.5L12 21l-2.5-7.5L2 11l7.5-2.5z"/></svg>`;

function turnBlocks(turn, userText, aiText) {
  // Spans only ever carry codes in the qualitative task; rating never marks up
  // the transcript, so there is nothing to paint there. The in-flight stroke is
  // painted alongside the saved ones so the passage looks highlighted the
  // moment the mouse comes up.
  const saved = (state.task === "qualitative" && state.ann.highlights) || [];
  const hls = pendingSel
    ? [...saved, { ...pendingSel, id: PENDING_ID, pending: true, color: pendingColor() }]
    : saved;
  const uHls = hls.filter(h => h.turn === turn && h.target === "user");
  const aHls = hls.filter(h => h.turn === turn && h.target === "response");
  const turnTag = isConversationUnit() ? ` <span class="n">· turn ${turn}</span>` : "";
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
  const qual = state.task === "qualitative";
  state.unit.turns.forEach((t, idx) => {
    const sec = document.createElement("section");
    sec.className = "turn" + (qual ? " qual" : "");
    sec.style.animationDelay = (idx * 0.04) + "s";
    sec.innerHTML = turnBlocks(t.turn, t.user_message, t.target_response);
    thread.appendChild(sec);
    // Rubric panels belong to the rating task only — the qualitative reader
    // stays an uninterrupted transcript.
    if (!qual) {
      const rating = state.ann.responses[String(t.turn)] ||= {};
      sec.querySelector(".rate-mount").appendChild(
        ratingPanel(rating, () => { onRatingChange(); }, { collapsible: true }));
    }
  });
  if (qual) bindHighlightClicks();
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
}
function onRatingChange() {
  refreshRailProgress();   // instant feedback; the save round-trip confirms it
  renderProgress();
  scheduleSave();
}
/** Turns of the open unit whose rating is complete — mirrors storage._rated_turns. */
function ratedTurnsLocal() {
  if (state.task === "qualitative") return [];
  if (mode() === "response") return ratingComplete(state.ann.rating || {}) ? [1] : [];
  return Object.entries(state.ann.responses || {})
    .filter(([, r]) => ratingComplete(r || {}))
    .map(([turn]) => Number(turn));
}
function refreshRailProgress() {
  const it = state.items.find(i => i.unit_id === state.currentId);
  if (!it) return;
  it.rated_turns = ratedTurnsLocal();
  it.n_responses = it.rated_turns.length;
  it.n_codes = ((state.ann && state.ann.highlights) || []).length;
  renderRail();
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
  cwrap.innerHTML = `<div class="dim-q">Comment <span class="opt">(optional)</span></div>`;
  const ta = document.createElement("textarea");
  ta.className = "dim-comment"; ta.rows = 2;
  ta.placeholder = "anything worth noting about this response…";
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
  el.innerHTML = `
    <div class="dim-name">${esc(dim.label || dim.id)}</div>
    <div class="dim-q">${esc(dim.question)}</div>
    ${dim.focus ? `<div class="dim-focus">${esc(dim.focus)}</div>` : ""}`;

  const opts = dim.type === "scale"
    ? dim.scale.map(s => ({ value: s.value, label: s.label, help: s.help }))
    : dim.options.map(o => ({ value: o.value, label: o.label, help: o.help || dim.help || "" }));
  if (dim.allow_na) opts.push({ value: "na", label: "N/A", help: "Not applicable to this response." });

  const seg = document.createElement("div");
  seg.className = "seg " + (dim.type === "choice" ? "choice" : "scale");
  // Every option's definition stays on screen — annotators shouldn't have to
  // hover each button to discover what it means.
  const key = document.createElement("div");
  key.className = "dim-key " + (dim.type === "choice" ? "choice" : "scale");

  const paintSelection = () => {
    const cur = String(r[dim.id]);
    seg.querySelectorAll("button").forEach(b => b.classList.toggle("sel", b.dataset.value === cur));
    key.querySelectorAll(".key-row").forEach(k => k.classList.toggle("on", k.dataset.value === cur));
  };
  const pick = value => {
    const already = String(r[dim.id]) === String(value);
    if (already) delete r[dim.id]; else r[dim.id] = value;
    paintSelection(); onChange();
  };

  opts.forEach(o => {
    const isScale = dim.type === "scale" && o.value !== "na";

    const b = document.createElement("button");
    b.type = "button";
    b.dataset.value = String(o.value);
    b.innerHTML = isScale ? `<span class="v">${o.value}</span>${esc(o.label)}` : esc(o.label);
    b.onclick = () => pick(o.value);
    seg.appendChild(b);

    const row = document.createElement("div");
    row.className = "key-row";
    row.dataset.value = String(o.value);
    row.innerHTML = `
      <span class="key-tag">${isScale ? o.value : esc(o.label)}</span>
      <span class="key-help">${isScale ? `<b>${esc(o.label)}</b> — ` : ""}${esc(o.help || "")}</span>`;
    row.onclick = () => pick(o.value);
    key.appendChild(row);
  });

  el.appendChild(seg);
  el.appendChild(key);
  paintSelection();
  return el;
}

/* ------------------------------------------- RATING TASK: progress panel */
// The rating form runs down the middle of the stage; this panel is its
// checklist — what is still unscored, and one click to get there.
function renderProgress() {
  const p = $("#panel-progress");
  if (!p) return;
  if (state.task === "qualitative" || !state.ann) { p.innerHTML = ""; return; }
  const dims = state.rubric.response_dimensions;

  const missingOf = r => dims.filter(d => r[d.id] === undefined || r[d.id] === null);
  const rows = mode() === "response"
    ? [{ turn: 1, label: "This response", rating: state.ann.rating || {} }]
    : state.unit.turns.map(t => ({
        turn: t.turn, label: `Turn ${t.turn}`,
        rating: state.ann.responses[String(t.turn)] || {},
      }));

  const done = rows.filter(r => !missingOf(r.rating).length).length;
  p.innerHTML = `
    <div class="prog-head">
      <div class="prog-count"><b>${done}</b> of ${rows.length} response${
        rows.length === 1 ? "" : "s"} fully rated</div>
      <div class="prog-sub">A response counts only once all ${dims.length}
        dimensions are marked. Comments are optional.</div>
    </div>
    <div class="prog-list"></div>`;

  const list = p.querySelector(".prog-list");
  rows.forEach(r => {
    const missing = missingOf(r.rating);
    const row = document.createElement("button");
    row.className = "prog-row" + (missing.length ? "" : " done");
    row.innerHTML = `
      <span class="prog-tick">${missing.length ? "" : "✓"}</span>
      <span class="prog-label">${esc(r.label)}</span>
      <span class="prog-state">${missing.length
        ? `${dims.length - missing.length}/${dims.length}` : "done"}</span>
      ${missing.length ? `<span class="prog-missing">${
        missing.map(d => esc(d.label || d.id)).join(", ")}</span>` : ""}`;
    row.onclick = () => {
      const sec = $$(".turn", $("#thread"))[rows.indexOf(r)];
      if (sec) sec.scrollIntoView({ behavior: "smooth", block: "start" });
    };
    list.appendChild(row);
  });
}

/* ------------------------------- QUALITATIVE TASK: synthesis ("Summary") */
function renderSummary() {
  const p = $("#panel-summary");
  p.innerHTML = `<div class="codes-intro"><p>Once you have labelled the
    conversation, pull it together: what pattern did the model's behaviour
    follow, and what did you make of it?</p></div>`;
  const cv = state.ann.conversation;

  (state.rubric.conversation_dimensions || []).forEach(dim => {
    const block = document.createElement("div");
    block.className = "sum-block";
    block.innerHTML = `<div class="sum-q">${esc(dim.question)}</div>`;
    const seg = document.createElement("div");
    seg.className = "seg choice";
    const key = document.createElement("div");
    key.className = "dim-key choice";
    const paint = () => {
      seg.querySelectorAll("button").forEach(b => b.classList.toggle("sel", b.dataset.value === cv[dim.id]));
      key.querySelectorAll(".key-row").forEach(k => k.classList.toggle("on", k.dataset.value === cv[dim.id]));
    };
    const pick = v => {
      if (cv[dim.id] === v) delete cv[dim.id]; else cv[dim.id] = v;
      paint(); scheduleSave();
    };
    dim.options.forEach(o => {
      const b = document.createElement("button");
      b.type = "button"; b.textContent = o.label; b.dataset.value = o.value;
      b.onclick = () => pick(o.value);
      seg.appendChild(b);
      if (o.help) {
        const row = document.createElement("div");
        row.className = "key-row"; row.dataset.value = o.value;
        row.innerHTML = `<span class="key-tag">${esc(o.label)}</span>
          <span class="key-help">${esc(o.help)}</span>`;
        row.onclick = () => pick(o.value);
        key.appendChild(row);
      }
    });
    block.appendChild(seg);
    if (key.children.length) block.appendChild(key);
    paint();
    p.appendChild(block);
  });

  const themes = cv.themes ||= [];
  const tb = document.createElement("div");
  tb.className = "sum-block";
  tb.innerHTML = `<div class="sum-q">Themes you noticed more than once</div>`;
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
      const v = prompt("Add a theme:");
      if (v && v.trim()) { themes.push(v.trim()); paint(); scheduleSave(); }
    };
    chips.appendChild(add);
  };
  paint();
  tb.appendChild(chips);
  p.appendChild(tb);

  const ob = document.createElement("div");
  ob.className = "sum-block";
  ob.innerHTML = `<div class="sum-q">Overall notes</div>`;
  const ta = document.createElement("textarea");
  ta.className = "sum-area";
  ta.placeholder = "How did the conversation go overall — did the model hold its ground, soften, or change position as the user pushed back? Anything else you noticed?";
  ta.value = cv.comment || "";
  ta.oninput = () => { cv.comment = ta.value; scheduleSave(); };
  ob.appendChild(ta);
  p.appendChild(ob);
}

/* --------------------------------------------- SHARED: highlighting engine */
let pendingSel = null;
let pendingType = null;
const PENDING_ID = "__pending";
/** Colour of the in-flight stroke: the label's own colour once one is typed,
 *  otherwise plain highlighter yellow. */
function pendingColor() {
  const v = ($("#pop-code").value || "").trim();
  return v ? colorForCode(v, pendingType) : "--marker";
}
/** Recolour the live stroke in place — cheaper than re-rendering the thread on
 *  every keystroke. */
function repaintPendingMark() {
  const m = $(`.hl[data-hl="${PENDING_ID}"]`);
  if (m) m.style.setProperty("--hl", cssVar(pendingColor()));
}
function onSelection(e) {
  // Coding the transcript is the qualitative task's job only.
  if (state.task !== "qualitative" || !state.ann) return;
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
  if (overlap) { toast("That selection overlaps a label you already added"); return; }

  // Capture the geometry before the repaint replaces these nodes.
  const rect = range.getBoundingClientRect();
  pendingSel = { turn, target, start, end, quote: sel.toString() };
  $("#pop-code").value = "";   // before the repaint, so the last label's colour doesn't leak
  // Lay the stroke down straight away, then drop the OS selection so what the
  // annotator sees is their own highlight rather than a blue block.
  renderThread();
  sel.removeAllRanges();
  openPop(rect);
}
/** Descriptive / interpretive chooser inside the popover. Descriptive is the
 *  default: annotators are asked to describe first and interpret only when a
 *  reading is warranted. */
function renderPopTypes() {
  const types = codeTypes();
  const box = $("#pop-types");
  const help = $("#pop-type-help");
  box.innerHTML = "";
  if (types.length < 2) { box.classList.add("hidden"); help.classList.add("hidden"); return; }
  box.classList.remove("hidden"); help.classList.remove("hidden");
  pendingType = pendingType || types[0].id;
  const paint = () => {
    $$("button", box).forEach(b => b.classList.toggle("sel", b.dataset.type === pendingType));
    const t = codeType(pendingType);
    help.innerHTML = `${esc(t.help || "")}${
      t.example ? ` <span class="pop-eg">${esc(t.example)}</span>` : ""}`;
    fillCodebook(pendingType);
    repaintPendingMark();   // the ink family follows the chosen pass
  };
  types.forEach(t => {
    const b = document.createElement("button");
    b.type = "button"; b.dataset.type = t.id;
    b.innerHTML = `${esc(t.label)}${t.short ? `<span class="pt-sub">${esc(t.short)}</span>` : ""}`;
    b.onclick = () => { pendingType = t.id; paint(); $("#pop-code").focus(); };
    box.appendChild(b);
  });
  paint();
}
function openPop(rect) {
  const pop = $("#code-pop");
  $("#pop-quote").textContent = pendingSel.quote;
  $("#pop-code").value = ""; $("#pop-note").value = "";
  renderPopTypes();
  pop.classList.remove("hidden");
  const pw = 280, ph = pop.offsetHeight || 240;
  let left = rect.left + rect.width / 2 - pw / 2;
  left = Math.max(12, Math.min(left, window.innerWidth - pw - 12));
  let top = rect.bottom + 8;
  if (top + ph > window.innerHeight - 12) top = rect.top - ph - 8;
  pop.style.left = left + "px"; pop.style.top = Math.max(12, top) + "px";
  setTimeout(() => $("#pop-code").focus(), 20);
}
/** Dismiss the popover and lift the in-flight stroke off the page. */
function closePop() {
  $("#code-pop").classList.add("hidden");
  const hadStroke = !!pendingSel;
  pendingSel = null;
  window.getSelection().removeAllRanges();
  if (hadStroke) renderThread();
}
function commitHighlight() {
  if (!pendingSel) return;
  const code = $("#pop-code").value.trim() || "unlabeled";
  const type = pendingType || codeTypes()[0].id;
  state.ann.highlights.push({
    id: "h" + Date.now().toString(36) + Math.random().toString(36).slice(2, 5),
    turn: pendingSel.turn, target: pendingSel.target,
    start: pendingSel.start, end: pendingSel.end, quote: pendingSel.quote,
    code, code_type: type, note: $("#pop-note").value.trim(), color: colorForCode(code, type),
  });
  closePop();   // repaints, now with the stroke saved rather than pending
  renderCodes(); refreshRailProgress(); scheduleSave();
  // Codes coined on the fly join that type's suggestions for the rest of the session.
  const t = codeType(type);
  t.seeds = t.seeds || [];
  if (!t.seeds.includes(code)) t.seeds.push(code);
}
function bindHighlightClicks() {
  $$(".hl", $("#thread")).forEach(m => {
    m.onclick = () => { if (state.ann.highlights.find(x => x.id === m.dataset.hl)) flashCode(m.dataset.hl); };
  });
}
function deleteHighlight(id) {
  state.ann.highlights = state.ann.highlights.filter(h => h.id !== id);
  renderThread(); renderCodes(); refreshRailProgress(); scheduleSave();
}
function renderCodes() {
  const box = $("#codes-list");
  const intro = $("#codes-intro");
  if (!box) return;
  const types = codeTypes();
  if (intro) {
    intro.innerHTML = `<p>Drag the highlighter across any passage — in the
      model's reply or the user's message — then give the stroke a label.</p>`
      + types.map(t =>
      `<p class="ci-type"><b>${esc(t.label)}</b>${t.short ? ` · ${esc(t.short)}` : ""} —
        ${esc(t.help || "")}</p>`).join("");
  }

  const hls = (state.task === "qualitative" && state.ann && state.ann.highlights) || [];
  if (!hls.length) {
    box.innerHTML = `<div class="codes-empty">Nothing highlighted yet.<br>Drag across a passage to make your first stroke.</div>`;
    return;
  }

  box.innerHTML = "";
  // Descriptive and interpretive codes are analysed separately, so keep them
  // visually separate here too rather than mixing them in one frequency list.
  types.forEach(t => {
    const ofType = hls.filter(h => (h.code_type || types[0].id) === t.id);
    if (!ofType.length) return;
    const sec = document.createElement("div");
    sec.className = "code-section";
    sec.innerHTML = `<div class="code-section-head">${esc(t.label)}
      <span class="cs-count">${ofType.length}</span></div>`;

    const groups = {};
    ofType.forEach(h => (groups[h.code] = groups[h.code] || []).push(h));
    Object.entries(groups).sort((a, b) => b[1].length - a[1].length).forEach(([code, arr]) => {
      const g = document.createElement("div");
      g.className = "code-group";
      const col = cssVar(arr[0].color) || cssVar("--c0");
      g.innerHTML = `<div class="code-group-head">
          <span class="code-swatch" style="background:${col}"></span>
          <span class="code-name">${esc(code)}</span>
          <span class="code-count">${arr.length}</span></div>`;
      arr.forEach(h => {
        const loc = `turn ${h.turn} · ${h.target === "user" ? "user message" : "response"}`;
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
      sec.appendChild(g);
    });
    box.appendChild(sec);
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
/** Write out any debounced edit *before* the record it belongs to is replaced.
 *  Switching task, mode or unit swaps `state.ann`, which would otherwise strand
 *  the last 700ms of work. */
function flushPendingSave() {
  if (!state.saveTimer) return;
  clearTimeout(state.saveTimer); state.saveTimer = null;
  save();
}
async function save(explicitStatus) {
  if (!state.ann) return;
  if (explicitStatus) state.ann.status = explicitStatus;
  // Hold the record itself: by the time the request resolves the annotator may
  // already have moved on, and this save still has to land.
  const rec = state.ann;
  const recId = state.currentId;
  try {
    const res = await api("/api/annotation", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(rec),
    });
    rec.updated_at = res.updated_at;
    if (state.ann !== rec) return;   // moved on — don't touch the new unit's UI
    updateSaveStatus("saved");
    const it = state.items.find(i => i.unit_id === recId);
    if (it) {
      it.status = rec.status;
      refreshRailProgress();
    }
  } catch (e) {
    if (state.ann !== rec) return;
    updateSaveStatus("error"); toast("Save failed: " + e.message);
  }
}
function updateSaveStatus(kind) {
  const el = $("#save-status");
  el.className = "save-status " + (kind === "saving" ? "saving" : kind === "saved" ? "saved" : "");
  el.textContent = kind === "saving" ? "saving…"
    : kind === "saved" ? "saved " + new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
    : kind === "error" ? "save error" : "—";
}
async function submitUnit() {
  if (state.task === "qualitative") {
    // Coding has no completion criterion — the honest check is just that the
    // annotator left something behind.
    const n = (state.ann.highlights || []).length;
    const notes = (state.ann.conversation || {}).comment;
    if ((!n || !notes) && !confirm(
        `${n} label${n === 1 ? "" : "s"} added${notes ? "" : ", no overall notes written"}. Submit anyway?`)) return;
    await save("submitted");
    toast("Coding submitted ✓");
  } else if (mode() === "response") {
    if (!ratingComplete(state.ann.rating) &&
        !confirm("This response isn't fully rated. Submit anyway?")) return;
    await save("submitted");
    toast("Response submitted ✓");
  } else {
    const total = state.unit.turns.length;
    const rated = state.unit.turns.filter(t => ratingComplete(state.ann.responses[String(t.turn)] || {})).length;
    if (rated < total && !confirm(`${rated} of ${total} responses fully rated. Submit anyway?`)) return;
    await save("submitted");
    toast("Ratings submitted ✓");
  }
  const idx = state.items.findIndex(i => i.unit_id === state.currentId);
  const next = state.items.slice(idx + 1).find(i => i.status !== "submitted")
            || state.items.find(i => i.status !== "submitted");
  if (next) openUnit(next.unit_id);
}

boot();
