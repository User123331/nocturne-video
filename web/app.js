/* Nocturne Video dashboard. */
"use strict";

const $ = (sel, el = document) => el.querySelector(sel);
const $$ = (sel, el = document) => [...el.querySelectorAll(sel)];

const SECTION_HINTS = {
  integrated_multimodal_description:
    "Shot-by-shot description. [Shot 1] … [Shot 2] At 00:02.500 hard cut …",
  overall_soundscape: "Diegetic sound: ambience, foley, footsteps, weather.",
  non_diegetic_music: "Score outside the scene. Leave empty for N/A.",
  subject_definitions:
    "Define every <Picture i> / <Video k> / <Audio j> once, e.g. <Picture 1> is a leather jacket on a chair.",
  summary: "One-paragraph summary of the whole clip.",
  retention_analysis:
    "What must carry over from each reference: wardrobe, face, palette, lighting.",
  detailed_description: "Full shot description; refer to references by their tags.",
};

const state = {
  mode: "t2va",
  quality: "quality",
  sections: {},
  files: { first_frame: null, last_frame: null, ref_images: [], ref_audios: [] },
  taskFilter: "",
  selected: null,
  promptSections: {},
  elapsedTimer: null,
};

async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

function toast(message, kind = "") {
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.textContent = message;
  $("#toast-rail").appendChild(el);
  setTimeout(() => el.remove(), 5200);
}

const bytesFmt = (n) => {
  if (n == null) return "—";
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  let i = 0, v = n;
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i++; }
  return `${v.toFixed(v >= 10 || i === 0 ? 0 : 1)} ${units[i]}`;
};

/* ---------- prompt sections per mode ---------- */

function renderPromptSections() {
  const wrap = $("#prompt-blocks");
  wrap.innerHTML = "";
  for (const section of state.promptSections[state.mode] || []) {
    const block = document.createElement("div");
    block.className = "prompt-block";
    const label = document.createElement("span");
    label.className = "field-label";
    label.textContent = section.replace(/_/g, " ");
    const ta = document.createElement("textarea");
    ta.id = `sec-${section}`;
    ta.placeholder = SECTION_HINTS[section] || "";
    ta.value = state.sections[section] || "";
    ta.addEventListener("input", () => { state.sections[section] = ta.value; });
    const count = document.createElement("div");
    count.className = "char-count mono dim";
    ta.addEventListener("input", () => { count.textContent = `${ta.value.length}`; });
    block.append(label, ta, count);
    if (SECTION_HINTS[section]) {
      const hint = document.createElement("div");
      hint.className = "prompt-hint";
      hint.textContent = SECTION_HINTS[section];
      block.append(hint);
    }
    wrap.appendChild(block);
  }
}

/* ---------- drop slots ---------- */

function clearFiles() {
  state.files = { first_frame: null, last_frame: null, ref_images: [], ref_audios: [] };
}

function readAsB64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(",")[1]);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function makeSlot({ kind, index, label, large }) {
  const slot = document.createElement("div");
  slot.className = `drop-slot ${large ? "large" : "small"}`;
  slot.dataset.kind = kind;
  slot.dataset.index = index ?? "";
  const lab = document.createElement("span");
  lab.className = "slot-label";
  lab.textContent = label;
  const name = document.createElement("span");
  name.className = "slot-name";
  slot.append(lab, name);
  slot.addEventListener("click", (e) => {
    if (e.target.classList.contains("slot-clear")) return;
    pickFile(slot);
  });
  slot.addEventListener("dragover", (e) => { e.preventDefault(); slot.classList.add("dragover"); });
  slot.addEventListener("dragleave", () => slot.classList.remove("dragover"));
  slot.addEventListener("drop", async (e) => {
    e.preventDefault();
    slot.classList.remove("dragover");
    await assignFile(slot, e.dataTransfer.files[0]);
  });
  return slot;
}

function pickFile(slot) {
  const input = document.createElement("input");
  input.type = "file";
  input.accept = slot.dataset.kind === "ref_audios" ? "audio/*" : "image/*";
  input.onchange = () => assignFile(slot, input.files[0]);
  input.click();
}

async function assignFile(slot, file) {
  if (!file) return;
  if (file.size > 12 * 1024 * 1024) {
    toast(`${file.name} is over 12 MiB; use a smaller image or a trimmed audio clip`, "err");
    return;
  }
  const kind = slot.dataset.kind;
  const index = slot.dataset.index;
  const b64 = await readAsB64(file);
  if (kind === "ref_images" || kind === "ref_audios") {
    state.files[kind][Number(index)] = { name: file.name, b64 };
  } else {
    state.files[kind] = { name: file.name, b64 };
  }
  paintSlot(slot, file);
}

function paintSlot(slot, file) {
  slot.classList.add("filled");
  slot.querySelector(".slot-name").textContent = file.name;
  let clear = slot.querySelector(".slot-clear");
  if (!clear) {
    clear = document.createElement("button");
    clear.className = "slot-clear";
    clear.textContent = "×";
    clear.title = "Remove";
    clear.addEventListener("click", () => unassign(slot));
    slot.appendChild(clear);
  }
  if (file.type.startsWith("image/")) {
    let img = slot.querySelector("img.slot-preview");
    if (!img) {
      img = document.createElement("img");
      img.className = "slot-preview";
      slot.prepend(img);
    }
    img.src = URL.createObjectURL(file);
  }
}

function unassign(slot) {
  const kind = slot.dataset.kind;
  const index = slot.dataset.index;
  if (kind === "ref_images" || kind === "ref_audios") {
    delete state.files[kind][Number(index)];
  } else {
    state.files[kind] = null;
  }
  slot.classList.remove("filled");
  slot.querySelector(".slot-preview")?.remove();
  slot.querySelector(".slot-clear")?.remove();
  slot.querySelector(".slot-name").textContent = "";
}

function renderDropzones() {
  const wrap = $("#dropzone-block");
  wrap.innerHTML = "";
  const hint = document.createElement("div");
  hint.className = "drop-hint";
  if (state.mode === "i2va") {
    hint.textContent = "Drop the first frame; the model animates from it (drag, click, or paste).";
    wrap.append(hint, makeSlot({ kind: "first_frame", label: "First frame", large: true }));
  } else if (state.mode === "flf2va") {
    hint.textContent = "Drop a first and last frame; H3 films the motion between them.";
    wrap.append(hint,
      makeSlot({ kind: "first_frame", label: "First frame", large: true }),
      makeSlot({ kind: "last_frame", label: "Last frame", large: true }));
  } else if (state.mode === "ref2va") {
    hint.textContent = "Add 1–9 reference images (identity, wardrobe, places) and optional audio refs. Refer to them as <Picture 1>, <Audio 1> in the prompt.";
    wrap.append(hint);
    for (let i = 0; i < 9; i++) wrap.append(makeSlot({ kind: "ref_images", index: i, label: `Pic ${i + 1}` }));
    for (let i = 0; i < 3; i++) wrap.append(makeSlot({ kind: "ref_audios", index: i, label: `Audio ${i + 1}` }));
  } else {
    hint.textContent = "Text-to-video+audio: describe shots, soundscape, and score. Nothing to attach.";
    wrap.append(hint);
  }
  if (state.mode === "t2va") wrap.classList.add("hidden");
  else wrap.classList.remove("hidden");
}

/* ---------- duration / canvas ---------- */

function snapFrames(seconds) {
  let n = Math.max(5, Math.round(seconds * 24));
  while (n % 17 !== 5) n += 1;
  return n;
}

function updateDurationReadout() {
  const secs = Number($("#duration").value);
  const frames = snapFrames(secs);
  $("#duration-readout").textContent = `${(frames / 24).toFixed(1)}s · ${frames}f`;
}

function updateCanvasReadout() {
  const value = $("#aspect").value;
  const custom = value === "custom";
  $("#custom-canvas").classList.toggle("hidden", !custom);
  const [w, h] = custom
    ? [$("#width").value || 1024, $("#height").value || 768]
    : value.split("x");
  $("#canvas-readout").textContent = `${w}×${h}`;
}

/* ---------- generate ---------- */

async function generate() {
  const prompt = {};
  for (const section of state.promptSections[state.mode] || []) {
    prompt[section] = $(`#sec-${section}`)?.value ?? "";
  }
  const files = {};
  if (state.files.first_frame) files.first_frame = state.files.first_frame.b64;
  if (state.files.last_frame) files.last_frame = state.files.last_frame.b64;
  const refImages = state.files.ref_images.filter(Boolean);
  const refAudios = state.files.ref_audios.filter(Boolean);
  if (refImages.length) files.ref_images = refImages.map((f) => f.b64);
  if (refAudios.length) files.ref_audios = refAudios.map((f) => f.b64);

  const body = {
    task: state.mode,
    quality: state.quality,
    prompt,
    files,
    duration_seconds: Number($("#duration").value),
  };
  const aspect = $("#aspect").value;
  if (aspect !== "custom") {
    const [w, h] = aspect.split("x").map(Number);
    body.width = w; body.height = h;
  } else {
    body.width = Number($("#width").value);
    body.height = Number($("#height").value);
  }
  const advanced = {
    steps: $("#steps").value, sampler_name: $("#sampler").value,
    shift_video: $("#shift_video").value, shift_audio: $("#shift_audio").value,
    seed: $("#seed").value,
  };
  for (const [k, v] of Object.entries(advanced)) {
    if (v !== "" && !(k === "seed" && v === "random")) body[k] = k === "seed" ? Number(v) : v;
  }
  if ($("#frame-interp").checked) {
    body.frame_interpolation = true;
    body.interpolation_multiplier = 2;
  }
  if ($("#chunk-ffn").checked) {
    body.chunk_ffn = true;
    body.chunk_count = 4;
  }
  const loras = $$("#lora-rows .lora-row")
    .map((row) => ({
      name: row.querySelector(".lora-name").value.trim(),
      strength: Number(row.querySelector(".strength").value || 1),
    }))
    .filter((l) => l.name);
  if (loras.length) body.loras = loras;

  const btn = $("#btn-generate");
  btn.disabled = true;
  try {
    await api("/api/generate", { method: "POST", body });
    toast("Queued. The worker is waking up.", "ok");
    refreshQueue();
  } catch (err) {
    toast(err.message, "err");
  } finally {
    btn.disabled = false;
  }
}

/* ---------- queue ---------- */

async function refreshQueue() {
  let jobs = [];
  try {
    ({ jobs } = await api("/api/jobs?limit=12"));
  } catch { return; }
  const list = $("#queue-list");
  list.innerHTML = "";
  const active = jobs.filter((j) => ["QUEUED", "RUNNING"].includes(j.status)).length;
  $("#queue-count").textContent = active ? `${active} running` : "idle";
  if (!jobs.length) {
    list.innerHTML = '<li class="queue-empty">no jobs yet</li>';
    return;
  }
  for (const job of jobs) {
    const li = document.createElement("li");
    li.className = "queue-item";
    const cls = { QUEUED: "run", RUNNING: "run", COMPLETED: "done", FAILED: "fail", CANCELLED: "fail" }[job.status] || "";
    const created = new Date(job.created_at + "Z").toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    li.innerHTML = `
      <div class="queue-item-top">
        <span class="qdot ${cls}"></span>
        <span class="qtask">${job.task.toUpperCase()}</span>
        <span class="qid mono">${job.id.slice(-10)}</span>
        ${["QUEUED", "RUNNING"].includes(job.status)
          ? '<button class="queue-cancel" title="Cancel">×</button>' : ""}
      </div>
      <div class="qmeta mono"><span>${job.status}</span><span>${created}</span></div>
      ${job.error ? `<div class="qmeta mono" style="color:var(--red)"><span>${job.error.slice(0, 140)}</span></div>` : ""}`;
    li.querySelector(".queue-cancel")?.addEventListener("click", async () => {
      try { await api(`/api/jobs/${job.id}/cancel`, { method: "POST" }); refreshQueue(); }
      catch (err) { toast(err.message, "err"); }
    });
    list.appendChild(li);
  }
}

/* ---------- library ---------- */

async function refreshLibrary() {
  const q = $("#library-search").value.trim();
  const params = new URLSearchParams({ q, task: state.taskFilter, limit: "48" });
  let data;
  try { data = await api(`/api/generations?${params}`); } catch { return; }
  const strip = $("#filmstrip");
  strip.innerHTML = "";
  $("#filmstrip-empty").classList.toggle("hidden", data.generations.length > 0);
  for (const gen of data.generations) {
    const card = document.createElement("div");
    card.className = "card" + (state.selected === gen.gen_id ? " active" : "");
    const promptText = gen.prompt_text || gen.gen_id;
    const local = gen.video_path ? "" : '<span class="card-missing">not local — sync needed</span>';
    card.innerHTML = `
      ${gen.video_path ? `<video src="/video/${gen.gen_id}" preload="metadata" muted></video>` : '<video preload="metadata" muted></video>'}
      <div class="card-body">
        <div class="card-id mono">${gen.gen_id}</div>
        <div class="card-text">${promptText.replace(/</g, "&lt;")}</div>
        <div class="card-tags">
          <span class="tag">${gen.task.toUpperCase()}</span>
          ${gen.quality === "turbo" ? '<span class="tag plain">TURBO</span>' : ""}
          ${gen.duration ? `<span class="tag plain">${gen.duration.toFixed(1)}s</span>` : ""}
          ${local}
        </div>
      </div>`;
    card.addEventListener("mouseenter", () => { const v = card.querySelector("video"); if (v?.src) v.play().catch(() => {}); });
    card.addEventListener("mouseleave", () => { const v = card.querySelector("video"); if (v) { v.pause(); v.currentTime = 0; } });
    card.addEventListener("click", () => loadIntoStage(gen));
    strip.appendChild(card);
  }
  $("#lib-count").textContent = data.stats.generations;
}

async function loadIntoStage(gen) {
  state.selected = gen.gen_id;
  $$(".card").forEach((c) => c.classList.toggle("active", c.querySelector(".card-id")?.textContent === gen.gen_id));
  if (!gen.video_path) {
    try { await api(`/api/generations/${gen.gen_id}/sync`, { method: "POST" }); }
    catch (err) { toast(err.message, "err"); return; }
  }
  const player = $("#player");
  player.src = `/video/${gen.gen_id}`;
  player.classList.remove("hidden");
  $("#stage-empty").classList.add("hidden");
  const meta = gen.meta_json || {};
  $("#stage-meta").classList.remove("hidden");
  $("#stage-meta").innerHTML = `
    <span><b>${(gen.task || "").toUpperCase()}</b></span>
    <span>seed <b>${gen.seed ?? "—"}</b></span>
    <span>${gen.width}×${gen.height}</span>
    <span>${gen.frames}f · ${(gen.duration || 0).toFixed(1)}s${meta.fps && meta.fps !== 24 ? ` · ${meta.fps}fps RIFE` : ""}</span>
    <span>${meta.steps || ""} steps · ${meta.sampler_name || ""}</span>
    <span>shift ${meta.shift_video ?? "—"} / ${meta.shift_audio ?? "—"}</span>
    <span class="dim">${(gen.sha256 || "").slice(0, 12)}</span>`;
  player.play().catch(() => {});
  window.scrollTo({ top: 0, behavior: "smooth" });
}

/* ---------- bootstrap / beacons ---------- */

async function bootstrap() {
  try {
    const data = await api("/api/bootstrap");
    const beaconEndpoint = $("#beacon-endpoint");
    if (!data.endpoint_id) setBeacon("endpoint", "err", "no endpoint");
    else if (data.endpoint_health?.error) setBeacon("endpoint", "err", "unreachable");
    else { setBeacon("endpoint", "ok", data.endpoint_id); }
    const s3 = data.s3 || {};
    if (!s3.configured) setBeacon("s3", "err", "no credentials");
    else if (s3.error) setBeacon("s3", "err", "unreachable");
    else { setBeacon("s3", "ok", `${s3.objects} objects`); $("#s3-objects").textContent = s3.objects; $("#s3-bytes").textContent = bytesFmt(s3.bytes); }
  } catch (err) {
    setBeacon("endpoint", "err", err.message);
  }
}

function setBeacon(which, stateName, title) {
  const beacon = $(`#beacon-${which}`);
  beacon.dataset.state = stateName;
  beacon.title = title;
}

/* ---------- wiring ---------- */

function setMode(mode) {
  state.mode = mode;
  $$(".mode-pill").forEach((p) => p.classList.toggle("active", p.dataset.mode === mode));
  clearFiles();
  renderPromptSections();
  renderDropzones();
}

function addLoraRow(name = "", strength = 1) {
  const row = document.createElement("div");
  row.className = "lora-row";
  row.innerHTML = `
    <input type="text" class="input lora-name mono" placeholder="my_lora.safetensors" value="${name.replace(/"/g, "&quot;")}">
    <input type="number" class="input strength mono" step="0.05" min="-4" max="4" value="${strength}">
    <button type="button" class="icon-btn remove" title="Remove">×</button>`;
  row.querySelector(".remove").addEventListener("click", () => row.remove());
  $("#lora-rows").appendChild(row);
}

function init() {
  $$(".mode-pill").forEach((pill) => pill.addEventListener("click", () => setMode(pill.dataset.mode)));
  $$(".seg").forEach((seg) => seg.addEventListener("click", () => {
    state.quality = seg.dataset.quality;
    $$(".seg").forEach((s) => s.classList.toggle("active", s === seg));
  }));
  $$("#task-chips .chip").forEach((chip) => chip.addEventListener("click", () => {
    state.taskFilter = chip.dataset.task;
    $$("#task-chips .chip").forEach((c) => c.classList.toggle("active", c === chip));
    refreshLibrary();
  }));
  $("#aspect").addEventListener("change", updateCanvasReadout);
  $("#width").addEventListener("input", updateCanvasReadout);
  $("#height").addEventListener("input", updateCanvasReadout);
  $("#duration").addEventListener("input", updateDurationReadout);
  $("#btn-dice").addEventListener("click", () => {
    $("#seed").value = String(Math.floor(Math.random() * 2 ** 53));
  });
  $("#btn-add-lora").addEventListener("click", () => addLoraRow());
  $("#btn-generate").addEventListener("click", generate);
  $("#library-search").addEventListener("input", debounce(refreshLibrary, 280));
  $("#btn-sync").addEventListener("click", async () => {
    toast("Syncing missing videos…");
    try {
      const res = await api("/api/sync-missing", { method: "POST" });
      toast(`Synced ${res.synced.length}` + (res.failed.length ? `, ${res.failed.length} failed` : ""),
        res.failed.length ? "err" : "ok");
      refreshLibrary(); refreshQueue();
    } catch (err) { toast(err.message, "err"); }
  });
  $("#btn-healthcheck").addEventListener("click", async () => {
    try {
      await api("/api/healthcheck", { method: "POST" });
      toast("Health-check queued: the worker will verify and stage the volume.", "ok");
      refreshQueue();
    } catch (err) { toast(err.message, "err"); }
  });
  document.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") { e.preventDefault(); generate(); }
    if (e.key === "Escape") { const p = $("#player"); p.pause(); }
  });
  document.addEventListener("paste", async (e) => {
    const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
    if (!item) return;
    const file = item.getAsFile();
    const openSlots = $$(".drop-slot").filter((s) => !s.classList.contains("filled") && s.dataset.kind !== "ref_audios");
    if (openSlots.length) await assignFile(openSlots[0], file);
  });
  refreshBootstrapLoop();
  setMode("t2va");
  updateDurationReadout();
  updateCanvasReadout();
  refreshQueue();
  refreshLibrary();
  setInterval(refreshQueue, 5000);
  setInterval(refreshLibrary, 20000);
}

function refreshBootstrapLoop() {
  bootstrap();
  setTimeout(refreshBootstrapLoop, 30000);
}

function debounce(fn, ms) {
  let t;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

(async function loadTemplates() {
  try {
    const data = await api("/api/templates");
    state.promptSections = data.sections;
  } catch { state.promptSections = {}; }
  init();
})();
