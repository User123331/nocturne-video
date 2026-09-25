/* Nocturne Video dashboard. */
"use strict";

const $ = (sel, el = document) => el.querySelector(sel);
const $$ = (sel, el = document) => [...el.querySelectorAll(sel)];

const SECTION_HINTS = {
  integrated_multimodal_description: "Shot-by-shot description. [Shot 1] … [Shot 2] At 00:02.500 hard cut …",
  overall_soundscape: "Diegetic sound: ambience, foley, footsteps, weather.",
  non_diegetic_music: "Score outside the scene. A blank line becomes N/A.",
  subject_definitions: "Define every <Picture i> / <Audio j> once.",
  summary: "One paragraph covering the whole clip.",
  retention_analysis: "What carries over from each reference.",
  detailed_description: "Full shot description; refer to references by their tags.",
};

/* Only these keys are Director sections; the flf2va/i2va alignment line is
   prose, not a section, so it renders as body text. */
const PROMPT_SECTION_KEYS = new Set([
  "integrated_multimodal_description", "overall_soundscape", "non_diegetic_music",
  "subject_definitions", "summary", "retention_analysis", "detailed_description",
]);

/* The three render tiers, filled with the workflow's own recommended
   settings (its Settings note lists, for the distilled build: euler/simple or
   lcm, shifts 6-12 video and 3-5 audio, 4 or 8 steps; for the non-distilled
   build: res_multistep/simple or euler, shifts 10-12 and 3-5, 20-25 steps).
   Draft is the distilled path, Studio and Final are the non-distilled one at
   the low and high end of the recommended step and shift ranges. */
const QUALITY_PRESETS = {
  draft: { sampler_name: "euler", scheduler: "simple", steps: 8, shift_video: 7, shift_audio: 4.5 },
  studio: { sampler_name: "res_multistep", scheduler: "simple", steps: 20, shift_video: 10, shift_audio: 4 },
  final: { sampler_name: "res_multistep", scheduler: "simple", steps: 25, shift_video: 11, shift_audio: 4 },
};
/* What the worker maps each tier to when no explicit fields are sent. */
const WORKER_QUALITY = { draft: "turbo", studio: "quality", final: "quality" };

const MP = 1024 * 1024;

const state = {
  mode: "t2va",
  quality: "studio",
  tierEdited: false,
  aspect: "auto",
  sections: {},
  files: { first_frame: null, last_frame: null, ref_images: [], ref_audios: [] },
  watermark: null,
  taskFilter: "",
  tagFilter: "",
  selected: null,
  promptSections: {},
  selection: new Set(),
  library: [],      // last rendered generations, for detail prev/next
  detailIndex: -1,
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

/* ---------- smart fit resolution ---------- */

/* DaSiWa Director: Auto aspect follows the first visual reference (4:3 with
   none); Auto resolution sizes the short side to 768px; MP presets divide a
   pixel budget by the aspect. Everything snaps to 32px (Div32). */
function firstAttachedImage() {
  return state.files.first_frame || state.files.last_frame
    || state.files.ref_images.find(Boolean) || null;
}

function aspectRatioOf(file) {
  if (file && file.w && file.h) return file.w / file.h;
  return 4 / 3;
}

function snap32(n) { return Math.max(256, Math.round(n / 32) * 32); }

function computeResolution() {
  const aspect = $("#aspect-chips .chip.active")?.dataset.aspect || "auto";
  const preset = $("#resolution-preset").value;
  if (aspect === "custom") {
    return { width: Number($("#width").value) || 1024, height: Number($("#height").value) || 768 };
  }
  const ratio = aspect === "auto" ? aspectRatioOf(firstAttachedImage())
    : aspect.split(":").reduce((a, b) => a / b);
  if (preset === "custom") {
    return { width: Number($("#width").value) || 1024, height: Number($("#height").value) || 768 };
  }
  let width, height;
  if (preset === "auto") {
    // 768 px short side, long side follows the ratio.
    if (ratio >= 1) { height = 768; width = 768 * ratio; }
    else { width = 768; height = 768 / ratio; }
  } else {
    const area = Number(preset) * MP;
    width = Math.sqrt(area * ratio);
    height = Math.sqrt(area / ratio);
  }
  width = snap32(width); height = snap32(height);
  // Mirror the worker's clamp: pin the constrained axis, derive the other
  // from the requested ratio so the frame never distorts.
  const ratioOut = width / height;
  if (width > 2560) { width = 2560; height = snap32(width / ratioOut); }
  if (height > 1440) { height = 1440; width = snap32(height * ratioOut); }
  const capArea = 2560 * 1440;
  if (width * height > capArea) {
    if (ratioOut >= 1) { width = snap32(Math.sqrt(capArea * ratioOut)); height = snap32(width / ratioOut); }
    else { height = snap32(Math.sqrt(capArea / ratioOut)); width = snap32(height * ratioOut); }
  }
  return { width, height };
}

function updateCanvasReadout() {
  const { width, height } = computeResolution();
  $("#canvas-readout").textContent = `${width}×${height}`;
  $("#custom-canvas").classList.toggle("hidden",
    !($("#aspect-chips .chip.active")?.dataset.aspect === "custom" || $("#resolution-preset").value === "custom"));
  const file = firstAttachedImage();
  $("#aspect-hint").textContent = ($("#aspect-chips .chip.active")?.dataset.aspect || "auto") === "auto"
    ? (file ? `Auto follows ${file.name} (${file.w}×${file.h}).` : "Auto follows the first attached image; 4:3 when nothing is attached.")
    : "Custom canvases snap to 32 px.";
}

const MAX_FRAMES = 362;

function updateDurationReadout() {
  const secs = Number($("#duration").value);
  const fps = Number($("#fps").value);
  let n = Math.max(5, Math.round(secs * fps));
  while (n % 17 !== 5) n += 1;
  const readout = $("#duration-readout");
  const over = n > MAX_FRAMES;
  readout.textContent = over
    ? `${(n / fps).toFixed(1)}s · ${n}f — over the ${MAX_FRAMES}-frame limit`
    : `${(n / fps).toFixed(1)}s · ${n}f`;
  readout.classList.toggle("over", over);
  return { frames: n, over };
}

function updateInterpNote() {
  const fps = Number($("#fps").value);
  const mult = Number($("#interp-multiplier").value);
  $("#interp-note").textContent = `RIFE ×${mult} · ${fps * mult} fps out`;
}

/* ---------- quality presets ---------- */

function applyQualityPreset(name) {
  const preset = QUALITY_PRESETS[name];
  if (!preset) return;
  $("#sampler").value = preset.sampler_name;
  $("#scheduler").value = preset.scheduler;
  $("#steps").value = preset.steps;
  $("#shift_video").value = preset.shift_video;
  $("#shift_audio").value = preset.shift_audio;
  setQuality(name);
}

function setQuality(name) {
  state.quality = name;
  $$("#quality-seg .seg").forEach((s) => {
    s.classList.toggle("active", s.dataset.quality === name);
  });
  $("#sampler-note").textContent = name === "custom"
    ? "custom values"
    : `${QUALITY_PRESETS[name].sampler_name} · ${QUALITY_PRESETS[name].steps} steps`;
}

/* Editing a field by hand leaves the tier buttons alone but records that the
   values no longer match a preset, so generate() sends them explicitly. */
function markCustomIfEdited() {
  const preset = QUALITY_PRESETS[state.quality];
  if (!preset) return;
  const edited = $("#sampler").value !== preset.sampler_name
    || $("#scheduler").value !== preset.scheduler
    || Number($("#steps").value) !== preset.steps
    || Number($("#shift_video").value) !== preset.shift_video
    || Number($("#shift_audio").value) !== preset.shift_audio;
  $$("#quality-seg .seg").forEach((s) => {
    s.classList.toggle("active", !edited && s.dataset.quality === state.quality);
  });
  $("#sampler-note").textContent = edited
    ? "custom values"
    : `${preset.sampler_name} · ${preset.steps} steps`;
  state.tierEdited = edited;
}

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
    count.textContent = `${ta.value.length}`;
    ta.addEventListener("input", () => { count.textContent = `${ta.value.length}`; });
    block.append(label, ta, count);
    wrap.appendChild(block);
  }
}

/* ---------- file slots ---------- */

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

/* Keep the natural size of image uploads: Auto aspect fits the canvas to it. */
async function withImageSize(file, entry) {
  if (!file.type.startsWith("image/")) return entry;
  return new Promise((resolve) => {
    const img = new Image();
    const url = URL.createObjectURL(file);
    img.onload = () => {
      entry.w = img.naturalWidth; entry.h = img.naturalHeight;
      URL.revokeObjectURL(url);
      resolve(entry);
    };
    img.onerror = () => { URL.revokeObjectURL(url); resolve(entry); };
    img.src = url;
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
  const entry = await withImageSize(file, { name: file.name, b64 });
  if (kind === "ref_images" || kind === "ref_audios") {
    state.files[kind][Number(index)] = entry;
  } else {
    state.files[kind] = entry;
  }
  paintSlot(slot, file);
  updateCanvasReadout();
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
  updateCanvasReadout();
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
    hint.textContent = "Add 1 to 9 reference images (identity, wardrobe, places) and optional audio refs. Refer to them as <Picture 1>, <Audio 1> in the prompt.";
    wrap.append(hint);
    for (let i = 0; i < 9; i++) wrap.append(makeSlot({ kind: "ref_images", index: i, label: `Pic ${i + 1}` }));
    for (let i = 0; i < 3; i++) wrap.append(makeSlot({ kind: "ref_audios", index: i, label: `Audio ${i + 1}` }));
  } else {
    hint.textContent = "Text-to-video+audio: describe shots, soundscape, and score. Nothing to attach.";
    wrap.append(hint);
  }
  if (state.mode === "t2va") wrap.classList.add("hidden");
  else wrap.classList.remove("hidden");
  $("#ref2va-extra").classList.toggle("hidden", state.mode !== "ref2va");
}

/* ---------- generate ---------- */

function buildUpscaleSpec() {
  const mode = $("#upscale-mode").value;
  if (mode === "off") return null;
  if (mode === "model") return { mode, model: $("#upscale-model").value };
  if (mode === "simple") {
    return { mode, multiplier: Number($("#simple-multiplier").value), interpolation: $("#simple-interp").value };
  }
  if (mode === "rtx") {
    return {
      mode,
      scale: Number($("#rtx-scale").value),
      upscale_quality: $("#rtx-quality").value,
      denoise: $("#rtx-denoise").checked,
      deblur: $("#rtx-deblur").checked,
    };
  }
  return { mode, precision: $("#h3-precision").value };
}

async function generate() {
  const prompt = {};
  for (const section of state.promptSections[state.mode] || []) {
    prompt[section] = $(`#sec-${section}`)?.value ?? "";
  }
  const { over } = updateDurationReadout();
  if (over) {
    toast(`Duration and frame rate together exceed the ${MAX_FRAMES}-frame limit`, "err");
    return;
  }
  const files = {};
  if (state.files.first_frame) files.first_frame = state.files.first_frame.b64;
  if (state.files.last_frame) files.last_frame = state.files.last_frame.b64;
  const refImages = state.files.ref_images.filter(Boolean);
  const refAudios = state.files.ref_audios.filter(Boolean);
  if (refImages.length) files.ref_images = refImages.map((f) => f.b64);
  if (refAudios.length) files.ref_audios = refAudios.map((f) => f.b64);

  const { width, height } = computeResolution();
  const body = {
    task: state.mode,
    // The tier maps to the worker's preset name; when a field was hand-edited
    // the explicit values below override whatever the preset would have set.
    quality: WORKER_QUALITY[state.quality] || "quality",
    prompt,
    files,
    duration_seconds: Number($("#duration").value),
    width, height,
    fps: Number($("#fps").value),
    checkpoint: $("#checkpoint").value,
  };
  if (state.mode === "ref2va") body.ref_image_size = $("#ref-image-size").value;

  // Always send the sampling values that are on screen: they are either the
  // tier's recommended settings or the user's edits, and sending them keeps the
  // record exact and makes /reuse faithful.
  const fields = {
    steps: $("#steps").value, sampler_name: $("#sampler").value,
    scheduler: $("#scheduler").value,
    shift_video: $("#shift_video").value, shift_audio: $("#shift_audio").value,
  };
  for (const [key, value] of Object.entries(fields)) {
    if (value === "") continue;
    const numeric = Number(value);
    body[key] = Number.isNaN(numeric) ? value : numeric;
  }
  if ($("#seed").value && $("#seed").value !== "random") body.seed = Number($("#seed").value);
  if ($("#frame-interp").checked) {
    body.frame_interpolation = true;
    body.interpolation_multiplier = Number($("#interp-multiplier").value);
  }
  if ($("#chunk-ffn").checked) {
    body.chunk_ffn = true;
    body.chunk_count = Number($("#chunk-count").value) || 4;
  }
  if ($("#cache-toggle").checked) {
    body.cache = {
      reuse_threshold: Number($("#cache-reuse").value),
      start_percent: Number($("#cache-start").value),
      end_percent: Number($("#cache-end").value),
      max_steps: Number($("#cache-max-steps").value),
    };
  }
  const upscale = buildUpscaleSpec();
  if (upscale) body.upscale = upscale;
  if ($("#watermark-toggle").checked) {
    if (!state.watermark) { toast("Drop a PNG for the watermark first"); return; }
    body.watermark = {
      image_b64: state.watermark.b64,
      position: $("#watermark-position").value,
      scale: Number($("#watermark-scale").value),
      transparency: Number($("#watermark-opacity").value),
    };
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

/* ---------- queue + endpoint status ---------- */

const PHASE_LABEL = {
  QUEUED: "Waiting for GPU", RUNNING: "Rendering", COMPLETED: "Done",
  FAILED: "Failed", CANCELLED: "Cancelled",
};

function elapsed(sinceIso) {
  if (!sinceIso) return "—";
  const start = new Date(sinceIso.endsWith("Z") ? sinceIso : sinceIso + "Z").getTime();
  const seconds = Math.max(0, Math.floor((Date.now() - start) / 1000));
  if (seconds < 60) return `${seconds}s`;
  return `${Math.floor(seconds / 60)}m ${String(seconds % 60).padStart(2, "0")}s`;
}

/* The endpoint's own counters, read from GET /health: how many workers are
   warm, what is queued, and what is running right now. This is the honest
   signal for "is the API actually working", so it drives the progress bar. */
function renderEndpointStatus(health) {
  const stateEl = $("#api-state");
  const fill = $("#api-progress");
  const set = (id, value) => { $(`#${id}`).textContent = value; };
  if (!health || health.error) {
    stateEl.textContent = "unreachable";
    fill.className = "progress-fill err";
    fill.style.width = "100%";
    set("api-workers", "—"); set("api-queue", "—");
    set("api-running", "—"); set("api-completed", "—");
    return;
  }
  const workers = health.workers || {};
  const jobs = health.jobs || {};
  const ready = workers.ready ?? 0;
  const throttled = workers.throttled ?? 0;
  const unhealthy = workers.unhealthy ?? 0;
  const inQueue = jobs.inQueue ?? 0;
  const running = jobs.inProgress ?? 0;
  const completed = jobs.completed ?? 0;
  set("api-workers", String(ready));
  set("api-queue", String(inQueue));
  set("api-running", String(running));
  set("api-completed", String(completed));

  // A worker that is warm and idle means the next job starts immediately;
  // otherwise the bar shows that we are waiting on capacity. Throttled workers
  // exist but are paused by the account's quota, which is a different problem
  // from a cold start and needs saying so.
  if (running > 0) {
    stateEl.textContent = "rendering";
    fill.className = "progress-fill indeterminate";
    fill.style.width = "";
  } else if (inQueue > 0) {
    stateEl.textContent = ready > 0 ? "dispatching" : "waiting for capacity";
    fill.className = "progress-fill indeterminate";
    fill.style.width = "";
  } else if (ready > 0) {
    stateEl.textContent = "ready";
    fill.className = "progress-fill done";
    fill.style.width = "100%";
  } else if (throttled > 0) {
    stateEl.textContent = `${throttled} throttled`;
    fill.className = "progress-fill err";
    fill.style.width = "100%";
  } else if (unhealthy > 0) {
    stateEl.textContent = `${unhealthy} unhealthy`;
    fill.className = "progress-fill err";
    fill.style.width = "100%";
  } else {
    stateEl.textContent = "cold start";
    fill.className = "progress-fill";
    fill.style.width = "0%";
  }
}

async function refreshQueue() {
  let jobs = [];
  try {
    ({ jobs } = await api("/api/jobs?limit=12"));
  } catch { return; }

  const active = jobs.filter((j) => ["QUEUED", "RUNNING"].includes(j.status));
  const newest = jobs[0];
  const pulse = $("#queue-pulse");
  pulse.className = "pulse";
  if (active.length) {
    pulse.classList.add("live");
    $("#queue-status").textContent = active.some((j) => j.status === "RUNNING")
      ? "Rendering" : "Queued";
  } else if (newest && newest.status === "FAILED") {
    pulse.classList.add("err");
    $("#queue-status").textContent = "Last job failed";
  } else if (newest) {
    pulse.classList.add("done");
    $("#queue-status").textContent = "Idle";
  } else {
    $("#queue-status").textContent = "Idle";
  }
  $("#queue-count").textContent = active.length ? `${active.length} active` : "";
  $("#queue-total").textContent = jobs.length ? `${jobs.length} shown` : "";

  // Current job + elapsed, from the most recent active job.
  const current = active[0];
  state.currentStartedAt = current ? current.created_at : null;
  if (current) {
    $("#api-job").textContent = `${current.task.toUpperCase()} ${current.id.slice(-6)}`;
    $("#api-elapsed").textContent = elapsed(current.created_at);
  } else {
    $("#api-job").textContent = "—";
    $("#api-elapsed").textContent = newest ? elapsed(newest.created_at) + " ago" : "—";
  }

  const list = $("#queue-list");
  list.innerHTML = "";
  if (!jobs.length) {
    list.innerHTML = '<li class="queue-empty">No jobs yet.</li>';
    return;
  }
  for (const job of jobs) {
    const li = document.createElement("li");
    li.className = "queue-item";
    const cls = { QUEUED: "run", RUNNING: "run", COMPLETED: "done", FAILED: "fail", CANCELLED: "fail" }[job.status] || "";
    const when = new Date(job.created_at.endsWith("Z") ? job.created_at : job.created_at + "Z")
      .toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    const canCancel = ["QUEUED", "RUNNING"].includes(job.status);
    li.innerHTML = `
      <div class="queue-item-top">
        <span class="qdot ${cls}"></span>
        <span class="qtask">${job.task.toUpperCase()}</span>
        <span class="qid mono">${job.id.slice(-6)}</span>
        ${canCancel ? '<button class="queue-cancel" title="Cancel this job">×</button>' : ""}
      </div>
      <div class="qmeta mono"><span>${PHASE_LABEL[job.status] || job.status}</span><span>${when}</span></div>
      ${job.error ? `<div class="qmeta mono"><span class="err" title="${escapeHtml(job.error)}">${escapeHtml(job.error.slice(0, 90))}</span></div>` : ""}`;
    li.querySelector(".queue-cancel")?.addEventListener("click", async () => {
      try { await api(`/api/jobs/${job.id}/cancel`, { method: "POST" }); refreshQueue(); }
      catch (err) { toast(err.message, "err"); }
    });
    list.appendChild(li);
  }
}

/* ---------- library ---------- */

const TAG_HUES = [16, 42, 92, 145, 200, 262, 320];
function tagColor(tag) {
  let h = 0;
  for (const ch of tag) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
  const hue = TAG_HUES[h % TAG_HUES.length];
  return `color: hsl(${hue} 45% 68%); border-color: hsl(${hue} 45% 40% / .5)`;
}

function escapeHtml(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function renderTagRail(tags) {
  const rail = $("#tag-rail");
  rail.innerHTML = '<button class="tag-chip" data-tag="">All</button>';
  for (const { tag, count } of tags.slice(0, 24)) {
    const chip = document.createElement("button");
    chip.className = "tag-chip" + (state.tagFilter === tag ? " active" : "");
    chip.dataset.tag = tag;
    chip.innerHTML = `#${escapeHtml(tag)}<b>${count}</b>`;
    chip.addEventListener("click", () => {
      state.tagFilter = state.tagFilter === tag ? "" : tag;
      refreshLibrary();
    });
    rail.appendChild(chip);
  }
  rail.firstChild.classList.toggle("active", !state.tagFilter);
}

async function refreshLibrary() {
  const q = $("#library-search").value.trim();
  const params = new URLSearchParams({
    q, task: state.taskFilter, tag: state.tagFilter,
    sort: $("#library-sort").value, limit: "60",
  });
  let data;
  try { data = await api(`/api/generations?${params}`); } catch { return; }
  state.library = data.generations;
  renderTagRail(data.tags || []);
  $("#library-count").textContent = `${data.stats.generations} item${data.stats.generations === 1 ? "" : "s"}`;
  const strip = $("#filmstrip");
  strip.innerHTML = "";
  $("#filmstrip-empty").classList.toggle("hidden", data.generations.length > 0);
  for (const gen of data.generations) {
    strip.appendChild(renderCard(gen));
  }
  $("#lib-count").textContent = data.stats.generations;
  updateSelectionBar();
}

function renderCard(gen) {
  const card = document.createElement("div");
  card.className = "card" + (state.selected === gen.gen_id ? " active" : "");
  const promptText = gen.prompt_text || gen.gen_id;
  const local = gen.video_path ? "" : '<span class="card-missing">not local</span>';
  const userTags = (gen.tags || [])
    .map((t) => `<span class="tag user" style="${tagColor(t)}">${escapeHtml(t)}</span>`).join("");
  const date = new Date(gen.created_at.endsWith("Z") ? gen.created_at : gen.created_at + "Z")
    .toLocaleDateString([], { day: "2-digit", month: "2-digit", year: "2-digit" });
  card.innerHTML = `
    ${gen.favorited ? '<button class="card-star on" title="Starred">★</button>' : '<button class="card-star" title="Star">☆</button>'}
    <input type="checkbox" class="card-check" aria-label="Select for tagging" ${state.selection.has(gen.gen_id) ? "checked" : ""}>
    ${gen.video_path ? `<video src="/video/${gen.gen_id}#t=0.1" preload="metadata" muted></video>` : '<video preload="metadata" muted></video>'}
    <div class="card-body">
      <div class="card-id mono">${gen.gen_id}</div>
      <div class="card-text">${escapeHtml(promptText)}</div>
      <div class="card-tags">
        <span class="tag">${gen.task.toUpperCase()}</span>
        ${gen.quality === "turbo" ? '<span class="tag plain">DRAFT</span>' : ""}
        ${gen.quality === "custom" ? '<span class="tag plain">CUSTOM</span>' : ""}
        ${gen.width ? `<span class="tag plain">${gen.width}×${gen.height}</span>` : ""}
        ${userTags}
        ${local}
        <span class="card-date">${date}</span>
      </div>
    </div>`;
  card.querySelector(".card-check").addEventListener("click", (e) => {
    e.stopPropagation();
    if (e.target.checked) state.selection.add(gen.gen_id);
    else state.selection.delete(gen.gen_id);
    updateSelectionBar();
  });
  card.querySelector(".card-star").addEventListener("click", async (e) => {
    e.stopPropagation();
    try {
      const res = await api(`/api/generations/${gen.gen_id}/favorite`, {
        method: "POST", body: { favorited: !gen.favorited },
      });
      gen.favorited = res.favorited;
      refreshLibrary();
    } catch (err) { toast(err.message, "err"); }
  });
  card.addEventListener("mouseenter", () => { const v = card.querySelector("video"); if (v?.src) v.play().catch(() => {}); });
  card.addEventListener("mouseleave", () => { const v = card.querySelector("video"); if (v) { v.pause(); v.currentTime = 0; } });
  card.addEventListener("click", (e) => {
    if (e.target.classList.contains("card-check")) return;
    openDetail(gen);
  });
  return card;
}

function updateSelectionBar() {
  const bar = $("#selection-bar");
  bar.classList.toggle("hidden", state.selection.size === 0);
  $("#selection-count").textContent = `${state.selection.size} selected`;
}

async function applySelectionTags() {
  const raw = $("#selection-tags").value.trim();
  if (!raw) { toast("Type one or more tags first"); return; }
  const tags = raw.split(",").map((t) => t.trim()).filter(Boolean);
  let done = 0;
  for (const genId of state.selection) {
    const gen = state.library.find((g) => g.gen_id === genId);
    const merged = [...new Set([...(gen?.tags || []), ...tags])];
    try {
      await api(`/api/generations/${genId}/tags`, { method: "POST", body: { tags: merged } });
      done++;
    } catch (err) { toast(err.message, "err"); }
  }
  state.selection.clear();
  $("#selection-tags").value = "";
  toast(`Tagged ${done} item${done === 1 ? "" : "s"}`, "ok");
  refreshLibrary();
}

/* ---------- detail / inspector ---------- */

function detailFields(gen) {
  const meta = gen.meta_json && typeof gen.meta_json === "object" ? gen.meta_json : {};
  const upscale = meta.upscale || null;
  const upLabel = !upscale ? "Off"
    : upscale.mode === "model" ? `Model · ${upscale.model}`
    : upscale.mode === "simple" ? `Simple · ×${upscale.multiplier} ${upscale.interpolation || ""}`.trim()
    : upscale.mode === "rtx" ? `RTX VSR · ×${upscale.scale}`
    : `H3 Latent · ${upscale.target_width || "2×"}${upscale.target_width ? `×${upscale.target_height}` : ""}`;
  const rows = [
    ["Seed", String(meta.seed ?? gen.seed ?? "—"), true],
    ["CFG", "1.00 · fixed"],
    ["Steps", String(meta.steps ?? "—")],
    ["Sampler", `${meta.sampler_name ?? "—"} / ${meta.scheduler ?? ""}`.trim()],
    ["Size", `${gen.width}×${gen.height}`],
    ["Frame rate", (() => {
      const genFps = meta.generation_fps ?? 24;
      const outFps = meta.fps ?? genFps;
      return outFps !== genFps ? `${genFps} fps → ${outFps} out` : `${genFps} fps`;
    })()],
    ["Duration", `${(gen.duration ?? meta.duration_seconds ?? 0).toFixed(2)}s · ${gen.frames ?? "—"}f`],
    ["Shifts", `${meta.shift_video ?? "—"} video / ${meta.shift_audio ?? "—"} audio`],
    ["Checkpoint", (meta.checkpoint || "").replace("dasiwa-hybrid-v2-", "Hybrid v2 ")] ,
    ["Tier", (() => {
      const hit = Object.entries(QUALITY_PRESETS).find(([, p]) =>
        p.sampler_name === meta.sampler_name && p.steps === meta.steps
        && p.shift_video === meta.shift_video);
      if (hit) return hit[0][0].toUpperCase() + hit[0].slice(1);
      return `${meta.steps ?? "?"} steps · custom`;
    })()],
    ["Interpolation", meta.frame_interpolation ? `RIFE ×${meta.interpolation_multiplier}` : "Off"],
    ["Upscale", upLabel],
    ["Cache", meta.cache ? `on · reuse ${meta.cache.reuse_threshold}` : "Off"],
    ["Chunking", meta.chunk_ffn ? `on · ${meta.chunk_count ?? 4} chunks` : "Off"],
    ["LoRAs", (meta.loras || []).map((l) => `${l.name} @ ${l.strength}`).join(", ") || "None"],
    ["Storage", gen.video_path ? "local + S3" : (gen.s3_uri ? "S3 only" : "—")],
  ];
  return rows;
}

function renderInspector(gen) {
  const grid = $("#inspector-grid");
  grid.innerHTML = "";
  for (const [k, v, amber] of detailFields(gen)) {
    const cell = document.createElement("div");
    cell.className = "field-card";
    cell.innerHTML = `<div class="k">${k}</div><div class="v ${amber ? "amber" : ""}">${escapeHtml(v)}</div>`;
    grid.appendChild(cell);
  }
  const meta = gen.meta_json && typeof gen.meta_json === "object" ? gen.meta_json : {};
  const prompts = $("#inspector-prompts");
  prompts.innerHTML = "";
  const finalText = meta.prompt_final || "";
  if (finalText) {
    for (const part of finalText.split(/\n\n+/)) {
      const idx = part.indexOf(": ");
      const key = idx === -1 ? "" : part.slice(0, idx);
      const isSection = PROMPT_SECTION_KEYS.has(key);
      const value = isSection ? part.slice(idx + 2) : part;
      if (!value.trim()) continue;
      const sec = document.createElement("div");
      sec.className = "psec";
      if (isSection) {
        sec.innerHTML = `<span class="field-label">${escapeHtml(key.replace(/_/g, " "))}</span>`;
      } else {
        sec.innerHTML = '<span class="field-label">Alignment</span>';
      }
      const p = document.createElement("p");
      p.textContent = value;
      sec.appendChild(p);
      prompts.appendChild(sec);
    }
  }
  renderDetailTags(gen);
  $("#btn-star").textContent = gen.favorited ? "★ Starred" : "☆ Star";
}

function renderDetailTags(gen) {
  const list = $("#detail-tag-list");
  list.innerHTML = "";
  for (const tag of gen.tags || []) {
    const chip = document.createElement("span");
    chip.className = "tag-chip";
    chip.style.cssText = tagColor(tag);
    chip.innerHTML = `#${escapeHtml(tag)}<span class="x" title="Remove">×</span>`;
    chip.querySelector(".x").addEventListener("click", async () => {
      const tags = (gen.tags || []).filter((t) => t !== tag);
      try {
        await api(`/api/generations/${gen.gen_id}/tags`, { method: "POST", body: { tags } });
        gen.tags = tags;
        renderDetailTags(gen);
        refreshLibrary();
      } catch (err) { toast(err.message, "err"); }
    });
    list.appendChild(chip);
  }
}

async function openDetail(gen) {
  state.selected = gen.gen_id;
  state.detailIndex = state.library.findIndex((g) => g.gen_id === gen.gen_id);
  $("#detail").classList.remove("hidden");
  document.body.style.overflow = "hidden";
  await showDetail(gen);
}

async function showDetail(gen) {
  $("#detail-title").textContent = `Generation · ${new Date(gen.created_at.endsWith("Z") ? gen.created_at : gen.created_at + "Z").toLocaleDateString()}`;
  $("#detail-size").textContent = gen.width ? `${gen.width} × ${gen.height}` : "";
  $("#detail-caption").textContent = gen.prompt_text || "";
  if (!gen.video_path) {
    try {
      await api(`/api/generations/${gen.gen_id}/sync`, { method: "POST" });
      gen.video_path = `/video/${gen.gen_id}`;
      await refreshLibrary();
    } catch (err) { toast(err.message, "err"); }
  }
  const player = $("#detail-player");
  player.src = `/video/${gen.gen_id}`;
  renderInspector(gen);
}

function closeDetail() {
  $("#detail").classList.add("hidden");
  document.body.style.overflow = "";
  $("#detail-player").pause();
}

function stepDetail(delta) {
  const idx = state.detailIndex + delta;
  if (idx < 0 || idx >= state.library.length) return;
  state.detailIndex = idx;
  const gen = state.library[idx];
  state.selected = gen.gen_id;
  showDetail(gen);
}

async function reuseSettings() {
  const genId = state.selected;
  try {
    const data = await api(`/api/generations/${genId}/reuse`);
    const spec = data.spec || {};
    const meta = data.meta || {};
    setMode(data.task || "t2va");
    state.sections = { ...state.sections, ...(data.prompt || {}) };
    renderPromptSections();
    // Canvas: explicit custom size, since the original aspect source is gone.
    $$("#aspect-chips .chip").forEach((c) => c.classList.toggle("active", c.dataset.aspect === "custom"));
    $("#resolution-preset").value = "custom";
    $("#width").value = meta.width || 1024;
    $("#height").value = meta.height || 768;
    $("#duration").value = Math.min(15, Math.max(1, Math.round(meta.duration_seconds || 5)));
    $("#fps").value = String(meta.generation_fps || 24);
    // Sampling: the spec only carries these when the user overrode them, so
    // fall back to the effective values the worker recorded in the meta.
    const eff = {
      sampler_name: spec.sampler_name ?? meta.sampler_name,
      scheduler: spec.scheduler ?? meta.scheduler,
      steps: spec.steps ?? meta.steps,
      shift_video: spec.shift_video ?? meta.shift_video,
      shift_audio: spec.shift_audio ?? meta.shift_audio,
    };
    const matches = Object.entries(QUALITY_PRESETS).find(([, p]) =>
      p.sampler_name === eff.sampler_name && p.scheduler === eff.scheduler
      && p.steps === eff.steps && p.shift_video === eff.shift_video
      && p.shift_audio === eff.shift_audio);
    if (matches) {
      applyQualityPreset(matches[0]);
    } else {
      $("#sampler").value = eff.sampler_name || "";
      $("#scheduler").value = eff.scheduler || "";
      $("#steps").value = eff.steps ?? "";
      $("#shift_video").value = eff.shift_video ?? "";
      $("#shift_audio").value = eff.shift_audio ?? "";
      setQuality("custom");
    }
    $("#seed").value = meta.seed ?? "";
    $("#frame-interp").checked = !!meta.frame_interpolation;
    $("#interp-multiplier").value = String(meta.interpolation_multiplier || 2);
    $("#chunk-ffn").checked = !!meta.chunk_ffn;
    $("#cache-toggle").checked = !!meta.cache;
    $("#upscale-mode").value = meta.upscale?.mode || "off";
    syncUpscaleUI();
    if (meta.upscale?.mode === "simple") {
      $("#simple-multiplier").value = meta.upscale.multiplier || 2;
      $("#simple-interp").value = meta.upscale.interpolation || "Lanczos";
    }
    closeDetail();
    window.scrollTo({ top: 0, behavior: "smooth" });
    toast("Settings loaded into the composer", "ok");
  } catch (err) { toast(err.message, "err"); }
}

/* ---------- upscale / watermark UI ---------- */

function syncUpscaleUI() {
  const mode = $("#upscale-mode").value;
  $("#upscale-params").classList.toggle("hidden", mode === "off");
  $$("#upscale-params [data-upscale]").forEach((el) => {
    el.classList.toggle("hidden", el.dataset.upscale !== mode);
  });
}

function setupWatermarkSlot() {
  const slot = $("#watermark-slot");
  const pick = () => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "image/png,image/*";
    input.onchange = async () => {
      const file = input.files[0];
      if (!file) return;
      slot.classList.add("filled");
      slot.textContent = file.name;
      state.watermark = { name: file.name, b64: await readAsB64(file) };
    };
    input.click();
  };
  slot.addEventListener("click", pick);
  slot.addEventListener("dragover", (e) => { e.preventDefault(); });
  slot.addEventListener("drop", async (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (!file) return;
    slot.classList.add("filled");
    slot.textContent = file.name;
    state.watermark = { name: file.name, b64: await readAsB64(file) };
  });
}

/* ---------- bootstrap / beacons ---------- */

async function bootstrap() {
  try {
    const data = await api("/api/bootstrap");
    const beaconEndpoint = $("#beacon-endpoint");
    if (!data.endpoint_id) {
      setBeacon("endpoint", "err", "no endpoint");
      renderEndpointStatus(null);
    } else if (data.endpoint_health?.error) {
      setBeacon("endpoint", "err", "unreachable");
      renderEndpointStatus({ error: data.endpoint_health.error });
    } else {
      setBeacon("endpoint", "ok", data.endpoint_id);
      renderEndpointStatus(data.endpoint_health);
    }
    const s3 = data.s3 || {};
    if (!s3.configured) setBeacon("s3", "err", "no credentials");
    else if (s3.error) setBeacon("s3", "err", "unreachable");
    else { setBeacon("s3", "ok", `${s3.objects} objects`); $("#s3-objects").textContent = s3.objects; $("#s3-bytes").textContent = bytesFmt(s3.bytes); }
  } catch (err) {
    setBeacon("endpoint", "err", err.message);
    renderEndpointStatus({ error: err.message });
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
  $$("#quality-seg .seg").forEach((seg) => seg.addEventListener("click", () => {
    applyQualityPreset(seg.dataset.quality);
  }));
  for (const id of ["sampler", "scheduler", "steps", "shift_video", "shift_audio"]) {
    $(`#${id}`).addEventListener("input", markCustomIfEdited);
  }
  $$("#aspect-chips .chip").forEach((chip) => chip.addEventListener("click", () => {
    $$("#aspect-chips .chip").forEach((c) => c.classList.toggle("active", c === chip));
    updateCanvasReadout();
  }));
  $("#resolution-preset").addEventListener("change", () => { updateCanvasReadout(); updateDurationReadout(); });
  $("#width").addEventListener("input", updateCanvasReadout);
  $("#height").addEventListener("input", updateCanvasReadout);
  $("#duration").addEventListener("input", updateDurationReadout);
  $("#fps").addEventListener("change", () => { updateDurationReadout(); updateInterpNote(); });
  $("#interp-multiplier").addEventListener("change", updateInterpNote);
  $("#frame-interp").addEventListener("change", () => {
    $("#interp-params").classList.toggle("hidden", !$("#frame-interp").checked);
    updateInterpNote();
  });
  $("#chunk-ffn").addEventListener("change", () => {
    $("#chunk-params").classList.toggle("hidden", !$("#chunk-ffn").checked);
  });
  $("#cache-toggle").addEventListener("change", () => {
    $("#cache-params").classList.toggle("hidden", !$("#cache-toggle").checked);
  });
  $("#upscale-mode").addEventListener("change", syncUpscaleUI);
  $("#watermark-toggle").addEventListener("change", () => {
    $("#watermark-params").classList.toggle("hidden", !$("#watermark-toggle").checked);
  });
  setupWatermarkSlot();
  $("#btn-dice").addEventListener("click", () => {
    $("#seed").value = String(Math.floor(Math.random() * 2 ** 53));
  });
  $("#btn-add-lora").addEventListener("click", () => addLoraRow());
  $("#btn-generate").addEventListener("click", generate);
  $("#library-search").addEventListener("input", debounce(refreshLibrary, 280));
  $("#library-sort").addEventListener("change", refreshLibrary);
  $("#btn-apply-tags").addEventListener("click", applySelectionTags);
  $("#btn-clear-selection").addEventListener("click", () => {
    state.selection.clear();
    refreshLibrary();
  });
  $("#btn-back").addEventListener("click", closeDetail);
  $("#detail-prev").addEventListener("click", () => stepDetail(-1));
  $("#detail-next").addEventListener("click", () => stepDetail(1));
  $("#btn-star").addEventListener("click", async () => {
    const gen = state.library[state.detailIndex];
    if (!gen) return;
    try {
      const res = await api(`/api/generations/${gen.gen_id}/favorite`, {
        method: "POST", body: { favorited: !gen.favorited },
      });
      gen.favorited = res.favorited;
      $("#btn-star").textContent = gen.favorited ? "★ Starred" : "☆ Star";
      refreshLibrary();
    } catch (err) { toast(err.message, "err"); }
  });
  $("#btn-download").addEventListener("click", () => {
    const a = document.createElement("a");
    a.href = `/video/${state.selected}`;
    a.download = `${state.selected}.mp4`;
    a.click();
  });
  $("#btn-reuse").addEventListener("click", reuseSettings);
  $("#btn-sync-one").addEventListener("click", async () => {
    try {
      await api(`/api/generations/${state.selected}/sync`, { method: "POST" });
      toast("Synced from S3", "ok");
      refreshLibrary();
    } catch (err) { toast(err.message, "err"); }
  });
  $("#btn-fullscreen").addEventListener("click", () => {
    const v = $("#detail-player");
    if (v.requestFullscreen) v.requestFullscreen();
  });
  $("#btn-raw").addEventListener("click", async () => {
    try {
      const data = await api(`/api/generations/${state.selected}/reuse`);
      $("#raw-json").textContent = JSON.stringify({ meta: data.meta, spec: data.spec }, null, 2);
      $("#raw-modal").classList.remove("hidden");
    } catch (err) { toast(err.message, "err"); }
  });
  $("#btn-close-raw").addEventListener("click", () => $("#raw-modal").classList.add("hidden"));
  $("#raw-modal").addEventListener("click", (e) => {
    if (e.target === e.currentTarget) $("#raw-modal").classList.add("hidden");
  });
  $("#detail-tag-input").addEventListener("keydown", async (e) => {
    if (e.key !== "Enter") return;
    const gen = state.library[state.detailIndex];
    if (!gen) return;
    const tag = e.target.value.trim().replace(/^#/, "");
    if (!tag) return;
    const tags = [...new Set([...(gen.tags || []), tag])];
    try {
      await api(`/api/generations/${gen.gen_id}/tags`, { method: "POST", body: { tags } });
      gen.tags = tags;
      e.target.value = "";
      renderDetailTags(gen);
      refreshLibrary();
    } catch (err) { toast(err.message, "err"); }
  });
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
      toast("Stage volume queued: the worker verifies every weight and reports gaps.", "ok");
      refreshQueue();
    } catch (err) { toast(err.message, "err"); }
  });
  document.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") { e.preventDefault(); generate(); }
    if (e.key === "Escape") {
      if (!$("#raw-modal").classList.contains("hidden")) { $("#raw-modal").classList.add("hidden"); return; }
      if (!$("#detail").classList.contains("hidden")) closeDetail();
      else $("#player").pause();
    }
    if (!$("#detail").classList.contains("hidden")) {
      if (e.key === "ArrowLeft") stepDetail(-1);
      if (e.key === "ArrowRight") stepDetail(1);
    }
  });
  document.addEventListener("paste", async (e) => {
    const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
    if (!item) return;
    const file = item.getAsFile();
    const openSlots = $$(".drop-slot").filter((s) => !s.classList.contains("filled") && s.dataset.kind !== "ref_audios");
    if (openSlots.length) await assignFile(openSlots[0], file);
  });
  setMode("t2va");
  applyQualityPreset("studio");
  updateDurationReadout();
  updateCanvasReadout();
  refreshBootstrapLoop();
  refreshQueue();
  refreshLibrary();
  setInterval(refreshQueue, 5000);
  setInterval(refreshLibrary, 30000);
  // Tick the elapsed clock between queue polls so a long render visibly moves.
  setInterval(() => {
    if (state.currentStartedAt) $("#api-elapsed").textContent = elapsed(state.currentStartedAt);
  }, 1000);
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
