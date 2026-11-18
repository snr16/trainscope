/* TrainScope dashboard. Plain JS, no build step. Talks to the ingest API on the same origin. */
(() => {
  "use strict";

  // ------------------------------------------------------------------ helpers
  const $ = (sel, root = document) => root.querySelector(sel);
  const store = {
    get(k, d) { try { const v = localStorage.getItem("ts." + k); return v === null ? d : JSON.parse(v); } catch { return d; } },
    set(k, v) { try { localStorage.setItem("ts." + k, JSON.stringify(v)); } catch { /* private mode */ } },
  };
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const num = (v, sig = 3) => {
    if (v === null || v === undefined || v === "") return "–";
    if (typeof v === "string") return v; // "nan" / "inf" from the API
    if (!isFinite(v)) return String(v);
    const a = Math.abs(v);
    if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(2);
    return Number(v.toPrecision(sig)).toLocaleString(undefined, { maximumFractionDigits: 6 });
  };
  const pct = (v) => (v === null || v === undefined || typeof v === "string") ? "–" : `${(v * 100).toFixed(v < 0.1 ? 1 : 0)}%`;
  const ago = (t) => {
    if (!t) return "–";
    const s = Date.now() / 1000 - t;
    if (s < 90) return "just now";
    if (s < 3600) return `${Math.round(s / 60)} min ago`;
    if (s < 86400) return `${Math.round(s / 3600)} h ago`;
    return new Date(t * 1000).toLocaleDateString();
  };
  const CAUSE_NAMES = {
    lr_too_high: "LR too high", missing_warmup: "missing warmup", scheduler_misstep: "scheduler stepped wrong",
    fp16_no_loss_scaling: "fp16 without loss scaling", nan_from_op: "NaN from an op", exploding_no_clip: "exploding grads, no clip",
    frozen_layers: "frozen layers", double_softmax: "double softmax", bad_init: "bad initialization",
    shuffled_labels: "shuffled labels", val_leakage: "val leakage", missing_normalization: "missing normalization",
    bad_padding_mask: "wrong padding mask", missing_zero_grad: "missing zero_grad", grad_accum_unscaled: "unscaled grad accumulation",
    eval_mode_training: "left in eval mode", overfitting: "overfitting", dataloader_bottleneck: "data-loader bottleneck",
    memory_leak: "memory leak", healthy: "healthy",
  };
  const causeName = (c) => CAUSE_NAMES[c] || c;

  let toastTimer;
  const toast = (msg) => {
    const t = $("#toast"); t.textContent = msg; t.classList.add("on");
    clearTimeout(toastTimer); toastTimer = setTimeout(() => t.classList.remove("on"), 2600);
  };

  // tooltip
  const tip = $("#tip");
  const showTip = (x, y, html) => {
    tip.innerHTML = html; tip.classList.add("on");
    const w = tip.offsetWidth, h = tip.offsetHeight;
    tip.style.left = Math.min(x + 14, window.innerWidth - w - 8) + "px";
    tip.style.top = Math.max(8, y - h - 12) + "px";
  };
  const hideTip = () => tip.classList.remove("on");

  // ------------------------------------------------------------------ API
  async function api(path, opts = {}) {
    const headers = { "Content-Type": "application/json" };
    const token = store.get("token", "");
    if (token) headers.Authorization = "Bearer " + token;
    const r = await fetch(path, { ...opts, headers });
    if (r.status === 401 || r.status === 403) {
      setConn("The server needs an API token. Add it in Settings.", true);
      throw new Error("unauthorized");
    }
    if (!r.ok) {
      let detail = r.statusText;
      try { detail = (await r.json()).detail || detail; } catch { /* not json */ }
      throw new Error(detail);
    }
    return r.json();
  }
  const setConn = (msg, bad) => { const c = $("#conn"); c.textContent = msg; c.style.color = bad ? "var(--critical)" : ""; };

  // ------------------------------------------------------------------ state
  const state = { runs: [], filter: "all", query: "", selected: null, detail: null, listTimer: null, runTimer: null };

  // ------------------------------------------------------------------ run list
  async function loadRuns() {
    try {
      const o = await api("/v1/overview");
      state.runs = o.runs;
      setConn(`${o.runs.length} runs · updated ${new Date().toLocaleTimeString()}` +
        (o.pending_findings ? ` · checking ${o.pending_findings} for issues…` : ""));
      renderRunList();
      renderWelcome();
      // come back sooner while findings are still being computed
      clearTimeout(state.listTimer);
      state.listTimer = setTimeout(loadRuns, o.pending_findings ? 3000 : 10000);
    } catch (e) {
      if (e.message !== "unauthorized") setConn("Can't reach the TrainScope server. Retrying…", true);
      clearTimeout(state.listTimer);
      state.listTimer = setTimeout(loadRuns, 5000);
    }
  }

  function badgeFor(r) {
    const f = r.findings;
    if (!f) return `<span class="badge wait" title="checking for issues">…</span>`;
    if (f.critical) return `<span class="badge crit" title="${esc(f.top)}">${f.critical} critical</span>`;
    if (f.warning) return `<span class="badge warn" title="${esc(f.top)}">${f.warning} warning</span>`;
    return `<span class="badge ok">OK</span>`;
  }

  function visibleRuns() {
    const q = state.query.trim().toLowerCase();
    return state.runs.filter((r) => {
      if (state.filter === "issues" && !(r.findings && (r.findings.critical || r.findings.warning))) return false;
      if (state.filter === "running" && !["running", "paused"].includes(r.status)) return false;
      if (state.filter === "crashed" && r.status !== "crashed") return false;
      if (!q) return true;
      const hay = [r.name, r.id, r.project, r.status, r.findings && r.findings.cause, r.findings && causeName(r.findings.cause), r.findings && r.findings.top]
        .join(" ").toLowerCase();
      return hay.includes(q);
    });
  }

  function renderRunList() {
    const list = $("#run-list");
    const runs = visibleRuns();
    $("#side-meta").textContent = `${runs.length} of ${state.runs.length} runs`;
    list.innerHTML = runs.map((r) => {
      const prog = r.total_steps ? `${r.last_step + 1}/${r.total_steps}` : `step ${r.last_step}`;
      const cause = r.findings && r.findings.cause ? ` · ${causeName(r.findings.cause)}` : "";
      return `<li class="run-item${r.id === state.selected ? " sel" : ""}" data-id="${esc(r.id)}" tabindex="0" role="button"
                 aria-label="${esc(r.name)}, ${esc(r.status)}">
        <span class="dot ${esc(r.status)}"></span>
        <span class="nm">${esc(r.name)}</span>
        ${badgeFor(r)}
        <span class="sub">${esc(r.project)} · ${prog} · loss ${num(r.last_loss)}${esc(cause)}</span>
      </li>`;
    }).join("") || `<li class="no-data" style="padding:12px">No runs match.</li>`;
  }

  function renderWelcome() {
    const n = state.runs.length;
    const live = state.runs.filter((r) => ["running", "paused"].includes(r.status)).length;
    const crit = state.runs.filter((r) => r.findings && r.findings.critical).length;
    const warn = state.runs.filter((r) => r.findings && !r.findings.critical && r.findings.warning).length;
    $("#welcome-stats").innerHTML = [
      [n, "runs"], [live, "live now"], [crit, "with critical issues"], [warn, "with warnings"],
    ].map(([v, k]) => `<div class="stat"><b>${v}</b><span>${k}</span></div>`).join("");
  }

  // ------------------------------------------------------------------ charts
  const NS = "http://www.w3.org/2000/svg";
  const svgEl = (tag, attrs, parent) => {
    const n = document.createElementNS(NS, tag);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  };

  /* Line chart with one shared y-axis, finding bands, NaN ticks and a crosshair tooltip. */
  function lineChart(host, cfg) {
    host.innerHTML = "";
    const series = cfg.series.filter((s) => s.points && s.points.length);
    if (!series.length) { host.innerHTML = `<div class="no-data">${esc(cfg.empty || "Not recorded for this run.")}</div>`; return; }
    // draw at the container's real width so text stays at its natural size in wide and narrow panels
    const W = Math.max(320, Math.round(host.clientWidth || 600)), H = cfg.height || 190, padL = 52, padR = 12, padT = 10, padB = 24;
    const xs = series.flatMap((s) => s.points.map((p) => p[0]));
    const ys = series.flatMap((s) => s.points.map((p) => p[1]));
    const xMin = Math.min(...xs, ...(cfg.nonfinite || [])), xMax = Math.max(...xs, ...(cfg.nonfinite || []), xMin + 1);
    let yMin = Math.min(...ys), yMax = Math.max(...ys);
    const positive = yMin > 0;
    const log = cfg.log ?? (positive && yMax / yMin > 50);
    if (cfg.zero && !log) yMin = Math.min(0, yMin);
    if (yMin === yMax) { yMin -= 1; yMax += 1; }
    const ty = log ? Math.log10 : (v) => v;
    const lo = ty(yMin), hi = ty(yMax), span = hi - lo;
    const y0 = lo - span * 0.04, y1 = hi + span * 0.06;
    const X = (v) => padL + ((v - xMin) / (xMax - xMin)) * (W - padL - padR);
    const Y = (v) => padT + (1 - (ty(v) - y0) / (y1 - y0)) * (H - padT - padB);

    const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": cfg.label || "" }, host);
    // finding bands (behind everything)
    for (const b of cfg.bands || []) {
      const a = X(Math.max(xMin, b.start)), z = X(Math.min(xMax, b.end));
      svgEl("rect", { x: a, y: padT, width: Math.max(2, z - a), height: H - padT - padB, class: `band-${b.severity}` }, svg);
    }
    // y grid + labels
    const ticks = [];
    if (log) {
      for (let e = Math.floor(y0); e <= Math.ceil(y1); e++) if (e >= y0 && e <= y1) ticks.push(10 ** e);
    } else {
      const step = niceStep((y1 - y0) / 4);
      for (let v = Math.ceil(y0 / step) * step; v <= y1; v += step) ticks.push(v);
    }
    for (const t of ticks) {
      svgEl("line", { x1: padL, x2: W - padR, y1: Y(t), y2: Y(t), class: "gridline" }, svg);
      const tx = svgEl("text", { x: padL - 6, y: Y(t) + 3.5, "text-anchor": "end" }, svg); tx.textContent = cfg.yFmt ? cfg.yFmt(t) : num(t, 2);
    }
    // x labels
    const xstep = niceStep((xMax - xMin) / Math.max(3, Math.round(W / 120)));
    for (let v = Math.ceil(xMin / xstep) * xstep; v <= xMax; v += xstep) {
      const tx = svgEl("text", { x: X(v), y: H - 6, "text-anchor": "middle" }, svg); tx.textContent = num(v, 4);
    }
    svgEl("line", { x1: padL, x2: W - padR, y1: H - padB, y2: H - padB, class: "axisline" }, svg);
    // NaN / Inf ticks along the top
    for (const s of (cfg.nonfinite || []).slice(0, 40)) {
      svgEl("line", { x1: X(s), x2: X(s), y1: padT, y2: padT + 8, stroke: "var(--critical)", "stroke-width": 2 }, svg);
    }
    // lines
    for (const s of series) {
      const d = s.points.map((p, i) => `${i ? "L" : "M"}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join("");
      svgEl("path", { d, fill: "none", stroke: s.color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round",
        ...(s.dash ? { "stroke-dasharray": "5 4" } : {}) }, svg);
      if (s.points.length <= 40) for (const p of s.points) svgEl("circle", { cx: X(p[0]), cy: Y(p[1]), r: 2.5, fill: s.color }, svg);
    }
    // crosshair
    const cross = svgEl("line", { x1: 0, x2: 0, y1: padT, y2: H - padB, class: "cross", visibility: "hidden" }, svg);
    const hit = svgEl("rect", { x: padL, y: padT, width: W - padL - padR, height: H - padT - padB, fill: "transparent" }, svg);
    const move = (ev) => {
      const r = svg.getBoundingClientRect();
      const sx = ((ev.clientX - r.left) / r.width) * W;
      const step = xMin + ((sx - padL) / (W - padL - padR)) * (xMax - xMin);
      const rows = series.map((s) => {
        const p = nearest(s.points, step);
        return p ? `<span style="color:${s.color}">■</span> ${esc(s.label)}: ${num(p[1], 4)}` : "";
      }).filter(Boolean);
      const band = (cfg.bands || []).find((b) => step >= b.start && step <= b.end);
      cross.setAttribute("x1", sx); cross.setAttribute("x2", sx); cross.setAttribute("visibility", "visible");
      showTip(ev.clientX, ev.clientY, `step ${Math.round(step)}<br>${rows.join("<br>")}${band ? `<br><span style="opacity:.75">${esc(band.label)}</span>` : ""}`);
    };
    hit.addEventListener("pointermove", move);
    hit.addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); hideTip(); });
  }
  const niceStep = (raw) => {
    if (!(raw > 0)) return 1;
    const p = 10 ** Math.floor(Math.log10(raw));
    const m = raw / p;
    return (m < 1.5 ? 1 : m < 3 ? 2 : m < 7 ? 5 : 10) * p;
  };
  const nearest = (pts, x) => {
    let lo = 0, hi = pts.length - 1;
    if (hi < 0) return null;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (pts[mid][0] < x) lo = mid + 1; else hi = mid; }
    const a = pts[Math.max(0, lo - 1)], b = pts[lo];
    return Math.abs(a[0] - x) <= Math.abs(b[0] - x) ? a : b;
  };

  // ------------------------------------------------------------------ run detail
  async function openRun(id, { quiet = false } = {}) {
    state.selected = id;
    renderRunList();
    $("#welcome").hidden = true;
    const view = $("#run-view");
    view.hidden = false;
    if (!quiet) view.innerHTML = `<div class="no-data">Loading run…</div>`;
    $("#sidebar").classList.remove("open");
    try {
      const SERIES = ["loss", "eval/loss", "lr", "grad/global_norm", "grad/carryover_ratio", "train/accuracy", "eval/accuracy"];
      const [summary, findings, series, layers, system, data, commands] = await Promise.all([
        api(`/v1/runs/${id}/summary`), api(`/v1/runs/${id}/findings`),
        api(`/v1/runs/${id}/series?names=${encodeURIComponent(SERIES.join(","))}`),
        api(`/v1/runs/${id}/layers`).catch(() => null), api(`/v1/runs/${id}/system`).catch(() => null),
        api(`/v1/runs/${id}/data`).catch(() => null), api(`/v1/runs/${id}/commands`),
      ]);
      if (state.selected !== id) return; // user moved on
      state.detail = { summary, findings, series, layers, system, data, commands };
      renderRun(id);
    } catch (e) {
      if (state.selected === id) view.innerHTML = `<div class="no-data">Couldn't load this run: ${esc(e.message)}</div>`;
    }
    clearTimeout(state.runTimer);
    const live = state.detail && ["running", "paused"].includes(state.detail.summary.status);
    if (live && store.get("live", true)) state.runTimer = setTimeout(() => state.selected === id && openRun(id, { quiet: true }), 5000);
  }

  function renderRun(id) {
    const { summary: S, findings: F, series, layers: L, system: SY, data: D, commands: C } = state.detail;
    const cfg = S.config || {};
    const model = cfg._model || {};
    const by = Object.fromEntries(series.series.map((s) => [s.name, s]));
    const live = ["running", "paused"].includes(S.status);
    const total = S.progress.total_steps;
    const done = total ? Math.min(1, (S.progress.last_step + 1) / total) : null;
    const env = S.environment || {};
    // shade only findings tied to a window of the run; a band over the whole chart carries no information
    const lastStep = Math.max(1, S.progress.last_step);
    const bands = F.filter((f) => (f.step_end - f.step_start) < 0.6 * lastStep)
      .map((f) => ({ start: f.step_start, end: f.step_end, severity: f.severity === "critical" ? "critical" : "warning", label: `${f.detector}: ${f.summary}` }));

    const configChips = ["task", "model", "optimizer", "lr", "batch_size", "scheduler", "warmup_steps", "grad_clip", "precision", "num_workers", "grad_accum"]
      .filter((k) => cfg[k] !== undefined && cfg[k] !== null && cfg[k] !== "")
      .map((k) => `<span>${esc(k)} <b>${esc(typeof cfg[k] === "number" ? num(cfg[k], 4) : cfg[k])}</b></span>`).join("");

    $("#run-view").innerHTML = `
      <div class="run-head">
        <div class="run-title">
          <h1>${esc(S.name)} <span class="status-pill ${esc(S.status)}">${esc(S.status)}</span></h1>
          <div class="meta"><span>project <b>${esc(S.project)}</b></span><span>id <b class="mono">${esc(S.run_id)}</b></span>
            <span>step <b>${S.progress.last_step}${total ? " / " + total : ""}</b></span>
            ${model.class ? `<span>model <b>${esc(model.class)}</b> · ${num(model.params, 3)} params${model.trainable_params !== undefined && model.trainable_params !== model.params ? ` (<b>${num(model.trainable_params, 3)}</b> trainable)` : ""}</span>` : ""}
            ${env.device ? `<span>device <b>${esc(env.device)}</b></span>` : ""}${env.torch ? `<span>torch <b>${esc(env.torch)}</b></span>` : ""}</div>
          ${done !== null ? `<div class="progress" title="${Math.round(done * 100)}% of configured steps"><i style="width:${(done * 100).toFixed(1)}%"></i></div>` : ""}
          <div class="meta">${configChips}</div>
        </div>
        <div class="actions">
          <button class="ghost" data-act="checkpoint" ${live ? "" : "disabled"} title="Save a checkpoint at the next step">Checkpoint</button>
          ${S.status === "paused"
            ? `<button class="ghost" data-act="resume">Resume</button>`
            : `<button class="ghost" data-act="pause" ${live ? "" : "disabled"}>Pause</button>`}
          <button class="ghost" data-act="set_lr" ${live ? "" : "disabled"}>Change LR</button>
        </div>
      </div>

      <div class="section-title">Findings</div>
      <div class="findings">${F.length ? F.map(findingCard).join("") : `<div class="all-clear">No issues detected. Every detector passed on the data recorded so far.</div>`}</div>

      <div class="section-title">Training curves</div>
      <div class="grid">
        <div class="panel"><div class="panel-h"><h2>Loss</h2><span class="s">${esc(lossNote(by))}</span></div>
          <div class="legend"><span><i style="background:var(--series-1)"></i>train</span>${by["eval/loss"] && by["eval/loss"].points.length ? `<span style="color:var(--series-2)"><i class="dash" style="background:var(--series-2)"></i>eval</span>` : ""}${(by.loss && by.loss.nonfinite.length) ? `<span><i style="background:var(--critical)"></i>NaN/Inf</span>` : ""}</div>
          <div class="chart" id="c-loss"></div></div>
        <div class="panel"><div class="panel-h"><h2>Learning rate</h2><span class="s">${esc(cfg.scheduler ? `${cfg.scheduler} schedule, ${cfg.warmup_steps || 0} warmup steps` : "")}</span></div><div class="chart" id="c-lr"></div></div>
        <div class="panel"><div class="panel-h"><h2>Gradient norm</h2><span class="s">measured before clipping</span></div><div class="chart" id="c-grad"></div></div>
        <div class="panel"><div class="panel-h"><h2>Gradient carry-over</h2><span class="s">.grad ÷ latest backward · healthy = 1${cfg.grad_accum > 1 ? ` (up to ${cfg.grad_accum} with accumulation)` : ""}</span></div><div class="chart" id="c-carry"></div></div>
        ${(by["train/accuracy"] && by["train/accuracy"].points.length) || (by["eval/accuracy"] && by["eval/accuracy"].points.length) ? `
        <div class="panel wide"><div class="panel-h"><h2>Accuracy</h2><span class="s">train (running) vs eval</span></div>
          <div class="legend"><span><i style="background:var(--series-1)"></i>train</span><span style="color:var(--series-2)"><i class="dash" style="background:var(--series-2)"></i>eval</span></div>
          <div class="chart" id="c-acc"></div></div>` : ""}
      </div>

      <div class="section-title">Layers</div>
      <div class="panel wide" id="p-layers"></div>

      <div class="section-title">System and data</div>
      <div class="grid">
        <div class="panel" id="p-system"></div>
        <div class="panel" id="p-data"></div>
      </div>

      <div class="section-title">Control</div>
      <div class="panel wide" id="p-cmds"></div>`;

    const s = (name, label, color, dash) => ({ label, color, dash, points: (by[name] || {}).points || [] });
    lineChart($("#c-loss"), { series: [s("loss", "train", "var(--series-1)"), s("eval/loss", "eval", "var(--series-2)", true)], bands, nonfinite: (by.loss || {}).nonfinite, label: "Loss over steps" });
    lineChart($("#c-lr"), { series: [s("lr", "lr", "var(--series-1)")], zero: true, log: false, label: "Learning rate over steps" });
    lineChart($("#c-grad"), { series: [s("grad/global_norm", "global norm", "var(--series-1)")], bands, label: "Gradient norm over steps" });
    lineChart($("#c-carry"), { series: [s("grad/carryover_ratio", "carry-over", "var(--series-1)")], bands, log: false, zero: true, empty: "Not recorded (needs the SDK's gradient hooks, torch ≥ 2.1).", label: "Gradient carry-over ratio" });
    if ($("#c-acc")) lineChart($("#c-acc"), { series: [s("train/accuracy", "train", "var(--series-1)"), s("eval/accuracy", "eval", "var(--series-2)", true)], zero: true, log: false, yFmt: (v) => Math.round(v * 100) + "%", label: "Accuracy over steps" });

    renderLayers(L);
    renderSystem(SY);
    renderData(D, F);
    renderCommands(C, live);
  }

  function lossNote(by) {
    const p = (by.loss || {}).points || [];
    if (!p.length) return "";
    const nf = (by.loss || {}).nonfinite || [];
    return `${num(p[0][1])} → ${num(p[p.length - 1][1])}` + (nf.length ? ` · NaN/Inf from step ${nf[0]}` : "");
  }

  function findingCard(f) {
    const ev = { ...(f.evidence || {}) };
    const causes = ev.suspected_causes || [];
    delete ev.suspected_causes;
    const conf = ev.confidence; delete ev.confidence;
    const range = f.step_start === f.step_end ? `step ${f.step_start}` : `steps ${f.step_start}–${f.step_end}`;
    return `<div class="finding ${esc(f.severity)}"><span class="stripe"></span><div class="body">
      <div class="row1"><span class="sev ${esc(f.severity)}">${esc(f.severity)}</span><span class="det">${esc(f.detector)}</span><span class="steps">${range}</span></div>
      <div class="sum">${esc(f.summary)}</div>
      ${causes.length ? `<div class="causes">likely cause ${causes.map((c, i) => `<span class="cause${i ? "" : " first"}">${esc(causeName(c))}</span>`).join("")}${conf !== undefined ? ` · confidence ${Math.round(conf * 100)}%` : ""}</div>` : ""}
      <details class="ev"><summary>Evidence</summary><pre>${esc(JSON.stringify(ev, null, 2))}</pre></details>
    </div></div>`;
  }

  function renderLayers(L) {
    const p = $("#p-layers");
    if (!L || !L.layers || !L.layers.length) {
      p.innerHTML = `<div class="panel-h"><h2>Per-layer statistics</h2></div><div class="no-data">No per-layer stats. Call <code>ts.watch(model)</code> to record them.</div>`;
      return;
    }
    const outl = new Set((L.outliers || []).map((o) => o.layer + "|" + o.metric));
    const cols = [
      ["grad/norm", "grad norm", true], ["weight/update_ratio", "update ratio", true], ["weight/norm", "weight norm", true],
      ["act/dead_frac", "dead units", false], ["grad/zero_frac", "zero grads", false],
    ].filter(([k]) => L.layers.some((r) => r[k] !== undefined));
    const ranges = Object.fromEntries(cols.map(([k, , lg]) => {
      const v = L.layers.map((r) => r[k]).filter((x) => typeof x === "number" && (!lg || x > 0));
      return [k, { min: Math.min(...v), max: Math.max(...v), log: lg }];
    }));
    const bar = (k, v) => {
      if (typeof v !== "number") return `<span class="flag">${esc(v ?? "–")}</span>`;
      const r = ranges[k];
      let f = 0;
      if (r.log) f = v > 0 && r.max > r.min ? (Math.log10(v) - Math.log10(r.min)) / (Math.log10(r.max) - Math.log10(r.min)) : (v > 0 ? 1 : 0);
      else f = r.max > 0 ? v / Math.max(r.max, k.endsWith("_frac") ? 1 : r.max) : 0;
      const shown = k.endsWith("_frac") ? pct(v) : num(v);
      return `<div class="cellbar"><span>${shown}</span><span class="b"><i style="width:${Math.max(2, f * 100).toFixed(1)}%"></i></span></div>`;
    };
    const frozen = L.layers.filter((r) => r["weight/trainable"] === 0).length;
    p.innerHTML = `<div class="panel-h"><h2>Per-layer statistics at step ${L.step}</h2>
        <span class="s">${L.n_layers} layers · ${(L.outliers || []).length} outlier value(s)${frozen ? ` · ${frozen} frozen` : ""} · sampled at ${L.sampled_steps ? L.sampled_steps[2] : "?"} steps</span></div>
      <div class="tbl-wrap"><table class="layers"><thead><tr><th>Layer</th>${cols.map(([, n]) => `<th>${n}</th>`).join("")}<th></th></tr></thead><tbody>
      ${L.layers.map((r) => {
        const flags = cols.filter(([k]) => outl.has(r.layer + "|" + k)).map(([, n]) => n);
        const isFrozen = r["weight/trainable"] === 0;
        return `<tr class="${flags.length || isFrozen ? "out" : ""}"><td class="ln" title="${esc(r.layer)}">${esc(r.layer)}</td>
          ${cols.map(([k]) => `<td>${outl.has(r.layer + "|" + k) ? bar(k, r[k]).replace("cellbar", "cellbar alert") : bar(k, r[k])}</td>`).join("")}
          <td class="flag">${isFrozen ? "frozen" : flags.length ? "outlier: " + esc(flags.join(", ")) : ""}</td></tr>`;
      }).join("")}</tbody></table></div>`;
  }

  function renderSystem(SY) {
    const p = $("#p-system");
    if (!SY) { p.innerHTML = `<div class="panel-h"><h2>System</h2></div><div class="no-data">No system metrics recorded.</div>`; return; }
    const mem = (SY.series || []).find((s) => s.name.startsWith("sys/"));
    const wait = SY.data_wait_frac, st = SY.step_time;
    const memKey = mem ? mem.name : null, m = memKey ? SY[memKey] : null;
    p.innerHTML = `<div class="panel-h"><h2>System</h2><span class="s">${esc(SY.summary || "")}</span></div>
      <dl class="kv">
        ${wait ? `<dt>waiting on data</dt><dd>${pct(wait.median)} of each step (median)${wait.median > 0.3 ? ' <span class="flag">· loader is the bottleneck</span>' : ""}</dd>` : ""}
        ${st ? `<dt>step time</dt><dd>${num(st.median * 1000)} ms median</dd>` : ""}
        ${m ? `<dt>memory floor</dt><dd>${num(m.floor_start)} → ${num(m.floor_end)} MB ${m.floor_trend === "rising" ? '<span class="flag">· rising</span>' : "· flat"}</dd>` : ""}
      </dl>
      ${mem ? `<div class="legend"><span><i style="background:var(--series-1)"></i>${esc(memKey.replace("sys/", ""))}</span><span><i style="background:var(--series-2)"></i>floor (rolling minimum)</span></div><div class="chart" id="c-mem"></div>` : ""}`;
    if (mem) lineChart($("#c-mem"), { series: [{ label: "memory MB", color: "var(--series-1)", points: mem.points }, { label: "floor", color: "var(--series-2)", points: mem.floor || [] }], log: false, height: 150, label: "Memory over steps" });
  }

  function renderData(D, F) {
    const p = $("#p-data");
    if (!D) { p.innerHTML = `<div class="panel-h"><h2>Data</h2></div><div class="no-data">No data statistics recorded.</div>`; return; }
    const overlap = F.find((f) => f.detector === "split_overlap");
    const first = D.label_distribution_first_half || {}, second = D.label_distribution_second_half || {};
    const labels = Object.keys(first).sort((a, b) => Number(a) - Number(b)).slice(0, 12);
    const max = Math.max(0.01, ...labels.flatMap((k) => [first[k] || 0, second[k] || 0]));
    const stat = (k, label, fmt = num) => D[k] ? `<dt>${label}</dt><dd>${fmt(D[k].median)} median (range ${fmt(D[k].min)}–${fmt(D[k].max)})</dd>` : "";
    p.innerHTML = `<div class="panel-h"><h2>Data</h2><span class="s">${esc(D.summary || "")}</span></div>
      <dl class="kv">
        ${stat("data/label_entropy", "label balance")}
        ${stat("data/majority_frac", "largest class", pct)}
        ${stat("data/pad_ratio", "padding", pct)}
        <dt>train / eval overlap</dt><dd>${overlap ? `<span class="flag">${esc(overlap.summary)}</span>` : (D.sample_ids ? "none found" : "not recorded (<code>ts.log_dataset_ids</code>)")}</dd>
      </dl>
      ${labels.length ? `<div class="legend"><span><i style="background:var(--series-1)"></i>first half of run</span><span><i style="background:var(--series-2)"></i>second half</span></div>
      <div class="bars">${labels.map((k) => `<div class="bar-row" data-tip="label ${esc(k)}: ${pct(first[k] || 0)} → ${pct(second[k] || 0)}"><span>class ${esc(k)}</span>
        <span class="b"><i class="first" style="width:${((first[k] || 0) / max * 100).toFixed(1)}%"></i><i class="second" style="width:${((second[k] || 0) / max * 100).toFixed(1)}%"></i></span>
        <span>${pct(second[k] || 0)}</span></div>`).join("")}</div>` : ""}`;
    p.querySelectorAll("[data-tip]").forEach((n) => {
      n.addEventListener("pointermove", (e) => showTip(e.clientX, e.clientY, esc(n.dataset.tip)));
      n.addEventListener("pointerleave", hideTip);
    });
  }

  function renderCommands(C, live) {
    const p = $("#p-cmds");
    const rows = [...C].reverse();
    const describe = (c) => c.action === "set_lr" ? `set LR to ${num(c.args && c.args.lr, 4)}` : c.action;
    p.innerHTML = `<div class="panel-h"><h2>Commands</h2><span class="s">checkpoints run at once · pause, resume and LR changes need a person to approve · applied at the next step boundary</span></div>
      ${rows.length ? `<div class="tbl-wrap"><table class="cmds"><thead><tr><th>Action</th><th>Status</th><th>Requested by</th><th>Approved by</th><th>When</th><th></th></tr></thead><tbody>
      ${rows.map((c) => `<tr><td>${esc(describe(c))}</td><td><span class="cmd-status ${esc(c.status)}">${esc(c.status.replace("_", " "))}</span></td>
        <td class="mono">${esc(c.requested_by || "–")}</td><td class="mono">${esc(c.approved_by || "–")}</td><td>${ago(c.created_at)}</td>
        <td>${c.status === "pending_approval" ? `<div class="cmd-btns"><button class="approve" data-approve="${esc(c.id)}">Approve</button><button class="reject" data-reject="${esc(c.id)}">Reject</button></div>`
          : c.result && Object.keys(c.result).length ? `<span class="note mono">${esc(JSON.stringify(c.result)).slice(0, 80)}</span>` : ""}</td></tr>`).join("")}
      </tbody></table></div>` : `<div class="no-data">${live ? "No commands yet. Use the buttons at the top to checkpoint, pause or change the learning rate." : "No commands were sent to this run."}</div>`}`;
  }

  // ------------------------------------------------------------------ actions
  function who() {
    const name = (store.get("name", "") || "").trim();
    if (!name) { openSettings(); toast("Add your name first, so approvals record who made them."); return null; }
    return name;
  }

  async function requestAction(action, args = {}) {
    const name = who(); if (!name) return;
    const id = state.selected;
    try {
      const cmd = await api(`/v1/runs/${id}/commands`, { method: "POST", body: JSON.stringify({ action, args, requested_by: "human:" + name }) });
      if (cmd.status === "pending_approval") {
        // a person asked for it directly, so their request is also the approval
        await api(`/v1/commands/${cmd.id}/approve`, { method: "POST", body: JSON.stringify({ approved_by: "human:" + name, approve: true }) });
      }
      toast(`${action === "set_lr" ? "LR change" : action[0].toUpperCase() + action.slice(1)} queued. The run applies it at the next step.`);
      openRun(id, { quiet: true });
    } catch (e) { toast(`Couldn't queue ${action}: ${e.message}`); }
  }

  async function decide(cmdId, approve) {
    const name = who(); if (!name) return;
    try {
      await api(`/v1/commands/${cmdId}/approve`, { method: "POST", body: JSON.stringify({ approved_by: "human:" + name, approve }) });
      toast(approve ? "Approved. The run applies it at the next step." : "Rejected.");
      openRun(state.selected, { quiet: true });
    } catch (e) { toast(`Couldn't ${approve ? "approve" : "reject"}: ${e.message}`); }
  }

  // ------------------------------------------------------------------ settings + theme
  function openSettings() {
    $("#set-name").value = store.get("name", "");
    $("#set-token").value = store.get("token", "");
    $("#set-live").checked = store.get("live", true);
    $("#settings").showModal();
  }
  $("#settings").addEventListener("close", () => {
    if ($("#settings").returnValue !== "save") return;
    store.set("name", $("#set-name").value.trim());
    store.set("token", $("#set-token").value.trim());
    store.set("live", $("#set-live").checked);
    toast("Settings saved.");
    loadRuns();
  });
  const applyTheme = (t) => { if (t) document.documentElement.setAttribute("data-theme", t); else document.documentElement.removeAttribute("data-theme"); };
  applyTheme(store.get("theme", null));
  $("#btn-theme").addEventListener("click", () => {
    const dark = matchMedia("(prefers-color-scheme: dark)").matches;
    const cur = document.documentElement.getAttribute("data-theme") || (dark ? "dark" : "light");
    const next = cur === "dark" ? "light" : "dark";
    applyTheme(next); store.set("theme", next);
    if (state.detail && state.selected) renderRun(state.selected);
  });
  $("#btn-settings").addEventListener("click", openSettings);
  $("#btn-menu").addEventListener("click", () => $("#sidebar").classList.toggle("open"));

  // ------------------------------------------------------------------ events + routing
  $("#search").addEventListener("input", (e) => { state.query = e.target.value; renderRunList(); });
  document.querySelectorAll(".filters .chip").forEach((b) => b.addEventListener("click", () => {
    document.querySelectorAll(".filters .chip").forEach((x) => x.classList.toggle("on", x === b));
    state.filter = b.dataset.filter; renderRunList();
  }));
  const pick = (el) => { const id = el.dataset.id; if (id) location.hash = "run=" + id; };
  $("#run-list").addEventListener("click", (e) => { const li = e.target.closest(".run-item"); if (li) pick(li); });
  $("#run-list").addEventListener("keydown", (e) => { if ((e.key === "Enter" || e.key === " ") && e.target.classList.contains("run-item")) { e.preventDefault(); pick(e.target); } });
  $("#run-view").addEventListener("click", (e) => {
    const act = e.target.closest("[data-act]"); const ap = e.target.closest("[data-approve]"); const rj = e.target.closest("[data-reject]");
    if (act && !act.disabled) {
      if (act.dataset.act === "set_lr") { $("#lr-value").value = ""; $("#lr-dialog").showModal(); }
      else requestAction(act.dataset.act);
    }
    if (ap) decide(ap.dataset.approve, true);
    if (rj) decide(rj.dataset.reject, false);
  });
  $("#lr-dialog").addEventListener("close", () => {
    if ($("#lr-dialog").returnValue !== "ok") return;
    const v = parseFloat($("#lr-value").value);
    if (!(v > 0 && v < 10)) { toast("Enter a learning rate between 0 and 10."); return; }
    requestAction("set_lr", { lr: v });
  });
  const route = () => {
    const m = /^#run=(.+)$/.exec(location.hash);
    if (m) openRun(decodeURIComponent(m[1]));
    else { state.selected = null; clearTimeout(state.runTimer); $("#run-view").hidden = true; $("#welcome").hidden = false; renderRunList(); }
  };
  window.addEventListener("hashchange", route);
  let resizeTimer;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => { if (state.detail && state.selected) renderRun(state.selected); }, 200);
  });

  loadRuns().then(route);
})();
