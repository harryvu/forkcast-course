/* Lesson 8 widgets: where the data lives during one step, what half precision keeps and loses,
   what a checkpoint must contain, and what augmentation does to a picture. */
(function () {
  "use strict";
  const FC = window.FC;
  const S = FC.svgEl;
  function rng(seed) { return function () { seed |= 0; seed = (seed + 0x6D2B79F5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }

  /* ---------- widget: one training step across the two devices ---------- */
  FC.widgets.deviceFlow = function (el) {
    const stages = [
      { id: "load", side: "cpu", label: "DataLoader builds a batch", note: "Workers read items and collate them on the CPU. With pin_memory=True the batch sits in page-locked memory, which the GPU can pull from directly." },
      { id: "move", side: "edge", label: "inputs.to(device, non_blocking=True)", note: "The copy across the PCIe bus. Tensors are copied: x.to(device) returns a new tensor and leaves x where it was. Modules are moved in place: model.to(device) changes the model itself." },
      { id: "fwd", side: "gpu", label: "with torch.autocast(...): output = model(*inputs)", note: "Thousands of cores run the matrix multiplies in float16 where it is safe. The model's parameters must already be on the GPU, or this is where the device mismatch error appears." },
      { id: "loss", side: "gpu", label: "loss = loss_fn(output.float(), target)", note: "The loss is computed in float32 on the GPU. Everything so far is asynchronous: the CPU has queued the work and moved on." },
      { id: "bwd", side: "gpu", label: "scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()", note: "Backward and the update stay on the GPU. The scaler multiplies the loss up before backward so tiny float16 gradients do not vanish, then unscales before the step and skips the step if anything overflowed." },
      { id: "item", side: "edge", label: "total += loss.item() * len(target)", note: ".item() copies one number back to the CPU and waits for the GPU to finish. It is the synchronization point: necessary once per step, ruinous inside an inner loop." },
      { id: "log", side: "cpu", label: "logging, checkpointing, printing", note: "Back on the CPU: TensorBoard, the history list, a checkpoint at the end of the epoch. The GPU is already working on the next batch." },
    ];
    let i = 0;
    el.innerHTML = `
      <div class="widget-head"><h3>One training step, two devices</h3><div class="presets btn-row"><button type="button" class="chip" data-next>next</button><button type="button" class="chip" data-reset>reset</button></div></div>
      <div class="widget-body"><svg role="img" aria-label="the stages of one training step on CPU and GPU"></svg><p class="readout" data-readout></p><p class="muted" data-note style="max-width:68ch"></p></div>`;
    const svg = el.querySelector("svg"), readout = el.querySelector("[data-readout]"), note = el.querySelector("[data-note]");
    function draw() {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      svg.setAttribute("viewBox", "0 0 760 300");
      svg.appendChild(S("rect", { x: 10, y: 30, width: 230, height: 260, rx: 8, style: "fill: var(--line-2); stroke: var(--line)" }));
      svg.appendChild(S("rect", { x: 420, y: 30, width: 330, height: 260, rx: 8, style: "fill: var(--dim-n); fill-opacity: .08; stroke: var(--dim-n)" }));
      svg.appendChild(S("text", { x: 20, y: 20, class: "axis-label", style: "fill: var(--ink-2)" }, "CPU: data, bookkeeping"));
      svg.appendChild(S("text", { x: 430, y: 20, class: "axis-label", style: "fill: var(--dim-n)" }, "GPU: the model and its parameters"));
      const pos = { load: [30, 60], move: [250, 60], fwd: [440, 60], loss: [440, 120], bwd: [440, 180], item: [250, 240], log: [30, 240] };
      stages.forEach((s, k) => {
        const [x, y] = pos[s.id], cur = k === i, w = s.side === "gpu" ? 290 : s.side === "edge" ? 160 : 190;
        svg.appendChild(S("rect", { x, y, width: w, height: 40, rx: 6, style: `fill: var(--panel); stroke: ${cur ? "var(--dim-y)" : s.side === "gpu" ? "var(--dim-n)" : "var(--ink-3)"}; stroke-width: ${cur ? 3 : 1.2}` }));
        svg.appendChild(S("text", { x: x + 10, y: y + 24, class: "cell-text", style: `fill: var(--ink); font-size: ${s.side === "gpu" ? 10 : 10.5}px` }, s.label.length > 46 ? s.label.slice(0, 44) + "…" : s.label));
      });
      const arrow = (x1, y1, x2, y2, dashed) => svg.appendChild(S("line", { x1, y1, x2, y2, style: `stroke: var(--ink-3); stroke-width: 1.5; ${dashed ? "stroke-dasharray: 5 4" : ""}` }));
      arrow(220, 80, 250, 80); arrow(410, 80, 440, 80); arrow(585, 100, 585, 120); arrow(585, 160, 585, 180); arrow(440, 220, 410, 260, true); arrow(250, 260, 220, 260);
      readout.innerHTML = `step ${i + 1} of ${stages.length}: <b>${stages[i].label}</b>`;
      note.textContent = stages[i].note;
    }
    el.querySelector("[data-next]").addEventListener("click", () => { i = (i + 1) % stages.length; draw(); });
    el.querySelector("[data-reset]").addEventListener("click", () => { i = 0; draw(); });
    draw();
  };

  /* ---------- widget: what float16 and bfloat16 keep ---------- */
  function toFloat16(x) {
    if (!Number.isFinite(x) || x === 0) return x;
    const sign = x < 0 ? -1 : 1; x = Math.abs(x);
    if (x >= 65520) return sign * Infinity;
    const e = Math.floor(Math.log2(x));
    if (e < -14) { const step = Math.pow(2, -24); return sign * Math.round(x / step) * step; }
    const step = Math.pow(2, e - 10), v = Math.round(x / step) * step;
    return v >= 65520 ? sign * Infinity : sign * v;
  }
  function toBFloat16(x) {
    if (!Number.isFinite(x) || x === 0) return x;
    const sign = x < 0 ? -1 : 1; x = Math.abs(x);
    const e = Math.floor(Math.log2(x)), step = Math.pow(2, e - 7);
    return sign * Math.round(x / step) * step;
  }
  FC.widgets.precisionDemo = function (el) {
    const samples = [["a weight", 0.1], ["a logit", 3.14159], ["a loss, early", 1234.5678], ["the biggest float16", 65504], ["a loss, exploding", 70000], ["a small gradient", 1e-5], ["a tiny gradient", 1e-8]];
    let scale = 1;
    el.innerHTML = `
      <div class="widget-head"><h3>Three ways to store a number</h3></div>
      <div class="widget-body">
        <table class="bc-table" data-table style="font-size:.9rem"></table>
        <div class="widget-controls" style="margin-top:1rem"><label>GradScaler factor <select data-scale><option value="1">1 (no scaler)</option><option value="1024">1024</option><option value="65536">65536 (the default start)</option></select></label><span class="muted">scales the gradients before they are stored in float16, then divides them back</span></div>
        <p class="readout" data-readout></p>
        <p class="muted" style="max-width:68ch">float32 spends 8 bits on range and 23 on precision. float16 cuts both: 5 and 10, so it holds nothing above 65504 and nothing below about 6e-8, which is exactly where small gradients live. bfloat16 keeps float32's 8 range bits and spends only 7 on precision: coarse, but it never overflows a loss or underflows a gradient, which is why no scaler is needed with it.</p>
      </div>`;
    const table = el.querySelector("[data-table]"), readout = el.querySelector("[data-readout]");
    const fmt = (v) => (v === Infinity ? "inf" : v === 0 ? "0" : Math.abs(v) < 1e-3 || Math.abs(v) >= 1e6 ? v.toExponential(3) : v.toPrecision(7).replace(/\.?0+$/, ""));
    function draw() {
      table.innerHTML = `<tr><th>value</th><th>float32</th><th>float16</th><th>bfloat16</th></tr>` + samples.map(([name, v]) => {
        const isGrad = /gradient/.test(name), x = isGrad ? v * scale : v;
        const f16 = toFloat16(x), b16 = toBFloat16(x);
        const bad16 = f16 === 0 || f16 === Infinity, badb = b16 === 0 || b16 === Infinity;
        return `<tr><th>${name}${isGrad && scale > 1 ? ` × ${scale}` : ""}<br><span class="muted">${fmt(x)}</span></th><td>${fmt(Math.fround(x))}</td><td class="${bad16 ? "clash" : Math.abs(f16 - x) / Math.abs(x) > 1e-4 ? "stretch" : ""}">${fmt(f16)}</td><td class="${badb ? "clash" : Math.abs(b16 - x) / Math.abs(x) > 1e-4 ? "stretch" : ""}">${fmt(b16)}</td></tr>`;
      }).join("");
      readout.innerHTML = scale === 1
        ? `<span class="c"># red: the number is gone (zero or inf). Amber: rounded. The tiny gradient vanishes in float16; 70000 overflows.</span>`
        : `<span class="c"># scaled by ${scale}: the small gradients now survive float16 and are divided back before the optimizer step. Overflow would make the scaler skip the step and halve its factor.</span>`;
    }
    el.querySelector("[data-scale]").addEventListener("change", (e) => { scale = +e.target.value; draw(); });
    draw();
  };

  /* ---------- widget: checkpoints and resume ---------- */
  FC.widgets.checkpointTimeline = function (el) {
    const EPOCHS = 12, CRASH = 7;
    const q = rng(8); const noise = Array.from({ length: EPOCHS }, () => (q() - 0.5) * 0.08);
    const base = (e) => 2.2 * Math.pow(0.78, e) + 0.35 + noise[e];
    let mode = "full";
    el.innerHTML = `
      <div class="widget-head"><h3>A crash at epoch 7</h3><div class="presets btn-row"><button type="button" class="chip" data-mode="full" aria-pressed="true">resume from a full checkpoint</button><button type="button" class="chip" data-mode="weights">resume with weights only</button><button type="button" class="chip" data-mode="none">no checkpoint</button></div></div>
      <div class="widget-body"><svg role="img" aria-label="validation loss per epoch, interrupted at epoch 7 and resumed"></svg><p class="readout" data-readout></p><p class="muted" data-note style="max-width:68ch"></p></div>`;
    const svg = el.querySelector("svg"), readout = el.querySelector("[data-readout]"), note = el.querySelector("[data-note]");
    function draw() {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      const W = 640, H = 260, L = 44, R = 12, T = 14, B = 34;
      svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
      const sx = (e) => L + (e / (EPOCHS - 1)) * (W - L - R), sy = (v) => T + (1 - v / 2.8) * (H - T - B);
      [0.5, 1, 1.5, 2, 2.5].forEach((v) => { svg.appendChild(S("line", { x1: L, y1: sy(v), x2: W - R, y2: sy(v), style: "stroke: var(--line-2)" })); svg.appendChild(S("text", { x: L - 6, y: sy(v) + 4, "text-anchor": "end", class: "cell-text", style: "fill: var(--ink-3)" }, String(v))); });
      for (let e = 0; e < EPOCHS; e++) svg.appendChild(S("text", { x: sx(e), y: H - B + 16, "text-anchor": "middle", class: "cell-text", style: "fill: var(--ink-3)" }, String(e)));
      svg.appendChild(S("text", { x: (L + W) / 2, y: H - 4, "text-anchor": "middle", class: "cell-text", style: "fill: var(--ink-2)" }, "epoch (a checkpoint is written after each one)"));
      const pts = []; for (let e = 0; e < EPOCHS; e++) pts.push([e, base(e)]);
      const path = (arr, style) => svg.appendChild(S("path", { d: arr.map((p, i) => (i ? "L" : "M") + sx(p[0]) + " " + sy(p[1])).join(" "), fill: "none", style }));
      path(pts, "stroke: var(--ink-3); stroke-dasharray: 4 4; stroke-width: 1.5");
      path(pts.slice(0, CRASH), "stroke: var(--ink); stroke-width: 2.5");
      for (let e = 0; e < CRASH; e++) svg.appendChild(S("circle", { cx: sx(e), cy: sy(base(e)), r: 4, style: "fill: var(--dim-w)" }));
      svg.appendChild(S("line", { x1: sx(CRASH) - 8, y1: sy(2.6), x2: sx(CRASH) - 8, y2: sy(0), style: "stroke: var(--bad); stroke-dasharray: 3 3" }));
      svg.appendChild(S("text", { x: sx(CRASH) - 14, y: sy(2.6) + 4, "text-anchor": "end", class: "cell-text", style: "fill: var(--bad)" }, "runtime disconnected during epoch 7"));
      if (mode === "full") path(pts.slice(CRASH), "stroke: var(--dim-y); stroke-width: 2.5");
      if (mode === "weights") { const bumped = pts.slice(CRASH).map(([e, v], k) => [e, v + 0.55 * Math.pow(0.6, k)]); path(bumped, "stroke: var(--dim-y); stroke-width: 2.5"); }
      readout.innerHTML = mode === "full"
        ? `ckpt = torch.load("ckpt.pt")  <span class="c"># keys: model, optimizer, epoch, history, best_val</span><br>start = ckpt["epoch"] + 1   <span class="c"># 7: the last finished epoch was 6</span>`
        : mode === "weights" ? `model.load_state_dict(torch.load("weights_only.pt"))   <span class="c"># a fresh Adam, no running averages, no history</span>` : `<span class="c"># nothing saved: start from epoch 0 and pay for seven epochs again</span>`;
      note.textContent = mode === "full"
        ? "Model weights, the optimizer's running averages, the epoch counter and the history: with all four restored, and the shuffle seeded by epoch number, the resumed run (magenta) retraces the uninterrupted one (dotted) exactly."
        : mode === "weights" ? "With only the weights restored, Adam starts from scratch: its first steps are too large and the loss jumps before settling again. The run recovers, but it is a different run, and the history and best epoch are lost." : "Colab runtimes disconnect; long trainings get interrupted. A checkpoint per epoch costs a few megabytes and a fraction of a second.";
    }
    el.querySelectorAll("[data-mode]").forEach((b) => b.addEventListener("click", () => { mode = b.getAttribute("data-mode"); el.querySelectorAll("[data-mode]").forEach((c) => c.setAttribute("aria-pressed", String(c === b))); draw(); }));
    draw();
  };

  /* ---------- widget: augmentation preview ---------- */
  FC.widgets.augmentPreview = function (el) {
    const N = 14;   // a 14x14 pixel plate with a dish and a fork, drawn in code
    const base = [];
    for (let y = 0; y < N; y++) { const row = []; for (let x = 0; x < N; x++) {
      const dx = x - 6.5, dy = y - 6.5, r = Math.sqrt(dx * dx + dy * dy);
      let c = [0.93, 0.93, 0.9];
      if (r < 5.5) c = [0.96, 0.96, 0.94];
      if (r < 3.4) c = [0.85, 0.45, 0.2];
      if (r < 1.6) c = [0.3, 0.6, 0.25];
      if (x === 12 && y >= 3 && y <= 10) c = [0.55, 0.55, 0.6];
      if (y === 3 && x >= 11 && x <= 13) c = [0.55, 0.55, 0.6];
      row.push(c);
    } base.push(row); }
    let seed = 3;
    el.innerHTML = `<div class="widget-head"><h3>Four random views of one dish</h3><div class="presets btn-row"><button type="button" class="chip" data-roll>re-roll</button></div></div>
      <div class="widget-body"><svg role="img" aria-label="the original picture and four augmented versions"></svg>
      <p class="muted" style="max-width:68ch;margin:.4rem 0 0">RandomHorizontalFlip, RandomResizedCrop(scale 0.7 to 1) and ColorJitter, as in <code>train_transform</code>. The label is still "the same dish". The model never sees the identical tensor twice, so it cannot memorise pixels and must learn what makes a dish a dish: regularization made from data. Never augment the validation set; you want to measure on the real pictures.</p></div>`;
    const svg = el.querySelector("svg");
    function render(img, x0, y0, cell, title) {
      svg.appendChild(S("text", { x: x0, y: y0 - 6, class: "cell-text", style: "fill: var(--ink-2)" }, title));
      for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) { const c = img[y][x]; svg.appendChild(S("rect", { x: x0 + x * cell, y: y0 + y * cell, width: cell, height: cell, fill: `rgb(${Math.round(255 * c[0])},${Math.round(255 * c[1])},${Math.round(255 * c[2])})` })); }
    }
    function augment(q) {
      const flip = q() < 0.5, scale = 0.7 + 0.3 * q(), size = Math.max(6, Math.round(N * scale)), ox = Math.floor(q() * (N - size + 1)), oy = Math.floor(q() * (N - size + 1)), bright = 0.85 + 0.3 * q(), contrast = 0.85 + 0.3 * q();
      const out = [];
      for (let y = 0; y < N; y++) { const row = []; for (let x = 0; x < N; x++) {
        const sx = ox + Math.floor((x / N) * size), sy = oy + Math.floor((y / N) * size);
        const c = base[sy][flip ? N - 1 - sx : sx].map((v) => Math.min(1, Math.max(0, ((v - 0.5) * contrast + 0.5) * bright)));
        row.push(c);
      } out.push(row); }
      return { out, desc: `${flip ? "flipped" : "unflipped"}, crop ${Math.round(scale * 100)}%, brightness ×${bright.toFixed(2)}` };
    }
    function draw() {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      svg.setAttribute("viewBox", "0 0 760 190");
      render(base, 10, 30, 10, "original");
      const q = rng(seed);
      for (let k = 0; k < 4; k++) { const a = augment(q); render(a.out, 190 + k * 142, 30, 10, `view ${k + 1}`); svg.appendChild(S("text", { x: 190 + k * 142, y: 184, class: "cell-text", style: "fill: var(--ink-3); font-size: 9px" }, a.desc)); }
    }
    el.querySelector("[data-roll]").addEventListener("click", () => { seed += 1; draw(); });
    draw();
  };
})();
