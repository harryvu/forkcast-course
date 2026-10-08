/* Lesson 6 widgets: how a DataLoader assembles batches, mini-batch vs full-batch
   training, and what an embedding table is. */
(function () {
  "use strict";
  const FC = window.FC;
  const S = FC.svgEl;
  function rng(seed) { return function () { seed |= 0; seed = (seed + 0x6D2B79F5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
  function gauss(r) { let u = 0, v = 0; while (u === 0) u = r(); while (v === 0) v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
  const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
  const f = (v, n = 1) => (Number.isFinite(v) ? v.toFixed(n) : "∞");

  /* ---------- widget: a DataLoader, one batch at a time ---------- */
  FC.widgets.loaderAnim = function (el) {
    const N = 22;
    const st = { bs: 8, shuffle: true, drop: true, order: null, pos: 0, epoch: 0 };
    let q = rng(5);
    function newEpoch() {
      st.order = Array.from({ length: N }, (_, i) => i);
      if (st.shuffle) for (let i = N - 1; i > 0; i--) { const j = Math.floor(q() * (i + 1)); [st.order[i], st.order[j]] = [st.order[j], st.order[i]]; }
      st.pos = 0; st.epoch += 1;
    }
    el.innerHTML = `
      <div class="widget-head"><h3>What a DataLoader does</h3>
        <div class="presets btn-row"><button type="button" class="chip" data-next>next batch</button><button type="button" class="chip" data-epoch>new epoch</button></div></div>
      <div class="widget-body">
        <div class="widget-controls">
          <label>batch_size <select data-bs><option>4</option><option selected>8</option><option>10</option></select></label>
          <label><input type="checkbox" data-shuffle checked> shuffle</label>
          <label><input type="checkbox" data-drop checked> drop_last</label>
        </div>
        <svg role="img" aria-label="22 dataset items, the current batch highlighted"></svg>
        <p class="readout" data-readout></p>
        <p class="muted" data-note style="max-width:68ch"></p>
      </div>`;
    const svg = el.querySelector("svg"), readout = el.querySelector("[data-readout]"), note = el.querySelector("[data-note]");
    function batchesPerEpoch() { return st.drop ? Math.floor(N / st.bs) : Math.ceil(N / st.bs); }
    function draw() {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      svg.setAttribute("viewBox", "0 0 760 170");
      const start = st.pos * st.bs, end = Math.min(N, start + st.bs), inBatch = new Set(st.order.slice(start, end));
      const dropped = st.drop && N % st.bs !== 0 ? new Set(st.order.slice(Math.floor(N / st.bs) * st.bs)) : new Set();
      svg.appendChild(S("text", { x: 20, y: 18, class: "cell-text", style: "fill: var(--ink-2)" }, "dataset[i], i = 0 … 21   (one trip each: continuous (4,), zones (2,), minutes ())"));
      for (let i = 0; i < N; i++) {
        const x = 20 + i * 33, hot = inBatch.has(i), gone = dropped.has(i);
        svg.appendChild(S("rect", { x, y: 30, width: 28, height: 28, rx: 4, style: `fill: ${hot ? "var(--dim-n)" : "var(--panel)"}; stroke: ${gone ? "var(--bad)" : hot ? "var(--dim-n)" : "var(--line)"}; stroke-dasharray: ${gone ? "3 3" : "none"}` }));
        svg.appendChild(S("text", { x: x + 14, y: 49, "text-anchor": "middle", class: "cell-text", style: `fill: ${hot ? "var(--paper)" : "var(--ink-2)"}` }, String(i)));
      }
      svg.appendChild(S("text", { x: 20, y: 90, class: "cell-text", style: "fill: var(--ink-2)" }, `this epoch's order: [${st.order.join(", ")}]`));
      const b = st.order.slice(start, end);
      svg.appendChild(S("text", { x: 20, y: 118, class: "axis-label", style: "fill: var(--dim-n)" }, `batch ${st.pos + 1} of ${batchesPerEpoch()}: indices [${b.join(", ")}]`));
      svg.appendChild(S("text", { x: 20, y: 142, class: "cell-text", style: "fill: var(--ink)" }, `collate -> continuous (${b.length}, 4)   zones (${b.length}, 2)   minutes (${b.length},)`));
      if (dropped.size) svg.appendChild(S("text", { x: 20, y: 162, class: "cell-text", style: "fill: var(--bad)" }, `drop_last: ${dropped.size} item${dropped.size > 1 ? "s" : ""} (${[...dropped].join(", ")}) skipped this epoch; they come back in the next shuffle`));
      readout.innerHTML = `DataLoader(dataset, batch_size=${st.bs}, shuffle=${st.shuffle ? "True" : "False"}, drop_last=${st.drop ? "True" : "False"})   <span class="c"># epoch ${st.epoch}, ${batchesPerEpoch()} batches, len(loader) == ${batchesPerEpoch()}</span>`;
      note.textContent = st.shuffle ? "Each epoch the loader draws a fresh permutation of the indices and walks it in chunks of batch_size. Every item is seen once per epoch, in a different company each time." : "Without shuffle the batches are the same every epoch, in dataset order. Fine for validation, bad for training: the model sees the same neighbours together forever, and any order in the file becomes order in the gradients.";
    }
    el.querySelector("[data-next]").addEventListener("click", () => { st.pos += 1; if (st.pos >= batchesPerEpoch()) newEpoch(); draw(); });
    el.querySelector("[data-epoch]").addEventListener("click", () => { newEpoch(); draw(); });
    el.querySelector("[data-bs]").addEventListener("change", (e) => { st.bs = +e.target.value; newEpoch(); draw(); });
    el.querySelector("[data-shuffle]").addEventListener("change", (e) => { st.shuffle = e.target.checked; newEpoch(); draw(); });
    el.querySelector("[data-drop]").addEventListener("change", (e) => { st.drop = e.target.checked; st.pos = 0; draw(); });
    newEpoch(); draw();
  };

  /* ---------- widget: mini-batches vs the whole batch ---------- */
  FC.widgets.batchCurves = function (el) {
    const truth = (d) => 8 + 3.5 * d + 5 * Math.max(0, d - 7);
    const r = rng(23), X = [], Y = [], XV = [], YV = [];
    for (let i = 0; i < 240; i++) { const d = 0.5 + 9.5 * r(); X.push((d - 5.25) / 2.74); Y.push((truth(d) + 3 * gauss(r) - 30) / 12); }
    for (let i = 0; i < 200; i++) { const d = 0.5 + 9.5 * r(); XV.push((d - 5.25) / 2.74); YV.push((truth(d) + 3 * gauss(r) - 30) / 12); }
    const EPOCHS = 30;
    function run(bs) {
      const net = FC.makeNet(16, 3), q = rng(77), va = [];
      let updates = 0;
      for (let e = 1; e <= EPOCHS; e++) {
        const order = Array.from({ length: X.length }, (_, i) => i);
        for (let i = order.length - 1; i > 0; i--) { const j = Math.floor(q() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
        for (let s = 0; s < order.length; s += bs) { const idx = order.slice(s, s + bs); net.step(idx.map((i) => X[i]), idx.map((i) => Y[i]), 0.01, 0); updates += 1; }
        va.push([e, net.loss(XV, YV) * 144]);
      }
      return { va, updates };
    }
    const runs = [[240, "var(--ink-3)", "whole dataset per step (1 update per epoch)"], [32, "var(--dim-n)", "batch_size=32 (8 updates per epoch)"], [8, "var(--dim-y)", "batch_size=8 (30 updates per epoch)"]].map(([bs, col, label]) => ({ bs, col, label, ...run(bs) }));
    el.innerHTML = `
      <div class="widget-head"><h3>Mini-batches: more updates per pass through the data</h3></div>
      <div class="widget-body">
        <svg role="img" aria-label="validation loss per epoch for three batch sizes"></svg>
        <ul class="legend">${runs.map((R) => `<li><span class="swatch" style="background:${R.col}"></span>${R.label}: ${R.updates} updates in ${EPOCHS} epochs, final validation loss ${f(R.va[R.va.length - 1][1])}</li>`).join("")}</ul>
        <p class="muted" style="max-width:68ch">Same model, same learning rate, same 240 deliveries, same 30 epochs. Each gradient step sees only a batch, so it is noisier, but there are many more of them. The curve for small batches is bumpier and far ahead. With real data in the millions of rows, a step on the whole dataset would not even fit in memory, so this is not a choice but a necessity; the noise turns out to help as well.</p>
      </div>`;
    const svg = el.querySelector("svg");
    const W = 640, H = 300, L = 48, Rr = 12, T = 14, B = 34, lo = 0, hi = 2.5;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    const sx = (e) => L + ((e - 1) / (EPOCHS - 1)) * (W - L - Rr), sy = (v) => T + (1 - (Math.max(lo, Math.min(hi, Math.log10(Math.max(v, 1)))) - lo) / (hi - lo)) * (H - T - B);
    [0, 1, 2].forEach((p) => { svg.appendChild(S("line", { x1: L, y1: sy(10 ** p), x2: W - Rr, y2: sy(10 ** p), style: "stroke: var(--line-2)" })); svg.appendChild(S("text", { x: L - 6, y: sy(10 ** p) + 4, "text-anchor": "end", class: "cell-text", style: "fill: var(--ink-3)" }, String(10 ** p))); });
    [1, 10, 20, 30].forEach((e) => svg.appendChild(S("text", { x: sx(e), y: H - B + 16, "text-anchor": "middle", class: "cell-text", style: "fill: var(--ink-3)" }, String(e))));
    svg.appendChild(S("text", { x: (L + W) / 2, y: H - 4, "text-anchor": "middle", class: "cell-text", style: "fill: var(--ink-2)" }, "epoch (one pass over the 240 deliveries)"));
    svg.appendChild(S("text", { x: L + 4, y: T + 6, class: "cell-text", style: "fill: var(--ink-2)" }, "validation loss, log scale"));
    runs.forEach((R) => svg.appendChild(S("path", { d: R.va.map((q, i) => (i ? "L" : "M") + sx(q[0]) + " " + sy(q[1])).join(" "), fill: "none", style: `stroke: ${R.col}; stroke-width: 2.5` })));
  };

  /* ---------- widget: an embedding table ---------- */
  FC.widgets.embeddingLookup = function (el) {
    const zones = [{ id: 4, name: "Alphabet City" }, { id: 43, name: "Central Park" }, { id: 79, name: "East Village" }, { id: 132, name: "JFK Airport" }, { id: 138, name: "LaGuardia Airport" }, { id: 161, name: "Midtown Center" }, { id: 230, name: "Times Square" }, { id: 263, name: "Yorkville West" }];
    const q = rng(132), table = zones.map((z) => Array.from({ length: 8 }, () => +(gauss(q) * 0.6).toFixed(2)));
    table[3] = [1.9, -0.4, 1.6, 0.3, -1.2, 0.8, 1.4, -0.2]; table[4] = [1.7, -0.3, 1.5, 0.5, -1.0, 0.6, 1.3, -0.4];   // the two airports end up near each other
    let sel = 3;
    el.innerHTML = `
      <div class="widget-head"><h3>nn.Embedding(266, 8): a table the model learns</h3><div class="presets btn-row" data-zones></div></div>
      <div class="widget-body">
        <svg role="img" aria-label="rows of an embedding table, one highlighted"></svg>
        <p class="readout" data-readout></p>
        <p class="muted" style="max-width:68ch">A one-hot zone would be a vector of 266 numbers, all zero but one, and the first Linear layer would multiply it by a (266, 8) weight matrix, which is just picking out one row. nn.Embedding skips the multiplication and does the pick directly. The rows start random and are trained like any other weights, so zones that behave alike end up with similar rows. These eight values are illustrative; the ones your model learns will differ.</p>
      </div>`;
    const svg = el.querySelector("svg"), readout = el.querySelector("[data-readout]"), zb = el.querySelector("[data-zones]");
    function draw() {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      svg.setAttribute("viewBox", "0 0 760 230");
      svg.appendChild(S("text", { x: 20, y: 16, class: "cell-text", style: "fill: var(--ink-2)" }, "weight: (266 zones, 8 numbers each); eight rows shown"));
      zones.forEach((z, i) => {
        const y = 28 + i * 22, hot = i === sel;
        svg.appendChild(S("text", { x: 70, y: y + 14, "text-anchor": "end", class: "cell-text", style: `fill: ${hot ? "var(--ink)" : "var(--ink-3)"}; font-weight: ${hot ? 600 : 400}` }, `${z.id}`));
        table[i].forEach((v, j) => {
          const x = 84 + j * 46, mag = Math.min(1, Math.abs(v) / 2);
          svg.appendChild(S("rect", { x, y, width: 42, height: 18, rx: 3, style: `fill: var(--dim-w); fill-opacity: ${0.08 + 0.55 * mag}; stroke: ${hot ? "var(--ink)" : "none"}` }));
          svg.appendChild(S("text", { x: x + 21, y: y + 13, "text-anchor": "middle", class: "cell-text", style: `fill: ${hot ? "var(--ink)" : "var(--ink-2)"}` }, v.toFixed(1)));
        });
        svg.appendChild(S("text", { x: 470, y: y + 14, class: "cell-text", style: `fill: ${hot ? "var(--ink)" : "var(--ink-3)"}` }, z.name));
      });
      svg.appendChild(S("text", { x: 20, y: 220, class: "cell-text", style: "fill: var(--ink-2)" }, "the airports share a pattern: long, predictable trips; the model has no idea they are airports, only that their rows should look alike"));
      const z = zones[sel];
      readout.innerHTML = `pu_zone = torch.tensor([${z.id}])        <span class="c"># ${z.name}</span><br>model.pu(pu_zone).shape   <span class="c"># torch.Size([1, 8])  one row</span><br>model.pu(pu_zone)         <span class="c"># tensor([[${table[sel].map((v) => v.toFixed(1)).join(", ")}]])</span>`;
    }
    zones.forEach((z, i) => { const b = document.createElement("button"); b.type = "button"; b.className = "chip"; b.textContent = String(z.id); b.setAttribute("aria-pressed", String(i === sel)); b.addEventListener("click", () => { sel = i; zb.querySelectorAll(".chip").forEach((c) => c.setAttribute("aria-pressed", "false")); b.setAttribute("aria-pressed", "true"); draw(); }); zb.appendChild(b); });
    draw();
  };
})();
