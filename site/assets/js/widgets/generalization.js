/* Lesson 5 widgets: overfitting you can dial in, splits you can see, dropout you can toggle. */
(function () {
  "use strict";
  const FC = window.FC;
  const S = FC.svgEl;
  function rng(seed) { return function () { seed |= 0; seed = (seed + 0x6D2B79F5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
  function gauss(r) { let u = 0, v = 0; while (u === 0) u = r(); while (v === 0) v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
  const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
  const f = (v, n = 1) => (Number.isFinite(v) ? v.toFixed(n) : "∞");
  function axes(svg, W, H, range, titles) {
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    const L = 48, R = 12, T = 14, B = 34;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    const sx = (x) => L + ((x - range.x0) / (range.x1 - range.x0)) * (W - L - R);
    const sy = (y) => T + (1 - (y - range.y0) / (range.y1 - range.y0)) * (H - T - B);
    const g = S("g");
    (range.yTicks || []).forEach(([v, label]) => { g.appendChild(S("line", { x1: L, y1: sy(v), x2: W - R, y2: sy(v), style: "stroke: var(--line-2)" })); g.appendChild(S("text", { x: L - 6, y: sy(v) + 4, "text-anchor": "end", class: "cell-text", style: "fill: var(--ink-3)" }, label)); });
    (range.xTicks || []).forEach((v) => g.appendChild(S("text", { x: sx(v), y: H - B + 16, "text-anchor": "middle", class: "cell-text", style: "fill: var(--ink-3)" }, String(v))));
    g.appendChild(S("line", { x1: L, y1: sy(range.y0), x2: W - R, y2: sy(range.y0), style: "stroke: var(--line)" }));
    g.appendChild(S("text", { x: (L + W - R) / 2, y: H - 4, "text-anchor": "middle", class: "cell-text", style: "fill: var(--ink-2)" }, titles.x));
    g.appendChild(S("text", { x: L + 4, y: T + 6, class: "cell-text", style: "fill: var(--ink-2)" }, titles.y));
    svg.appendChild(g);
    return { g, sx, sy, clampY: (y) => Math.max(range.y0, Math.min(range.y1, y)) };
  }

  /* A one-input ReLU network with manual backprop and Adam; optional L2 weight decay. */
  function makeNet(H, seed) {
    const q = rng(seed), p = { W1: [], b1: [], W2: [], b2: 0 };
    for (let j = 0; j < H; j++) { p.W1.push((q() < 0.5 ? -1 : 1) * (0.7 + 0.6 * q())); p.b1.push(-1.6 + 3.2 * q()); p.W2.push(0.3 * (q() - 0.5)); }
    const z = () => new Array(H).fill(0), A = { m: { W1: z(), b1: z(), W2: z(), b2: 0 }, v: { W1: z(), b1: z(), W2: z(), b2: 0 }, t: 0 };
    const net = {
      H, p,
      predict(x) { let s = p.b2; for (let j = 0; j < H; j++) s += p.W2[j] * Math.max(0, p.W1[j] * x + p.b1[j]); return s; },
      loss(X, Y) { return mean(X.map((x, i) => (net.predict(x) - Y[i]) ** 2)); },
      step(X, Y, lr, wd) {
        const N = X.length, gW1 = z(), gb1 = z(), gW2 = z(); let gb2 = 0;
        for (let i = 0; i < N; i++) {
          const zz = [], h = []; let pred = p.b2;
          for (let j = 0; j < H; j++) { zz[j] = p.W1[j] * X[i] + p.b1[j]; h[j] = Math.max(0, zz[j]); pred += p.W2[j] * h[j]; }
          const g = (2 * (pred - Y[i])) / N; gb2 += g;
          for (let j = 0; j < H; j++) { gW2[j] += g * h[j]; const dz = g * p.W2[j] * (zz[j] > 0 ? 1 : 0); gW1[j] += dz * X[i]; gb1[j] += dz; }
        }
        A.t += 1; const b1 = 0.9, b2 = 0.999, eps = 1e-8;
        const upd = (val, grad, m, v) => { grad += wd * val; const mn = b1 * m + (1 - b1) * grad, vn = b2 * v + (1 - b2) * grad * grad; return [val - (lr * (mn / (1 - b1 ** A.t))) / (Math.sqrt(vn / (1 - b2 ** A.t)) + eps), mn, vn]; };
        for (let j = 0; j < H; j++) { [p.W1[j], A.m.W1[j], A.v.W1[j]] = upd(p.W1[j], gW1[j], A.m.W1[j], A.v.W1[j]); [p.b1[j], A.m.b1[j], A.v.b1[j]] = upd(p.b1[j], gb1[j], A.m.b1[j], A.v.b1[j]); [p.W2[j], A.m.W2[j], A.v.W2[j]] = upd(p.W2[j], gW2[j], A.m.W2[j], A.v.W2[j]); }
        [p.b2, A.m.b2, A.v.b2] = upd(p.b2, gb2, A.m.b2, A.v.b2);
      },
    };
    return net;
  }
  FC.makeNet = makeNet;   // reused by Lesson 6's mini-batch widget

  /* ---------- widget: the overfitting lab ---------- */
  FC.widgets.overfitLab = function (el) {
    const truth = (d) => 8 + 3.5 * d + 5 * Math.max(0, d - 7);
    const r = rng(41), pool = [], val = [];
    for (let i = 0; i < 200; i++) { const d = 0.5 + 9.5 * r(); pool.push([d, truth(d) + 3 * gauss(r)]); }
    for (let i = 0; i < 200; i++) { const d = 0.5 + 9.5 * r(); val.push([d, truth(d) + 3 * gauss(r)]); }
    const mx = 5.25, sxd = 2.74, my = 30, syd = 12;   // fixed scaling so every run is comparable
    const nx = (d) => (d - mx) / sxd, ny = (m) => (m - my) / syd;
    const st = { n: 20, H: 32, wd: false, result: null };
    el.innerHTML = `
      <div class="widget-head"><h3>The overfitting lab</h3></div>
      <div class="widget-body">
        <div class="widget-controls">
          <label>training deliveries <select data-n>${[10, 20, 40, 80, 200].map((v) => `<option value="${v}" ${v === 20 ? "selected" : ""}>${v}</option>`).join("")}</select></label>
          <label>hidden units <select data-h>${[2, 8, 32, 64].map((v) => `<option value="${v}" ${v === 32 ? "selected" : ""}>${v}</option>`).join("")}</select></label>
          <label><input type="checkbox" data-wd> weight decay</label>
          <span class="muted">400 epochs, Adam. Validation set: 200 deliveries the model never trains on.</span>
        </div>
        <div class="two-col">
          <div><svg data-fit role="img" aria-label="the fitted curve over the training deliveries"></svg></div>
          <div><svg data-curves role="img" aria-label="training and validation loss per epoch"></svg></div>
        </div>
        <p class="readout" data-readout></p>
        <p class="muted" data-note style="max-width:68ch"></p>
      </div>`;
    const svgFit = el.querySelector("[data-fit]"), svgCurves = el.querySelector("[data-curves]"), readout = el.querySelector("[data-readout]"), note = el.querySelector("[data-note]");
    function trainRun() {
      const train = pool.slice(0, st.n), X = train.map((p) => nx(p[0])), Y = train.map((p) => ny(p[1])), XV = val.map((p) => nx(p[0])), YV = val.map((p) => ny(p[1]));
      const net = makeNet(st.H, 7 + st.H), tr = [], va = [];
      let best = { val: Infinity, epoch: 0, p: null };
      for (let e = 1; e <= 400; e++) {
        net.step(X, Y, 0.02, st.wd ? 0.02 : 0);
        if (e % 5 === 0) {
          const lt = net.loss(X, Y) * syd * syd, lv = net.loss(XV, YV) * syd * syd;
          tr.push([e, lt]); va.push([e, lv]);
          if (lv < best.val) best = { val: lv, epoch: e, p: JSON.parse(JSON.stringify(net.p)) };
        }
      }
      st.result = { net, train, tr, va, best, finalTrain: tr[tr.length - 1][1], finalVal: va[va.length - 1][1] };
    }
    function draw() {
      const R = st.result;
      const P = axes(svgFit, 420, 300, { x0: 0, x1: 10.5, y0: 0, y1: 70, xTicks: [0, 2, 4, 6, 8, 10], yTicks: [[20, "20"], [40, "40"], [60, "60"]] }, { x: "distance (km)", y: "minutes" });
      let d = "";
      for (let k = 0; k <= 100; k++) { const x = (10.5 * k) / 100, y = truth(x); d += (k ? " L" : "M") + P.sx(x) + " " + P.sy(P.clampY(y)); }
      P.g.appendChild(S("path", { d, fill: "none", style: "stroke: var(--ink-3); stroke-dasharray: 4 4; stroke-width: 1.5" }));
      R.train.forEach((p) => P.g.appendChild(S("circle", { cx: P.sx(p[0]), cy: P.sy(P.clampY(p[1])), r: 3.5, style: "fill: var(--ink)" })));
      d = "";
      for (let k = 0; k <= 160; k++) { const x = (10.5 * k) / 160, y = my + syd * R.net.predict(nx(x)); d += (k ? " L" : "M") + P.sx(x) + " " + P.sy(P.clampY(y)); }
      P.g.appendChild(S("path", { d, fill: "none", style: "stroke: var(--dim-y); stroke-width: 3", "stroke-linecap": "round" }));
      const lo = 0, hi = 3;   // log10 loss axis from 1 to 1000
      const C = axes(svgCurves, 420, 300, { x0: 0, x1: 400, y0: lo, y1: hi, xTicks: [0, 100, 200, 300, 400], yTicks: [[0, "1"], [1, "10"], [2, "100"], [3, "1000"]] }, { x: "epoch", y: "loss, log scale" });
      const lg = (v) => Math.max(lo, Math.min(hi, Math.log10(Math.max(v, 1))));
      const path = (arr, style) => C.g.appendChild(S("path", { d: arr.map((q, i) => (i ? "L" : "M") + C.sx(q[0]) + " " + C.sy(lg(q[1]))).join(" "), fill: "none", style }));
      path(R.va, "stroke: var(--dim-y); stroke-width: 2.5"); path(R.tr, "stroke: var(--ink); stroke-width: 2");
      C.g.appendChild(S("line", { x1: C.sx(R.best.epoch), y1: C.sy(lo), x2: C.sx(R.best.epoch), y2: C.sy(hi), style: "stroke: var(--dim-n); stroke-dasharray: 4 4" }));
      C.g.appendChild(S("text", { x: C.sx(R.best.epoch) + 4, y: C.sy(hi) + 12, class: "cell-text", style: "fill: var(--dim-n)" }, `best val, epoch ${R.best.epoch}`));
      const gap = R.finalVal / R.finalTrain, climbed = R.finalVal > 1.25 * R.best.val;
      readout.innerHTML = `${st.n} training deliveries, MLP(1, (${st.H},))${st.wd ? ", weight decay" : ""}: after 400 epochs <span class="c">train</span> ${f(R.finalTrain)} <span class="c">val</span> <b class="dim-y">${f(R.finalVal)}</b>   <span class="c"># best validation ${f(R.best.val)} at epoch ${R.best.epoch}; noise floor about 9</span>`;
      note.textContent = gap > 4 || climbed
        ? "The training loss keeps falling while the validation loss turned around: the model is memorising these particular deliveries, noise included, and the curve wiggles through points no new delivery will land on. Add data, shrink the model, stop at the blue line, or regularise."
        : gap > 1.8
          ? "A visible gap: the model does better on deliveries it has seen than on new ones. Not a disaster, but the blue line is where you would have stopped."
          : "Training and validation losses agree: whatever this model learned transfers to deliveries it never saw. More data, or less capacity, or regularisation got you here.";
    }
    const retrain = () => { trainRun(); draw(); };
    el.querySelector("[data-n]").addEventListener("change", (e) => { st.n = +e.target.value; retrain(); });
    el.querySelector("[data-h]").addEventListener("change", (e) => { st.H = +e.target.value; retrain(); });
    el.querySelector("[data-wd]").addEventListener("change", (e) => { st.wd = e.target.checked; retrain(); });
    retrain();
  };

  /* ---------- widget: splits ---------- */
  FC.widgets.splitBars = function (el) {
    const r = rng(3), N = 60, rows = [];
    for (let i = 0; i < N; i++) rows.push(0.5 + 9.5 * r());
    const sorted = rows.slice().sort((a, b) => a - b);
    let mode = "shuffled";
    el.innerHTML = `
      <div class="widget-head"><h3>Three parts, three jobs</h3>
        <div class="presets btn-row"><button type="button" class="chip" data-mode="shuffled" aria-pressed="true">shuffle, then cut</button><button type="button" class="chip" data-mode="sorted">cut the stored order</button></div></div>
      <div class="widget-body">
        <svg role="img" aria-label="sixty deliveries assigned to training, validation and test sets"></svg>
        <ul class="legend"><li><span class="swatch" style="background:var(--ink-3)"></span>train: fit the parameters</li><li><span class="swatch" style="background:var(--dim-c)"></span>validation: choose between models and know when to stop</li><li><span class="swatch" style="background:var(--dim-y)"></span>test: report one number, once, at the very end</li></ul>
        <p class="readout" data-readout></p>
      </div>`;
    const svg = el.querySelector("svg"), readout = el.querySelector("[data-readout]");
    function draw() {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      svg.setAttribute("viewBox", "0 0 760 150");
      const data = mode === "sorted" ? sorted : rows;
      const perm = Array.from({ length: N }, (_, i) => i);
      if (mode === "shuffled") { const q = rng(9); for (let i = N - 1; i > 0; i--) { const j = Math.floor(q() * (i + 1)); [perm[i], perm[j]] = [perm[j], perm[i]]; } }
      const which = new Array(N); perm.forEach((idx, k) => { which[idx] = k < 42 ? "train" : k < 51 ? "val" : "test"; });
      const colour = { train: "var(--ink-3)", val: "var(--dim-c)", test: "var(--dim-y)" };
      data.forEach((d, i) => {
        const x = 20 + i * 12, h = 8 + d * 8;
        svg.appendChild(S("rect", { x, y: 110 - h, width: 9, height: h, rx: 1.5, style: `fill: ${colour[which[i]]}` }));
      });
      svg.appendChild(S("text", { x: 20, y: 130, class: "cell-text", style: "fill: var(--ink-2)" }, mode === "sorted" ? "the rows as stored: sorted by distance (bar height)" : "the rows after torch.randperm: distances mixed"));
      svg.appendChild(S("text", { x: 20, y: 145, class: "cell-text", style: "fill: var(--ink-2)" }, "first 70% → train, next 15% → validation, last 15% → test"));
      const test = data.filter((_, i) => which[i] === "test");
      readout.innerHTML = mode === "sorted"
        ? `<span class="err">test set: distances ${f(Math.min(...test))} to ${f(Math.max(...test))} km only.</span> The model trains on short deliveries and is judged on long ones it never saw. Its test score says nothing about its real accuracy.`
        : `<span class="ok">test set: distances ${f(Math.min(...test))} to ${f(Math.max(...test))} km.</span> Each part is a fair sample of all deliveries, so a score on one part predicts the score on new deliveries.`;
    }
    el.querySelectorAll("[data-mode]").forEach((b) => b.addEventListener("click", () => { mode = b.getAttribute("data-mode"); el.querySelectorAll("[data-mode]").forEach((c) => c.setAttribute("aria-pressed", String(c === b))); draw(); }));
    draw();
  };

  /* ---------- widget: dropout ---------- */
  FC.widgets.dropoutDemo = function (el) {
    const H = 24, p = 0.3; let mode = "train", mask = null, q = rng(17);
    el.innerHTML = `
      <div class="widget-head"><h3>nn.Dropout(0.3), one hidden layer</h3>
        <div class="presets btn-row"><button type="button" class="chip" data-fwd>model.train(); forward pass</button><button type="button" class="chip" data-eval>model.eval(); forward pass</button></div></div>
      <div class="widget-body">
        <svg role="img" aria-label="24 hidden units, some switched off by dropout"></svg>
        <p class="readout" data-readout></p>
      </div>`;
    const svg = el.querySelector("svg"), readout = el.querySelector("[data-readout]");
    function draw() {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      svg.setAttribute("viewBox", "0 0 760 90");
      let off = 0;
      for (let j = 0; j < H; j++) {
        const dead = mode === "train" && mask && mask[j]; if (dead) off += 1;
        const cx = 24 + j * 30;
        svg.appendChild(S("circle", { cx, cy: 40, r: 11, style: dead ? "fill: var(--line-2); stroke: var(--line)" : "fill: var(--dim-w); stroke: var(--dim-w)" }));
        if (dead) svg.appendChild(S("text", { x: cx, y: 44, "text-anchor": "middle", class: "cell-text", style: "fill: var(--ink-3)" }, "×"));
      }
      svg.appendChild(S("text", { x: 24, y: 78, class: "cell-text", style: "fill: var(--ink-2)" }, mode === "train" && mask ? `${off} of ${H} units zeroed this pass; the ${H - off} survivors are multiplied by 1/(1−0.3) = 1.43 so the layer's total stays the same size` : mode === "eval" ? "eval mode: every unit on, nothing scaled; the same input always gives the same output" : "press a button to run a forward pass"));
      readout.innerHTML = mode === "train" ? `model.train()   <span class="c"># a fresh random mask on every forward pass: the network cannot rely on any one unit, so it spreads what it knows</span>` : `model.eval()    <span class="c"># dropout is a no-op; this is the mode for validation, testing and serving</span>`;
    }
    el.querySelector("[data-fwd]").addEventListener("click", () => { mode = "train"; mask = Array.from({ length: H }, () => q() < p); draw(); });
    el.querySelector("[data-eval]").addEventListener("click", () => { mode = "eval"; draw(); });
    mode = "none"; draw();
  };
})();
