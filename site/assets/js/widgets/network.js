/* Lesson 4 widgets. A tiny neural network trained live in the browser, and the
   residuals a straight line leaves behind on the nonlinear deliveries. */
(function () {
  "use strict";
  const FC = window.FC;
  const S = FC.svgEl;
  function rng(seed) { return function () { seed |= 0; seed = (seed + 0x6D2B79F5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
  function gauss(r) { let u = 0, v = 0; while (u === 0) u = r(); while (v === 0) v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
  const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
  const f = (v, n = 2) => (Number.isFinite(v) ? v.toFixed(n) : "∞");

  function axes(svg, W, H, range, titles) {
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    const L = 46, R = 12, T = 14, B = 34;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    const sx = (x) => L + ((x - range.x0) / (range.x1 - range.x0)) * (W - L - R);
    const sy = (y) => T + (1 - (y - range.y0) / (range.y1 - range.y0)) * (H - T - B);
    const g = S("g");
    (range.yTicks || []).forEach((v) => { g.appendChild(S("line", { x1: L, y1: sy(v), x2: W - R, y2: sy(v), style: "stroke: var(--line-2)" })); g.appendChild(S("text", { x: L - 6, y: sy(v) + 4, "text-anchor": "end", class: "cell-text", style: "fill: var(--ink-3)" }, String(v))); });
    (range.xTicks || []).forEach((v) => g.appendChild(S("text", { x: sx(v), y: H - B + 16, "text-anchor": "middle", class: "cell-text", style: "fill: var(--ink-3)" }, String(v))));
    g.appendChild(S("line", { x1: L, y1: sy(range.y0), x2: W - R, y2: sy(range.y0), style: "stroke: var(--line)" }));
    g.appendChild(S("text", { x: (L + W - R) / 2, y: H - 4, "text-anchor": "middle", class: "cell-text", style: "fill: var(--ink-2)" }, titles.x));
    g.appendChild(S("text", { x: L + 4, y: T + 6, class: "cell-text", style: "fill: var(--ink-2)" }, titles.y));
    svg.appendChild(g);
    const clampY = (y) => Math.max(range.y0, Math.min(range.y1, y));
    return { g, sx, sy, clampY };
  }

  /* ---------- widget: a one-input network, trained in the browser ---------- */
  FC.widgets.netPlayground = function (el) {
    const r = rng(21), N = 80, xs = [], ys = [];
    for (let i = 0; i < N; i++) { const d = 0.5 + 9.5 * r(); xs.push(d); ys.push(8 + 3.5 * d + 5 * Math.max(0, d - 7) + 2 * gauss(r)); }
    const mx = mean(xs), sxd = Math.sqrt(mean(xs.map((v) => (v - mx) ** 2))), my = mean(ys), syd = Math.sqrt(mean(ys.map((v) => (v - my) ** 2)));
    const X = xs.map((v) => (v - mx) / sxd), Y = ys.map((v) => (v - my) / syd);
    const acts = {
      relu: { name: "ReLU", fn: (z) => Math.max(0, z), d: (z) => (z > 0 ? 1 : 0) },
      tanh: { name: "tanh", fn: Math.tanh, d: (z) => 1 - Math.tanh(z) ** 2 },
      none: { name: "none", fn: (z) => z, d: () => 1 },
    };
    const st = { H: 8, act: "relu", step: 0, p: null, adam: null, timer: null };
    function init() {
      const q = rng(5 + st.H);
      const W1 = [], b1 = [], W2 = [];
      for (let j = 0; j < st.H; j++) { W1.push((q() < 0.5 ? -1 : 1) * (0.7 + 0.6 * q())); b1.push(-1.6 + 3.2 * q()); W2.push(0.3 * (q() - 0.5)); }
      st.p = { W1, b1, W2, b2: 0 };
      st.adam = { m: { W1: W1.map(() => 0), b1: W1.map(() => 0), W2: W1.map(() => 0), b2: 0 }, v: { W1: W1.map(() => 0), b1: W1.map(() => 0), W2: W1.map(() => 0), b2: 0 }, t: 0 };
      st.step = 0;
    }
    function predictStd(x) { const a = acts[st.act]; let s = st.p.b2; for (let j = 0; j < st.H; j++) s += st.p.W2[j] * a.fn(st.p.W1[j] * x + st.p.b1[j]); return s; }
    function loss() { return mean(X.map((x, i) => (predictStd(x) - Y[i]) ** 2)) * syd * syd; }
    function trainStep() {
      const a = acts[st.act], p = st.p, H = st.H;
      const gW1 = new Array(H).fill(0), gb1 = new Array(H).fill(0), gW2 = new Array(H).fill(0); let gb2 = 0;
      for (let i = 0; i < N; i++) {
        const z = [], h = [];
        let pred = p.b2;
        for (let j = 0; j < H; j++) { z[j] = p.W1[j] * X[i] + p.b1[j]; h[j] = a.fn(z[j]); pred += p.W2[j] * h[j]; }
        const g = (2 * (pred - Y[i])) / N;
        gb2 += g;
        for (let j = 0; j < H; j++) { gW2[j] += g * h[j]; const dz = g * p.W2[j] * a.d(z[j]); gW1[j] += dz * X[i]; gb1[j] += dz; }
      }
      const A = st.adam, lr = 0.03, b1 = 0.9, b2 = 0.999, eps = 1e-8; A.t += 1;
      const upd = (val, grad, m, v) => { const mn = b1 * m + (1 - b1) * grad, vn = b2 * v + (1 - b2) * grad * grad; const mh = mn / (1 - b1 ** A.t), vh = vn / (1 - b2 ** A.t); return [val - (lr * mh) / (Math.sqrt(vh) + eps), mn, vn]; };
      for (let j = 0; j < H; j++) {
        [p.W1[j], A.m.W1[j], A.v.W1[j]] = upd(p.W1[j], gW1[j], A.m.W1[j], A.v.W1[j]);
        [p.b1[j], A.m.b1[j], A.v.b1[j]] = upd(p.b1[j], gb1[j], A.m.b1[j], A.v.b1[j]);
        [p.W2[j], A.m.W2[j], A.v.W2[j]] = upd(p.W2[j], gW2[j], A.m.W2[j], A.v.W2[j]);
      }
      [p.b2, A.m.b2, A.v.b2] = upd(p.b2, gb2, A.m.b2, A.v.b2);
      st.step += 1;
    }
    el.innerHTML = `
      <div class="widget-head"><h3>A one-input network you can train</h3>
        <div class="presets btn-row"><button type="button" class="chip" data-train>Train 600 steps</button><button type="button" class="chip" data-reset>Reset weights</button></div></div>
      <div class="widget-body">
        <div class="widget-controls">
          <label>hidden units <select data-h>${[1, 2, 4, 8, 16].map((h) => `<option value="${h}" ${h === 8 ? "selected" : ""}>${h}</option>`).join("")}</select></label>
          <label>activation <select data-act><option value="relu">ReLU</option><option value="tanh">tanh</option><option value="none">none (linear)</option></select></label>
          <label><input type="checkbox" data-units checked> show each hidden unit</label>
        </div>
        <svg role="img" aria-label="delivery minutes against distance with the network's fit"></svg>
        <p class="readout" data-readout></p>
        <p class="muted" data-note style="max-width:68ch"></p>
      </div>`;
    const svg = el.querySelector("svg"), readout = el.querySelector("[data-readout]"), note = el.querySelector("[data-note]");
    const selH = el.querySelector("[data-h]"), selA = el.querySelector("[data-act]"), chkU = el.querySelector("[data-units]");
    function draw() {
      const P = axes(svg, 640, 340, { x0: 0, x1: 10.5, y0: 0, y1: 70, xTicks: [0, 2, 4, 6, 8, 10], yTicks: [20, 40, 60] }, { x: "distance (km)", y: "minutes" });
      const a = acts[st.act];
      if (chkU.checked && st.act !== "none" && st.H <= 16) {
        for (let j = 0; j < st.H; j++) {
          let d = "";
          for (let k = 0; k <= 60; k++) { const x = (10.5 * k) / 60, xsd = (x - mx) / sxd; const y = my + syd * st.p.W2[j] * a.fn(st.p.W1[j] * xsd + st.p.b1[j]); d += (k ? " L" : "M") + P.sx(x) + " " + P.sy(P.clampY(y)); }
          P.g.appendChild(S("path", { d, fill: "none", style: "stroke: var(--dim-w); stroke-width: 1; opacity: .45" }));
        }
      }
      xs.forEach((x, i) => P.g.appendChild(S("circle", { cx: P.sx(x), cy: P.sy(P.clampY(ys[i])), r: 3, style: "fill: var(--ink-3)" })));
      let d = "";
      for (let k = 0; k <= 100; k++) { const x = (10.5 * k) / 100, y = my + syd * predictStd((x - mx) / sxd); d += (k ? " L" : "M") + P.sx(x) + " " + P.sy(P.clampY(y)); }
      P.g.appendChild(S("path", { d, fill: "none", style: "stroke: var(--dim-y); stroke-width: 3", "stroke-linecap": "round" }));
      readout.innerHTML = `MLP(1, hidden=(${st.H},), activation=${a.name})   <span class="c"># ${3 * st.H + 1} parameters</span><br>step ${st.step}, loss ${f(loss(), 1)} minutes²   <span class="c"># the noise floor is about 4</span>`;
      note.textContent = st.act === "none"
        ? "No activation: a sum of straight lines is a straight line, however many units you add. Train as long as you like; the fit stays straight and the loss stays high."
        : st.act === "relu"
          ? `Each thin line is one hidden unit: a ramp that is flat, then rises or falls from its own kink. The thick line is their sum. With ${st.H} unit${st.H > 1 ? "s" : ""} the fit can bend at most ${st.H} time${st.H > 1 ? "s" : ""}; the data bends once, at 7 km.`
          : "tanh units are smooth S-curves instead of ramps. Their sum can bend too, with soft corners rather than kinks. ReLU trains faster for large networks, which is why it is the default.";
    }
    function train() {
      if (st.timer) return;
      const reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      if (reduce) { for (let i = 0; i < 600; i++) trainStep(); draw(); return; }
      let done = 0;
      st.timer = setInterval(() => { for (let i = 0; i < 20; i++) trainStep(); done += 20; draw(); if (done >= 600) { clearInterval(st.timer); st.timer = null; } }, 40);
    }
    function reset() { if (st.timer) { clearInterval(st.timer); st.timer = null; } st.H = +selH.value; st.act = selA.value; init(); draw(); }
    el.querySelector("[data-train]").addEventListener("click", train);
    el.querySelector("[data-reset]").addEventListener("click", reset);
    selH.addEventListener("change", reset); selA.addEventListener("change", reset); chkU.addEventListener("change", draw);
    init(); draw();
  };

  /* ---------- widget: what the straight line leaves behind ---------- */
  FC.widgets.residuals = function (el) {
    const r = rng(33), N = 400, rows = [], y = [];
    for (let i = 0; i < N; i++) {
      const d = 0.5 + 9.5 * r(), items = 1 + Math.floor(6 * r()), rain = r() < 0.3 ? 1 : 0, rush = r() < 0.25 ? 1 : 0;
      rows.push([d, items, rain, rush, 1]);
      y.push(8 + 3.5 * d + 1.5 * items + 4 * rain + 6 * rush + 1.5 * rain * d + 5 * Math.max(0, d - 7) + 2 * gauss(r));
    }
    // ordinary least squares, 5 unknowns, by Gaussian elimination on the normal equations
    const k = 5, A = Array.from({ length: k }, () => new Array(k + 1).fill(0));
    rows.forEach((x, i) => { for (let a = 0; a < k; a++) { for (let b = 0; b < k; b++) A[a][b] += x[a] * x[b]; A[a][k] += x[a] * y[i]; } });
    for (let c = 0; c < k; c++) { let piv = c; for (let rr = c + 1; rr < k; rr++) if (Math.abs(A[rr][c]) > Math.abs(A[piv][c])) piv = rr; [A[c], A[piv]] = [A[piv], A[c]]; for (let rr = 0; rr < k; rr++) if (rr !== c) { const m = A[rr][c] / A[c][c]; for (let cc = c; cc <= k; cc++) A[rr][cc] -= m * A[c][cc]; } }
    const w = A.map((row, i) => row[k] / row[i]);
    const res = rows.map((x, i) => x.reduce((s, v, j) => s + v * w[j], 0) - y[i]);
    const mse = mean(res.map((v) => v * v));
    el.innerHTML = `
      <div class="widget-head"><h3>What the straight line leaves behind</h3>
        <div class="presets btn-row"><button type="button" class="chip" data-col="none" aria-pressed="true">plain</button><button type="button" class="chip" data-col="2">colour by rain</button><button type="button" class="chip" data-col="3">colour by rush hour</button></div></div>
      <div class="widget-body">
        <svg role="img" aria-label="residuals of the linear model against distance"></svg>
        <p class="readout" data-readout></p>
        <p class="muted" style="max-width:68ch">A residual is prediction minus actual. If the straight line had captured everything, these points would be a shapeless band around zero. The trend line shows the pattern it missed: far deliveries take longer than the line thinks, and when it rains the error grows with distance. There is structure left to learn, and a line cannot learn it.</p>
      </div>`;
    const svg = el.querySelector("svg"), readout = el.querySelector("[data-readout]");
    let col = "none";
    function draw() {
      const P = axes(svg, 640, 300, { x0: 0, x1: 10.5, y0: -14, y1: 14, xTicks: [0, 2, 4, 6, 8, 10], yTicks: [-10, 0, 10] }, { x: "distance (km)", y: "prediction − actual (minutes)" });
      P.g.appendChild(S("line", { x1: P.sx(0), y1: P.sy(0), x2: P.sx(10.5), y2: P.sy(0), style: "stroke: var(--ink); stroke-dasharray: 4 4" }));
      rows.forEach((x, i) => {
        const on = col !== "none" && x[+col] === 1;
        P.g.appendChild(S("circle", { cx: P.sx(x[0]), cy: P.sy(P.clampY(res[i])), r: 3, style: `fill: ${on ? "var(--dim-d)" : "var(--ink-3)"}; opacity: ${col === "none" || on ? 0.85 : 0.35}` }));
      });
      // binned trend of the residuals (all points, or the highlighted group)
      const pick = (x) => col === "none" || x[+col] === 1;
      let d = "";
      for (let b = 0; b < 10; b++) {
        const lo = 0.5 + b * 0.95, hi = lo + 0.95, idx = rows.map((x, i) => i).filter((i) => pick(rows[i]) && rows[i][0] >= lo && rows[i][0] < hi);
        if (!idx.length) continue;
        d += (d ? " L" : "M") + P.sx((lo + hi) / 2) + " " + P.sy(mean(idx.map((i) => res[i])));
      }
      P.g.appendChild(S("path", { d, fill: "none", style: "stroke: var(--dim-y); stroke-width: 3", "stroke-linecap": "round" }));
      readout.innerHTML = `LinearRegressor fitted to the nonlinear deliveries: loss ${f(mse, 1)} minutes²   <span class="c"># the noise floor is about 4</span>${col !== "none" ? `<br><span class="c"># amber points: ${col === "2" ? "raining" : "rush hour"}; the trend line follows them</span>` : ""}`;
    }
    el.querySelectorAll("[data-col]").forEach((b) => b.addEventListener("click", () => { col = b.getAttribute("data-col"); el.querySelectorAll("[data-col]").forEach((c) => c.setAttribute("aria-pressed", String(c === b))); draw(); }));
    draw();
  };
})();
