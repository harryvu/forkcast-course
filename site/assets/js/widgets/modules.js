/* Lesson 3 widgets: nn.Linear's shapes, why normalization fixes the learning
   rate, and how the optimizer's three lines map onto Lesson 2's loop. */
(function () {
  "use strict";
  const FC = window.FC;
  const S = FC.svgEl;
  const f = (v, n = 2) => (Number.isFinite(v) ? v.toFixed(n) : "∞");

  /* ---------- widget: nn.Linear anatomy ---------- */
  FC.widgets.linearAnatomy = function (el) {
    let out = 1, N = 32;
    el.innerHTML = `
      <div class="widget-head"><h3>What nn.Linear(4, out) holds</h3>
        <div class="presets btn-row"><button type="button" class="chip" data-out="1" aria-pressed="true">out = 1</button><button type="button" class="chip" data-out="3">out = 3</button><button type="button" class="chip" data-n="1">one order</button><button type="button" class="chip" data-n="32" aria-pressed="true">a batch of 32</button></div></div>
      <div class="widget-body">
        <svg role="img" aria-label="shapes of the input, the weight, the bias and the output of a linear layer"></svg>
        <p class="readout" data-readout></p>
        <p class="muted" data-note style="max-width:68ch"></p>
      </div>`;
    const svg = el.querySelector("svg"), readout = el.querySelector("[data-readout]"), note = el.querySelector("[data-note]");
    const CW = 26, CH = 18, GAP = 3;
    function block(x, y, rows, cols, stroke, label, shapeHtml) {
      const r = Math.min(rows, 6), c = cols;
      for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) svg.appendChild(S("rect", { x: x + j * (CW + GAP), y: y + i * (CH + GAP), width: CW, height: CH, rx: 3, style: `fill: var(--panel); stroke: ${stroke}; stroke-width: 1.2` }));
      if (rows > 6) for (let k = 0; k < 3; k++) svg.appendChild(S("circle", { cx: x + (c * (CW + GAP)) / 2, cy: y + 6 * (CH + GAP) + 4 + k * 5, r: 1.3, class: "dots" }));
      const w = c * (CW + GAP) - GAP, h = (rows > 6 ? 7 : r) * (CH + GAP);
      svg.appendChild(S("text", { x: x + w / 2, y: y + h + 16, "text-anchor": "middle", class: "axis-label", style: "fill: var(--ink-2)" }, label));
      const t = S("text", { x: x + w / 2, y: y + h + 31, "text-anchor": "middle", class: "cell-text" }); t.innerHTML = shapeHtml; svg.appendChild(t);
      return w;
    }
    const sp = (n, role) => `<tspan style="fill: var(--dim-${role})">${n}</tspan>`;
    function draw() {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      svg.setAttribute("viewBox", "0 0 760 215");
      let x = 20; const y = 30;
      const wx = block(x, y, N, 4, "var(--dim-n)", "X", `(${sp(N, "n")}, ${sp(4, "d")})`); x += wx + 34;
      svg.appendChild(S("text", { x: x - 18, y: y + 50, "text-anchor": "middle", class: "axis-label" }, "@"));
      const ww = block(x, y, 4, out, "var(--dim-w)", "weight.T", `(${sp(4, "d")}, ${sp(out, "y")})`); x += ww + 34;
      svg.appendChild(S("text", { x: x - 18, y: y + 50, "text-anchor": "middle", class: "axis-label" }, "+"));
      const wb = block(x, y, 1, out, "var(--dim-w)", "bias", `(${sp(out, "y")},)`); x += wb + 34;
      svg.appendChild(S("text", { x: x - 18, y: y + 50, "text-anchor": "middle", class: "axis-label" }, "="));
      block(x, y, N, out, "var(--dim-y)", "layer(X)", `(${sp(N, "n")}, ${sp(out, "y")})`);
      readout.innerHTML = `layer = nn.Linear(4, ${out})<br>layer.weight.shape   <span class="c">torch.Size([${out}, 4])   stored as (out, in); the layer transposes it for you</span><br>layer.bias.shape     <span class="c">torch.Size([${out}])</span><br>layer(X).shape       <span class="c">torch.Size([${N}, ${out}])   ${out === 1 ? "note the trailing 1: squeeze it before comparing with y" : "one column per output"}</span><br>parameters           <span class="c">${out * 4} weights + ${out} bias${out === 1 ? "" : "es"} = ${out * 5}</span>`;
      note.textContent = out === 1
        ? "With one output this is exactly Lesson 2's X @ w + b, with w stored as a (1, 4) row. nn.Linear also creates the parameters with sensible random starting values, registers them so the optimizer can find them, and knows how to save itself."
        : "With three outputs the same input produces three predictions at once: three weighted sums with their own weights and biases. Lesson 4 stacks layers like this to build a neural network.";
    }
    el.querySelectorAll("[data-out]").forEach((b) => b.addEventListener("click", () => { out = +b.getAttribute("data-out"); el.querySelectorAll("[data-out]").forEach((c) => c.setAttribute("aria-pressed", String(c === b))); draw(); }));
    el.querySelectorAll("[data-n]").forEach((b) => b.addEventListener("click", () => { N = +b.getAttribute("data-n"); el.querySelectorAll("[data-n]").forEach((c) => c.setAttribute("aria-pressed", String(c === b))); draw(); }));
    draw();
  };

  /* ---------- widget: two loss bowls, one learning rate ---------- */
  FC.widgets.conditioning = function (el) {
    function rng(seed) { return function () { seed |= 0; seed = (seed + 0x6D2B79F5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
    function gauss(r) { let u = 0, v = 0; while (u === 0) u = r(); while (v === 0) v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
    const r = rng(11), N = 60, X = [], Y = [];
    for (let i = 0; i < N; i++) { const d = 0.5 + 9.5 * r(), it = 1 + Math.floor(6 * r()); X.push([d, it]); Y.push(8 + 3.5 * d + 1.5 * it + 2 * gauss(r)); }
    const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
    const mu = [0, 1].map((j) => mean(X.map((x) => x[j]))), sd = [0, 1].map((j) => Math.sqrt(mean(X.map((x) => (x[j] - mu[j]) ** 2))));
    const Xn = X.map((x) => [(x[0] - mu[0]) / sd[0], (x[1] - mu[1]) / sd[1]]);
    // Ordinary least squares for (w1, w2, b) via centred normal equations, for each feature matrix
    function ols(M) {
      const m = [0, 1].map((j) => mean(M.map((x) => x[j]))), my = mean(Y);
      const c = (a, b) => mean(M.map((x, i) => (x[a] - m[a]) * (x[b] - m[b])));
      const cy = (a) => mean(M.map((x, i) => (x[a] - m[a]) * (Y[i] - my)));
      const a11 = c(0, 0), a12 = c(0, 1), a22 = c(1, 1), det = a11 * a22 - a12 * a12;
      const w1 = (a22 * cy(0) - a12 * cy(1)) / det, w2 = (a11 * cy(1) - a12 * cy(0)) / det;
      return { w: [w1, w2], b: my - w1 * m[0] - w2 * m[1] };
    }
    function panelFor(M, name) {
      const sol = ols(M);
      const A = [[mean(M.map((x) => x[0] * x[0])), mean(M.map((x) => x[0] * x[1]))], [0, mean(M.map((x) => x[1] * x[1]))]]; A[1][0] = A[0][1];
      const loss = (w) => mean(M.map((x, i) => (w[0] * x[0] + w[1] * x[1] + sol.b - Y[i]) ** 2));
      const grad = (w) => [2 * (A[0][0] * (w[0] - sol.w[0]) + A[0][1] * (w[1] - sol.w[1])), 2 * (A[1][0] * (w[0] - sol.w[0]) + A[1][1] * (w[1] - sol.w[1]))];
      const tr = A[0][0] + A[1][1], det = A[0][0] * A[1][1] - A[0][1] * A[0][1], disc = Math.sqrt(Math.max(0, tr * tr / 4 - det));
      const l1 = tr / 2 + disc, l2 = tr / 2 - disc, angle = (Math.atan2(A[0][1], l1 - A[1][1]) * 180) / Math.PI;
      const R = name === "raw" ? 6 : 12;
      return { name, sol, A, loss, grad, l1, l2, angle, R, lmin: loss(sol.w), w: null, path: [], steps: 0 };
    }
    const panels = [panelFor(X, "raw"), panelFor(Xn, "normalized")];
    const lr = { value: 0.05 };
    function reset(p) { p.w = [p.sol.w[0] - 0.7 * p.R, p.sol.w[1] + 0.7 * p.R]; p.path = [p.w.slice()]; p.steps = 0; p.state = "ready"; }
    panels.forEach(reset);
    el.innerHTML = `
      <div class="widget-head"><h3>Two loss bowls, one learning rate</h3>
        <div class="presets btn-row"><button type="button" class="chip" data-step>Step once</button><button type="button" class="chip" data-run>Run 40 steps</button><button type="button" class="chip" data-reset>Reset</button></div></div>
      <div class="widget-body">
        <div class="widget-controls">
          <label style="display:grid;grid-template-columns:auto 1fr auto;gap:.6rem;align-items:center;min-width:300px">learning rate <input type="range" min="0.005" max="0.6" step="0.005" value="0.05" data-lr aria-label="learning rate"><span class="shape" data-lrv>0.050</span></label>
          <span class="muted">Try 0.01, 0.05, then 0.4.</span>
        </div>
        <div class="two-col">
          <div><svg data-raw role="img" aria-label="loss contours for raw features"></svg><p class="readout" data-raw-out></p></div>
          <div><svg data-norm role="img" aria-label="loss contours for normalized features"></svg><p class="readout" data-norm-out></p></div>
        </div>
        <p class="muted" style="max-width:68ch">Same deliveries, same model, same start, same learning rate. Left: distance in km and items as they are. Right: each feature scaled to mean 0, std 1. The loss is a bowl in both cases; normalization just makes it round, and a round bowl is the one gradient descent is good at.</p>
      </div>`;
    const svgs = { raw: el.querySelector("[data-raw]"), normalized: el.querySelector("[data-norm]") };
    const outs = { raw: el.querySelector("[data-raw-out]"), normalized: el.querySelector("[data-norm-out]") };
    const ilr = el.querySelector("[data-lr]");
    function draw(p) {
      const svg = svgs[p.name]; while (svg.firstChild) svg.removeChild(svg.firstChild);
      const W = 380, H = 330, L = 44, T = 26, B = 36, Rr = 10, size = Math.min(W - L - Rr, H - T - B);
      svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
      const cx0 = p.sol.w[0], cy0 = p.sol.w[1];
      const sx = (w) => L + ((w - (cx0 - p.R)) / (2 * p.R)) * size, sy = (w) => T + (1 - (w - (cy0 - p.R)) / (2 * p.R)) * size;
      svg.appendChild(S("rect", { x: L, y: T, width: size, height: size, style: "fill: none; stroke: var(--line)" }));
      svg.appendChild(S("text", { x: L + size / 2, y: H - 8, "text-anchor": "middle", class: "cell-text", style: "fill: var(--ink-2)" }, p.name === "raw" ? "w for distance (km)" : "w for distance (standardized)"));
      svg.appendChild(S("text", { x: 14, y: T + size / 2, "text-anchor": "middle", class: "cell-text", style: "fill: var(--ink-2)", transform: `rotate(-90 14 ${T + size / 2})` }, "w for items"));
      svg.appendChild(S("text", { x: L, y: T - 10, class: "axis-label", style: "fill: var(--ink)" }, p.name === "raw" ? "raw features" : "normalized features"));
      const scale = size / (2 * p.R), Lmax = p.R * p.R * p.l2;
      for (let k = 1; k <= 6; k++) {
        const lvl = Lmax * (k / 6) ** 2;
        svg.appendChild(S("ellipse", { cx: 0, cy: 0, rx: Math.sqrt(lvl / p.l1) * scale, ry: Math.sqrt(lvl / p.l2) * scale, transform: `translate(${sx(cx0)} ${sy(cy0)}) rotate(${-p.angle})`, style: `fill: none; stroke: var(--dim-n); stroke-opacity: ${0.9 - k * 0.1}` }));
      }
      svg.appendChild(S("circle", { cx: sx(cx0), cy: sy(cy0), r: 3.5, style: "fill: var(--dim-w)" }));
      svg.appendChild(S("text", { x: sx(cx0) + 7, y: sy(cy0) - 6, class: "cell-text", style: "fill: var(--dim-w); font-weight: 600" }, "best w"));
      const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
      const pts = p.path.map((w) => [clamp(sx(w[0]), L, L + size), clamp(sy(w[1]), T, T + size)]);
      if (pts.length > 1) svg.appendChild(S("path", { d: pts.map((q, i) => (i ? "L" : "M") + q[0].toFixed(1) + " " + q[1].toFixed(1)).join(" "), fill: "none", style: "stroke: var(--dim-y); stroke-width: 2" }));
      const last = pts[pts.length - 1];
      svg.appendChild(S("circle", { cx: last[0], cy: last[1], r: 6, style: "fill: var(--dim-y); stroke: var(--panel); stroke-width: 2" }));
      const dist = Math.hypot(p.w[0] - cx0, p.w[1] - cy0);
      const status = !Number.isFinite(dist) || dist > 50 * p.R ? `<span class="err">diverged</span>` : dist < 0.02 * p.R ? `<span class="ok">converged in ${p.steps} steps</span>` : `step ${p.steps}`;
      outs[p.name].innerHTML = `${status}, loss ${f(p.loss(p.w), 1)} <span class="c">(best possible ${f(p.lmin, 1)})</span><br><span class="c">gradient descent is stable while lr &lt; ${f(1 / p.l1, 3)}</span>`;
    }
    function step(p) {
      if (!p.w.every(Number.isFinite) || Math.hypot(p.w[0] - p.sol.w[0], p.w[1] - p.sol.w[1]) > 50 * p.R) return;
      const g = p.grad(p.w); p.w = [p.w[0] - lr.value * g[0], p.w[1] - lr.value * g[1]]; p.path.push(p.w.slice()); p.steps += 1;
    }
    function drawAll() { el.querySelector("[data-lrv]").textContent = lr.value.toFixed(3); panels.forEach(draw); }
    ilr.addEventListener("input", () => { lr.value = +ilr.value; drawAll(); });
    el.querySelector("[data-step]").addEventListener("click", () => { panels.forEach(step); drawAll(); });
    el.querySelector("[data-run]").addEventListener("click", () => {
      const reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      if (reduce) { for (let i = 0; i < 40; i++) panels.forEach(step); drawAll(); return; }
      let n = 0; const t = setInterval(() => { panels.forEach(step); drawAll(); if (++n >= 40) clearInterval(t); }, 70);
    });
    el.querySelector("[data-reset]").addEventListener("click", () => { panels.forEach(reset); drawAll(); });
    drawAll();
  };

  /* ---------- widget: the loop, line by line ---------- */
  FC.widgets.loopStepper = function (el) {
    const stages = [
      { now: "optimizer.zero_grad()", before: "w.grad.zero_()\nb.grad.zero_()", note: "Clean slate for every parameter the optimizer was given, however many there are." },
      { now: "pred = model(X)", before: "pred = X @ w + b", note: "Calling the module runs forward(). The arithmetic is the same; the knobs now live inside the layer." },
      { now: "loss = loss_fn(pred, y)", before: "loss = ((pred - y) ** 2).mean()", note: "nn.MSELoss() is that line as an object, so the loop can take any loss without changing." },
      { now: "loss.backward()", before: "loss.backward()", note: "Unchanged. Autograd fills .grad on every parameter, exactly as in Lesson 2." },
      { now: "optimizer.step()", before: "with torch.no_grad():\n    w -= lr * w.grad\n    b -= lr * b.grad", note: "The update rule. For SGD it is Lesson 2's line; for Adam it is a smarter rule, and your loop does not change at all." },
    ];
    let i = 0;
    el.innerHTML = `
      <div class="widget-head"><h3>The loop, line by line</h3><div class="presets btn-row"><button type="button" class="chip" data-next>Next line</button><button type="button" class="chip" data-reset>Reset</button></div></div>
      <div class="widget-body">
        <div class="two-col">
          <div><p class="muted" style="margin:0 0 .3rem;font-size:.9rem">Lesson 3</p><pre class="no-copy" data-now></pre></div>
          <div><p class="muted" style="margin:0 0 .3rem;font-size:.9rem">Lesson 2, the same thing by hand</p><pre class="no-copy" data-before></pre></div>
        </div>
        <p class="muted" data-note style="max-width:68ch;margin:0"></p>
      </div>`;
    const now = el.querySelector("[data-now]"), before = el.querySelector("[data-before]"), note = el.querySelector("[data-note]");
    const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;");
    function draw() {
      now.innerHTML = stages.map((s, k) => `<code style="display:block;${k === i ? "background: rgba(255,255,255,0.12); outline: 1px solid var(--dim-n); border-radius: 4px" : "opacity: .55"}">${esc(s.now)}</code>`).join("");
      before.innerHTML = stages.map((s, k) => `<code style="display:block;white-space:pre;${k === i ? "background: rgba(255,255,255,0.12); outline: 1px solid var(--dim-n); border-radius: 4px" : "opacity: .55"}">${esc(s.before)}</code>`).join("");
      note.textContent = stages[i].note;
    }
    el.querySelector("[data-next]").addEventListener("click", () => { i = (i + 1) % stages.length; draw(); });
    el.querySelector("[data-reset]").addEventListener("click", () => { i = 0; draw(); });
    draw();
  };
})();
