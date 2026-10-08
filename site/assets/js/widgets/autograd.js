/* Autograd widgets for Lesson 2. Pure SVG, no dependencies.
   Forward values are shown in ink; gradients flowing backward are shown in the
   batch-blue used for slopes in Lesson 1's playground. */
(function () {
  "use strict";
  const FC = window.FC;
  const S = FC.svgEl;
  const f = (v, n = 3) => (Math.abs(v - Math.round(v)) < 1e-9 ? String(Math.round(v)) : v.toFixed(n).replace(/0+$/, "").replace(/\.$/, ""));
  const vec = (a, n) => "[" + a.map((v) => f(v, n)).join(", ") + "]";

  /* ---------- widget: the autograd graph, stepped through ---------- */
  FC.widgets.autogradGraph = function (el) {
    // Three deliveries, w = 3, b = 8: small enough to check every number by hand.
    const x = [1, 2, 3], y = [12, 15, 17], w = 3, b = 8;
    const wx = x.map((v) => w * v), pred = wx.map((v) => v + b), e = pred.map((v, i) => v - y[i]), sq = e.map((v) => v * v);
    const loss = sq.reduce((s, v) => s + v, 0) / 3;
    const g_sq = [1 / 3, 1 / 3, 1 / 3], g_e = e.map((v) => (2 * v) / 3), g_b = g_e.reduce((s, v) => s + v, 0), g_w = g_e.reduce((s, v, i) => s + v * x[i], 0);

    const nodes = {
      w:    { x: 150, y: 50,  label: "w", kind: "leaf", value: f(w), grad: f(g_w, 3), role: "w" },
      x:    { x: 150, y: 125, label: "x", kind: "data", value: vec(x), role: "d" },
      b:    { x: 150, y: 205, label: "b", kind: "leaf", value: f(b), grad: f(g_b, 3), role: "w" },
      y:    { x: 150, y: 280, label: "y", kind: "data", value: vec(y), role: "s" },
      mul:  { x: 320, y: 88,  label: "w * x", value: vec(wx), grad: vec(g_e, 2), local: "× x each" },
      add:  { x: 480, y: 125, label: "+ b", value: vec(pred), grad: vec(g_e, 2), local: "× 1" },
      sub:  { x: 640, y: 205, label: "− y", value: vec(e), grad: vec(g_e, 2), local: "× 1" },
      pow:  { x: 790, y: 205, label: "** 2", value: vec(sq), grad: vec(g_sq, 2), local: "× 2e", below: true },
      mean: { x: 930, y: 205, label: "mean", value: f(loss, 3), grad: "1", local: "× 1/3" },
    };
    const edges = [["w", "mul"], ["x", "mul"], ["mul", "add"], ["b", "add"], ["add", "sub"], ["y", "sub"], ["sub", "pow"], ["pow", "mean"]];
    // backward stages: which node's gradient is revealed at each click
    const stages = [
      { node: "mean", text: "loss.backward() starts at the end. The slope of the loss with respect to itself is 1." },
      { node: "pow", text: "mean of three numbers: each one contributes 1/3 of the result, so each receives 1/3 of the gradient." },
      { node: "sub", text: "e² has slope 2e. Multiply what arrived (1/3) by 2e: the gradient reaching each error is 2e/3." },
      { node: "add", text: "Subtracting y changes nothing for the prediction: slope 1. The gradient passes straight through." },
      { node: "mul", text: "Adding b: slope 1 again. Each prediction's gradient reaches w·x unchanged." },
      { node: "b", text: "b was added to all three predictions, so its gradient is the sum of theirs: −1.333. That is 2 × mean(e), Lesson 1's formula." },
      { node: "w", text: "w was multiplied by each x, so each gradient is scaled by its x before the three are summed: −2. That is 2 × mean(e·x). Same formula, derived by the machine." },
    ];
    let forward = false, stage = 0;

    el.innerHTML = `
      <div class="widget-head"><h3>What .backward() does: the graph of one loss</h3>
        <div class="presets btn-row"><button type="button" class="chip" data-fwd>Run forward</button><button type="button" class="chip" data-bwd>backward: one step</button><button type="button" class="chip" data-reset>Reset</button></div></div>
      <div class="widget-body">
        <svg role="img" aria-label="computation graph from w, x, b and y to the loss"></svg>
        <p class="readout" data-code></p>
        <p class="muted" data-note style="max-width:68ch"></p>
      </div>`;
    const svg = el.querySelector("svg"), code = el.querySelector("[data-code]"), note = el.querySelector("[data-note]");

    function draw() {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      svg.setAttribute("viewBox", "0 0 1010 340");
      const revealed = new Set(stages.slice(0, stage).map((s) => s.node));
      const current = stage > 0 ? stages[stage - 1].node : null;
      const defs = S("defs");
      defs.innerHTML = `<marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" style="fill: var(--ink-3)"/></marker>
        <marker id="arrg" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" style="fill: var(--dim-n)"/></marker>`;
      svg.appendChild(defs);
      edges.forEach(([a, c]) => {
        const A = nodes[a], C = nodes[c];
        svg.appendChild(S("line", { x1: A.x + 44, y1: A.y, x2: C.x - 46, y2: C.y, style: "stroke: var(--ink-3); stroke-width: 1.5", "marker-end": "url(#arr)" }));
        if (revealed.has(c) && (revealed.has(a) || nodes[a].kind === "leaf" && revealed.has(a))) {
          // gradient arrow drawn backwards, offset below the forward edge
          svg.appendChild(S("line", { x1: C.x - 46, y1: C.y + 10, x2: A.x + 44, y2: A.y + 10, style: "stroke: var(--dim-n); stroke-width: 1.5; stroke-dasharray: 5 4", "marker-end": "url(#arrg)" }));
        }
      });
      Object.entries(nodes).forEach(([key, n]) => {
        const isCur = key === current;
        const stroke = n.kind === "leaf" ? "var(--dim-w)" : n.kind === "data" ? "var(--ink-3)" : "var(--ink)";
        svg.appendChild(S("rect", { x: n.x - 44, y: n.y - 20, width: 88, height: 40, rx: 8, style: `fill: var(--panel); stroke: ${stroke}; stroke-width: ${isCur ? 3 : 1.5}` }));
        svg.appendChild(S("text", { x: n.x, y: n.y + 5, "text-anchor": "middle", class: "axis-label", style: `fill: ${n.kind === "leaf" ? "var(--dim-w)" : "var(--ink)"}; font-size: 14px` }, n.label));
        if (forward) svg.appendChild(S("text", { x: n.x, y: n.y + 36, "text-anchor": "middle", class: "cell-text", style: "fill: var(--ink-2)" }, n.value));
        if (n.grad !== undefined && revealed.has(key)) {
          if (n.kind === "leaf") {   // leaves: gradient label to the left, where nothing else is drawn
            svg.appendChild(S("text", { x: n.x - 52, y: n.y + 4, "text-anchor": "end", class: "cell-text", style: "fill: var(--dim-n); font-weight: 600" }, "grad " + n.grad));
          } else {   // alternate above/below so neighbouring labels never touch
            const gy = n.below ? n.y + 54 : n.y - 28, ly = n.below ? n.y + 68 : n.y - 42;
            svg.appendChild(S("text", { x: n.x, y: gy, "text-anchor": "middle", class: "cell-text", style: "fill: var(--dim-n); font-weight: 600" }, "grad " + n.grad));
            if (n.local) svg.appendChild(S("text", { x: n.x, y: ly, "text-anchor": "middle", class: "cell-text", style: "fill: var(--dim-n); font-size: 10px" }, n.local));
          }
        }
      });
      if (!forward) {
        code.innerHTML = `<span class="c"># three deliveries, two knobs</span><br>x = torch.tensor([1., 2., 3.]);  y = torch.tensor([12., 15., 17.])<br>w = torch.tensor(3.0, requires_grad=True);  b = torch.tensor(8.0, requires_grad=True)`;
        note.textContent = "Every operation on a tensor that requires a gradient is recorded as a node. Run forward to compute the values, then step backward to watch the slopes come out.";
      } else if (stage === 0) {
        code.innerHTML = `loss = ((w * x + b - y) ** 2).mean()   <span class="c"># tensor(${f(loss, 4)}, grad_fn=&lt;MeanBackward0&gt;)</span><br>w.grad, b.grad   <span class="c"># (None, None): nothing has flowed back yet</span>`;
        note.textContent = "Forward is just arithmetic. The grad_fn on the result is the proof that PyTorch kept notes. Now step backward.";
      } else {
        const done = stage === stages.length;
        code.innerHTML = `loss.backward()<br>w.grad   <span class="c"># ${revealed.has("w") ? "tensor(" + f(g_w, 4) + ")" : "..."}</span><br>b.grad   <span class="c"># ${revealed.has("b") ? "tensor(" + f(g_b, 4) + ")" : "..."}</span>${done ? `<br><span class="c"># by hand (Lesson 1): dw = 2·mean(e·x) = ${f(g_w, 4)},  db = 2·mean(e) = ${f(g_b, 4)}</span>` : ""}`;
        note.textContent = stages[stage - 1].text;
      }
    }
    el.querySelector("[data-fwd]").addEventListener("click", () => { forward = true; stage = 0; draw(); });
    el.querySelector("[data-bwd]").addEventListener("click", () => { if (!forward) forward = true; if (stage < stages.length) stage += 1; draw(); });
    el.querySelector("[data-reset]").addEventListener("click", () => { forward = false; stage = 0; draw(); });
    draw();
  };

  /* ---------- widget: gradients accumulate ---------- */
  FC.widgets.gradAccumulate = function (el) {
    // Lesson 1's data, 40 points, same seed as the regression widgets.
    function rng(seed) { return function () { seed |= 0; seed = (seed + 0x6D2B79F5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
    function gauss(r) { let u = 0, v = 0; while (u === 0) u = r(); while (v === 0) v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
    const r = rng(7), X = [], Y = [];
    for (let i = 0; i < 40; i++) { const d = 0.5 + 9.5 * r(); X.push(d); Y.push(8 + 3.5 * d + 2 * gauss(r)); }
    const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
    const grads = (w, b) => { const e = X.map((x, i) => w * x + b - Y[i]); return { dw: 2 * mean(e.map((v, i) => v * X[i])), db: 2 * mean(e), loss: mean(e.map((v) => v * v)) }; };
    function run(zero) {
      let w = 0, b = 0, gw = 0, gb = 0; const out = [];
      for (let s = 0; s < 60; s++) {
        const g = grads(w, b); out.push(g.loss);
        if (zero) { gw = g.dw; gb = g.db; } else { gw += g.dw; gb += g.db; }   // without zeroing, .grad keeps adding up
        w -= 0.005 * gw; b -= 0.005 * gb;
        if (!Number.isFinite(w)) { while (out.length < 60) out.push(Infinity); break; }
      }
      return out;
    }
    const good = run(true), bad = run(false);
    const single = grads(1, 2).dw;
    let count = 0;
    el.innerHTML = `
      <div class="widget-head"><h3>.grad adds up until you zero it</h3>
        <div class="presets btn-row"><button type="button" class="chip" data-bwd>loss.backward()</button><button type="button" class="chip" data-zero>w.grad.zero_()</button></div></div>
      <div class="widget-body">
        <div class="two-col">
          <div>
            <p class="readout" data-readout style="margin-top:0"></p>
            <svg data-bar role="img" aria-label="the size of w.grad after repeated backward calls"></svg>
          </div>
          <div>
            <p class="muted" style="margin:0 0 .4rem;font-size:.95rem">Sixty training steps on Lesson 1's data, learning rate 0.005:</p>
            <svg data-chart role="img" aria-label="loss per step with and without zeroing the gradient"></svg>
            <ul class="legend"><li><span class="swatch" style="background:var(--ink)"></span>zero_() every step: settles</li><li><span class="swatch" style="background:var(--bad)"></span>never zeroed: each step is the sum of all previous slopes</li></ul>
          </div>
        </div>
      </div>`;
    const readout = el.querySelector("[data-readout]"), bar = el.querySelector("[data-bar]"), chart = el.querySelector("[data-chart]");
    function drawBar() {
      while (bar.firstChild) bar.removeChild(bar.firstChild);
      bar.setAttribute("viewBox", "0 0 400 70");
      const total = single * count;
      const width = Math.min(360, (Math.abs(total) / Math.abs(single * 5)) * 360);
      bar.appendChild(S("rect", { x: 20, y: 20, width: 360, height: 22, rx: 4, style: "fill: var(--line-2)" }));
      if (count) bar.appendChild(S("rect", { x: 20, y: 20, width, height: 22, rx: 4, style: "fill: var(--dim-n)" }));
      bar.appendChild(S("text", { x: 20, y: 60, class: "cell-text", style: "fill: var(--ink-2)" }, count >= 5 ? "the bar is full: five slopes stacked on top of each other" : "same loss, same data, same slope each time"));
      readout.innerHTML = `w = torch.tensor(1.0, requires_grad=True)   <span class="c"># b = 2.0</span><br>backward called <b>${count}</b> time${count === 1 ? "" : "s"}, w.grad = <b class="dim-n">${count ? f(total, 1) : "None"}</b>${count > 1 ? `   <span class="c"># ${count} × ${f(single, 1)}</span>` : ""}`;
    }
    function drawChart() {
      while (chart.firstChild) chart.removeChild(chart.firstChild);
      const W = 420, H = 220, L = 46, R = 10, T = 10, B = 30;
      chart.setAttribute("viewBox", `0 0 ${W} ${H}`);
      const lo = 0, hi = 6;   // log10 scale from 1 to 1e6
      const sx = (i) => L + (i / 59) * (W - L - R), sy = (v) => T + (1 - (Math.min(hi, Math.max(lo, Math.log10(Math.max(v, 1)))) - lo) / (hi - lo)) * (H - T - B);
      [0, 2, 4, 6].forEach((p) => { chart.appendChild(S("line", { x1: L, y1: sy(10 ** p), x2: W - R, y2: sy(10 ** p), style: "stroke: var(--line-2)" })); chart.appendChild(S("text", { x: L - 6, y: sy(10 ** p) + 4, "text-anchor": "end", class: "cell-text", style: "fill: var(--ink-3)" }, p === 0 ? "1" : "1e" + p)); });
      chart.appendChild(S("text", { x: (L + W) / 2, y: H - 6, "text-anchor": "middle", class: "cell-text", style: "fill: var(--ink-2)" }, "step"));
      chart.appendChild(S("text", { x: L + 4, y: T + 10, class: "cell-text", style: "fill: var(--ink-2)" }, "loss, log scale"));
      const path = (arr, style) => chart.appendChild(S("path", { d: arr.map((v, i) => (i ? "L" : "M") + sx(i) + " " + sy(v)).join(" "), fill: "none", style }));
      path(bad, "stroke: var(--bad); stroke-width: 2");
      path(good, "stroke: var(--ink); stroke-width: 2");
    }
    el.querySelector("[data-bwd]").addEventListener("click", () => { if (count < 5) count += 1; drawBar(); });
    el.querySelector("[data-zero]").addEventListener("click", () => { count = 0; drawBar(); readout.innerHTML += `<br><span class="c"># w.grad is tensor(0.) again; the next backward starts fresh</span>`; });
    drawBar(); drawChart();
  };

  /* ---------- widget: X @ w, one row at a time ---------- */
  FC.widgets.matmulRows = function (el) {
    const names = ["km", "items", "rain", "rush"];
    const X = [[2, 1, 0, 1], [5.5, 3, 1, 0], [8, 2, 1, 1]];
    const w = [3.5, 1.5, 4, 6], b = 8;
    const preds = X.map((row) => row.reduce((s, v, j) => s + v * w[j], 0) + b);
    let sel = 0;
    el.innerHTML = `
      <div class="widget-head"><h3>A batch of weighted sums is one matrix multiply</h3>
        <div class="presets btn-row" data-rows></div></div>
      <div class="widget-body">
        <svg role="img" aria-label="X of shape 3 by 4 times w of shape 4 plus b gives predictions of shape 3"></svg>
        <p class="readout" data-readout></p>
      </div>`;
    const svg = el.querySelector("svg"), readout = el.querySelector("[data-readout]"), rowsBox = el.querySelector("[data-rows]");
    const CW = 54, CH = 30;
    function cell(x, y, text, style, strokeStyle) {
      svg.appendChild(S("rect", { x, y, width: CW, height: CH, rx: 3, style: `fill: var(--panel); stroke: ${strokeStyle || "var(--ink-3)"}; stroke-width: 1` }));
      svg.appendChild(S("text", { x: x + CW / 2, y: y + CH / 2 + 4, "text-anchor": "middle", class: "cell-text", style: style || "fill: var(--ink); font-size: 12px" }, text));
    }
    function draw() {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      svg.setAttribute("viewBox", "0 0 760 215");
      const x0 = 60, y0 = 50;
      names.forEach((n, j) => svg.appendChild(S("text", { x: x0 + j * CW + CW / 2, y: y0 - 8, "text-anchor": "middle", class: "cell-text", style: "fill: var(--dim-d); font-weight: 600" }, n)));
      X.forEach((row, i) => row.forEach((v, j) => cell(x0 + j * CW, y0 + i * CH, f(v, 1), i === sel ? "fill: var(--ink); font-weight: 600; font-size: 12px" : "fill: var(--ink-3); font-size: 12px", i === sel ? "var(--dim-n)" : "var(--line)")));
      svg.appendChild(S("text", { x: x0 + 2 * CW, y: y0 + 3 * CH + 22, "text-anchor": "middle", class: "axis-label", style: "fill: var(--ink-2)" }, "X"));
      svg.appendChild(S("text", { x: x0 + 2 * CW, y: y0 + 3 * CH + 38, "text-anchor": "middle", class: "cell-text" }, "")).innerHTML = `(<tspan style="fill: var(--dim-n)">3</tspan>, <tspan style="fill: var(--dim-d)">4</tspan>)`;
      const xw = x0 + 4 * CW + 40;
      svg.appendChild(S("text", { x: xw - 16, y: y0 + 2 * CH - 10, "text-anchor": "middle", class: "axis-label" }, "@"));
      w.forEach((v, j) => cell(xw, y0 + j * CH - 15, f(v, 1), "fill: var(--dim-w); font-weight: 600; font-size: 12px", "var(--dim-w)"));
      svg.appendChild(S("text", { x: xw + CW / 2, y: y0 + 4 * CH + 10, "text-anchor": "middle", class: "axis-label", style: "fill: var(--ink-2)" }, "w"));
      svg.appendChild(S("text", { x: xw + CW / 2, y: y0 + 4 * CH + 26, "text-anchor": "middle", class: "cell-text" }, "")).innerHTML = `(<tspan style="fill: var(--dim-d)">4</tspan>,)`;
      const xb = xw + CW + 40;
      svg.appendChild(S("text", { x: xb - 16, y: y0 + CH + 20, "text-anchor": "middle", class: "axis-label" }, "+"));
      cell(xb, y0 + CH, f(b), "fill: var(--dim-w); font-weight: 600; font-size: 12px", "var(--dim-w)");
      svg.appendChild(S("text", { x: xb + CW / 2, y: y0 + 2 * CH + 24, "text-anchor": "middle", class: "axis-label", style: "fill: var(--ink-2)" }, "b  ()"));
      const xp = xb + CW + 40;
      svg.appendChild(S("text", { x: xp - 16, y: y0 + CH + 20, "text-anchor": "middle", class: "axis-label" }, "="));
      preds.forEach((v, i) => cell(xp, y0 + i * CH, f(v, 2), i === sel ? "fill: var(--dim-y); font-weight: 600; font-size: 12px" : "fill: var(--ink-3); font-size: 12px", i === sel ? "var(--dim-y)" : "var(--line)"));
      svg.appendChild(S("text", { x: xp + CW / 2, y: y0 + 3 * CH + 22, "text-anchor": "middle", class: "axis-label", style: "fill: var(--ink-2)" }, "pred"));
      svg.appendChild(S("text", { x: xp + CW / 2, y: y0 + 3 * CH + 38, "text-anchor": "middle", class: "cell-text" }, "")).innerHTML = `(<tspan style="fill: var(--dim-n)">3</tspan>,)`;
      const row = X[sel];
      readout.innerHTML = `pred[${sel}] = ${row.map((v, j) => `<b>${f(v, 1)}</b>×<b class="dim-w">${f(w[j], 1)}</b>`).join(" + ")} + <b class="dim-w">${f(b)}</b> = <b class="dim-y">${f(preds[sel], 2)}</b> minutes<br><span class="c"># the same number as (X[${sel}] * w).sum() + b</span>`;
    }
    X.forEach((_, i) => {
      const btn = document.createElement("button"); btn.type = "button"; btn.className = "chip"; btn.textContent = `order ${i}`; btn.setAttribute("aria-pressed", String(i === 0));
      btn.addEventListener("click", () => { sel = i; rowsBox.querySelectorAll(".chip").forEach((c) => c.setAttribute("aria-pressed", "false")); btn.setAttribute("aria-pressed", "true"); draw(); });
      rowsBox.appendChild(btn);
    });
    draw();
  };
})();
