/* Regression widgets for Lesson 1. The data is generated here with a seeded
   PRNG from the same rule as forkcast.data.make_deliveries: minutes = 8 + 3.5 km + noise.
   Colours: data points in ink, the model's line in the outputs colour, w and b
   in the weights colour, residuals in the status red because they are errors. */
(function () {
  "use strict";
  const FC = window.FC;
  const S = FC.svgEl;
  const TRUE = { w: 3.5, b: 8 };

  function rng(seed) {
    return function () {
      seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gauss(r) {
    let u = 0, v = 0;
    while (u === 0) u = r();
    while (v === 0) v = r();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  function makeData(n, seed) {
    const r = rng(seed); const x = [], y = [];
    for (let i = 0; i < n; i++) { const d = 0.5 + 9.5 * r(); x.push(d); y.push(TRUE.b + TRUE.w * d + 2 * gauss(r)); }
    return { x, y };
  }
  const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
  function mse(d, w, b) { return mean(d.x.map((x, i) => (w * x + b - d.y[i]) ** 2)); }
  function grads(d, w, b) {
    const e = d.x.map((x, i) => w * x + b - d.y[i]);
    return { dw: 2 * mean(e.map((v, i) => v * d.x[i])), db: 2 * mean(e) };
  }
  function ols(d) {
    const mx = mean(d.x), my = mean(d.y);
    const w = mean(d.x.map((x, i) => (x - mx) * (d.y[i] - my))) / mean(d.x.map((x) => (x - mx) ** 2));
    return { w, b: my - w * mx };
  }
  const fmt = (v, n = 2) => (Number.isFinite(v) ? v.toFixed(n) : "∞");

  /* A small plotting helper: axes, ticks, mapping functions. */
  function plot(svg, box, range, titles) {
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    const { W, H } = box, L = 46, R = 12, T = 12, B = 34;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    const sx = (x) => L + ((x - range.x0) / (range.x1 - range.x0)) * (W - L - R);
    const sy = (y) => T + (1 - (y - range.y0) / (range.y1 - range.y0)) * (H - T - B);
    const g = S("g");
    g.appendChild(S("line", { x1: L, y1: sy(range.y0), x2: W - R, y2: sy(range.y0), style: "stroke: var(--line)" }));
    g.appendChild(S("line", { x1: L, y1: T, x2: L, y2: sy(range.y0), style: "stroke: var(--line)" }));
    const xt = range.xTicks || [], yt = range.yTicks || [];
    xt.forEach((v) => { g.appendChild(S("text", { x: sx(v), y: H - B + 16, class: "axis-label", style: "fill: var(--ink-3); font-weight: 400", "text-anchor": "middle" }, String(v))); });
    yt.forEach((v) => { g.appendChild(S("text", { x: L - 6, y: sy(v) + 4, class: "axis-label", style: "fill: var(--ink-3); font-weight: 400", "text-anchor": "end" }, String(v))); g.appendChild(S("line", { x1: L, y1: sy(v), x2: W - R, y2: sy(v), style: "stroke: var(--line-2)" })); });
    g.appendChild(S("text", { x: (L + W - R) / 2, y: H - 4, class: "axis-label", style: "fill: var(--ink-2); font-weight: 500", "text-anchor": "middle" }, titles.x));
    g.appendChild(S("text", { x: L + 6, y: T + 4, class: "axis-label", style: "fill: var(--ink-2); font-weight: 500" }, titles.y));
    svg.appendChild(g);
    const clampY = (y) => Math.max(range.y0, Math.min(range.y1, y));
    return { sx, sy, g, clampY, L, R, T, B, W, H };
  }
  function drawScatter(svg, d, w, b, opts) {
    const p = plot(svg, { W: 420, H: 300 }, { x0: 0, x1: 10.5, y0: 0, y1: 50, xTicks: [0, 2, 4, 6, 8, 10], yTicks: [10, 20, 30, 40, 50] }, { x: "distance (km)", y: "minutes" });
    if (opts && opts.residuals) d.x.forEach((x, i) => p.g.appendChild(S("line", { x1: p.sx(x), y1: p.sy(p.clampY(d.y[i])), x2: p.sx(x), y2: p.sy(p.clampY(w * x + b)), style: "stroke: var(--bad); stroke-width: 1.5; opacity: .7" })));
    d.x.forEach((x, i) => p.g.appendChild(S("circle", { cx: p.sx(x), cy: p.sy(p.clampY(d.y[i])), r: 3.5, style: "fill: var(--ink-3)" })));
    if (Number.isFinite(w) && Number.isFinite(b)) {
      const y0 = w * 0 + b, y1 = w * 10.5 + b;
      p.g.appendChild(S("line", { x1: p.sx(0), y1: p.sy(y0), x2: p.sx(10.5), y2: p.sy(y1), style: "stroke: var(--dim-y); stroke-width: 3", "stroke-linecap": "round", "clip-path": "inset(0)" }));
    }
  }
  const wb = (w, b) => `minutes = <b class="dim-w">${fmt(w)}</b> × km + <b class="dim-w">${fmt(b)}</b>`;

  /* ---------- widget: fit a line by hand ---------- */
  FC.widgets.fitLine = function (el) {
    const d = makeData(40, 7);
    const best = ols(d);
    el.innerHTML = `
      <div class="widget-head"><h3>Move the two knobs until the line fits</h3>
        <div class="presets btn-row"><button type="button" class="chip" data-reset>Reset</button><button type="button" class="chip" data-best>Show the best line</button></div></div>
      <div class="widget-body">
        <div class="two-col">
          <div><svg role="img" aria-label="delivery minutes against distance with a fitted line"></svg></div>
          <div>
            <div class="widget-controls" style="flex-direction:column;align-items:stretch;gap:.9rem">
              <label style="display:grid;grid-template-columns:auto 1fr auto;gap:.6rem;align-items:center"><span class="shape dim-w">w</span><input type="range" min="0" max="7" step="0.1" value="1" data-w aria-label="slope w"><span class="shape" data-wv>1.00</span></label>
              <label style="display:grid;grid-template-columns:auto 1fr auto;gap:.6rem;align-items:center"><span class="shape dim-w">b</span><input type="range" min="0" max="20" step="0.5" value="2" data-b aria-label="intercept b"><span class="shape" data-bv>2.00</span></label>
              <label><input type="checkbox" data-res checked> show the errors</label>
            </div>
            <p class="readout" data-readout></p>
            <p class="muted" style="font-size:.95rem;margin:.6rem 0 0">The red lines are the errors: how far each real delivery is from the line. The loss is the mean of their squares. Smaller is better; the best possible here is ${fmt(mse(d, best.w, best.b))}.</p>
          </div>
        </div>
      </div>`;
    const svg = el.querySelector("svg"), iw = el.querySelector("[data-w]"), ib = el.querySelector("[data-b]"), ires = el.querySelector("[data-res]");
    const readout = el.querySelector("[data-readout]");
    function render() {
      const w = +iw.value, b = +ib.value;
      el.querySelector("[data-wv]").textContent = fmt(w); el.querySelector("[data-bv]").textContent = fmt(b);
      drawScatter(svg, d, w, b, { residuals: ires.checked });
      const loss = mse(d, w, b);
      readout.innerHTML = `${wb(w, b)}<br>loss (mean squared error) = <b>${fmt(loss)}</b>`;
    }
    [iw, ib, ires].forEach((i) => i.addEventListener("input", render));
    el.querySelector("[data-reset]").addEventListener("click", () => { iw.value = 1; ib.value = 2; render(); });
    el.querySelector("[data-best]").addEventListener("click", () => { iw.value = Math.round(best.w * 10) / 10; ib.value = Math.round(best.b * 2) / 2; render(); });
    render();
  };

  /* ---------- widget: gradient descent on the loss curve ---------- */
  FC.widgets.descent = function (el) {
    const d = makeData(40, 7);
    const best = ols(d);
    const b = best.b;                      // b is held at its best value so the loss is a curve in w alone
    const lossAt = (w) => mse(d, w, b);
    const state = { w: 0, step: 0, lr: 0.01, diverged: false };
    el.innerHTML = `
      <div class="widget-head"><h3>Roll downhill: gradient descent on w</h3>
        <div class="presets btn-row"><button type="button" class="chip" data-step>Step once</button><button type="button" class="chip" data-run>Run 25 steps</button><button type="button" class="chip" data-reset>Reset</button></div></div>
      <div class="widget-body">
        <div class="widget-controls">
          <label style="display:grid;grid-template-columns:auto 1fr auto;gap:.6rem;align-items:center;min-width:280px">learning rate <input type="range" min="0.001" max="0.06" step="0.001" value="0.01" data-lr aria-label="learning rate"><span class="shape" data-lrv>0.010</span></label>
          <span class="muted" data-hint>Try 0.005, then 0.02, then 0.04.</span>
        </div>
        <div class="two-col">
          <div><svg role="img" aria-label="loss as a function of w, with the current position marked"></svg></div>
          <div><svg role="img" aria-label="the data with the line for the current w"></svg></div>
        </div>
        <p class="readout" data-readout></p>
        <p class="muted" data-note style="max-width:68ch"></p>
      </div>`;
    const [svgLoss, svgFit] = el.querySelectorAll("svg");
    const ilr = el.querySelector("[data-lr]"), readout = el.querySelector("[data-readout]"), note = el.querySelector("[data-note]");
    const W0 = -2, W1 = 9;
    const yMax = Math.ceil(Math.max(lossAt(W0), lossAt(W1)) / 100) * 100;

    function drawLoss() {
      const p = plot(svgLoss, { W: 420, H: 300 }, { x0: W0, x1: W1, y0: 0, y1: yMax, xTicks: [-2, 0, 2, 4, 6, 8], yTicks: [yMax / 2, yMax] }, { x: "w (minutes per km)", y: "loss" });
      let path = "";
      for (let i = 0; i <= 120; i++) { const w = W0 + ((W1 - W0) * i) / 120; path += (i ? " L" : "M") + p.sx(w) + " " + p.sy(Math.min(yMax, lossAt(w))); }
      p.g.appendChild(S("path", { d: path, fill: "none", style: "stroke: var(--ink); stroke-width: 2" }));
      p.g.appendChild(S("line", { x1: p.sx(best.w), y1: p.sy(0), x2: p.sx(best.w), y2: p.sy(yMax), style: "stroke: var(--dim-w); stroke-dasharray: 4 4" }));
      p.g.appendChild(S("text", { x: p.sx(best.w) + 5, y: p.sy(yMax) + 14, class: "axis-label", style: "fill: var(--dim-w)" }, `best w = ${fmt(best.w)}`));
      if (Number.isFinite(state.w)) {
        const wc = Math.max(W0, Math.min(W1, state.w));
        const loss = lossAt(state.w), g = grads(d, state.w, b).dw;
        const lc = Math.min(yMax, loss);
        // tangent: slope dL/dw, drawn over a span of 1.2 in w
        if (Math.abs(g) < yMax * 4) {
          const span = 0.6, y1 = Math.max(0, Math.min(yMax, loss - g * span)), y2 = Math.max(0, Math.min(yMax, loss + g * span));
          p.g.appendChild(S("line", { x1: p.sx(wc - span), y1: p.sy(y1), x2: p.sx(wc + span), y2: p.sy(y2), style: "stroke: var(--dim-n); stroke-width: 2", "stroke-linecap": "round" }));
        }
        p.g.appendChild(S("circle", { cx: p.sx(wc), cy: p.sy(lc), r: 7, style: "fill: var(--dim-y); stroke: var(--panel); stroke-width: 2" }));
      }
    }
    function render() {
      drawLoss();
      drawScatter(svgFit, d, state.w, b);
      const loss = lossAt(state.w), g = grads(d, state.w, b).dw;
      el.querySelector("[data-lrv]").textContent = (+ilr.value).toFixed(3);
      readout.innerHTML = `step ${state.step}: <span class="shape dim-w">w</span> = <b>${fmt(state.w, 3)}</b>, loss = <b>${fmt(loss, 1)}</b>, slope dL/dw = <b class="dim-n">${fmt(g, 1)}</b><br>next w = w − lr × slope = ${fmt(state.w, 3)} − ${(+ilr.value).toFixed(3)} × ${fmt(g, 1)} = <b>${fmt(state.w - ilr.value * g, 3)}</b>`;
      if (!Number.isFinite(state.w) || Math.abs(state.w) > 1e6) {
        note.innerHTML = `<span class="err">Diverged.</span> Each step overshoots the bottom and lands higher up the other wall. The slope there is steeper, so the next step is bigger still. Lower the learning rate and reset.`;
      } else if (Math.abs(g) < 0.05 && state.step > 0) {
        note.textContent = "The slope is almost zero: you are at the bottom. Notice that steps shrink on their own as the slope flattens, even though the learning rate never changed.";
      } else if (state.step === 0) {
        note.textContent = "The blue segment is the slope of the loss at the current w. Gradient descent moves w a small amount against the slope: downhill. The amount is the learning rate times the slope.";
      } else {
        note.textContent = Math.abs(ilr.value * 2 * mean(d.x.map((x) => x * x))) > 1.6 ? "The steps are too big: the ball jumps across the valley instead of rolling down it. Watch what happens if you keep going." : "Rolling. Keep stepping and watch the slope shrink as the ball nears the bottom.";
      }
    }
    function step() {
      if (!Number.isFinite(state.w) || Math.abs(state.w) > 1e6) return;
      const g = grads(d, state.w, b).dw;
      state.w = state.w - ilr.value * g;
      state.step += 1;
    }
    el.querySelector("[data-step]").addEventListener("click", () => { step(); render(); });
    el.querySelector("[data-run]").addEventListener("click", () => {
      const reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      if (reduce) { for (let i = 0; i < 25; i++) step(); render(); return; }
      let n = 0;
      const timer = setInterval(() => { step(); render(); if (++n >= 25) clearInterval(timer); }, 90);
    });
    el.querySelector("[data-reset]").addEventListener("click", () => { state.w = 0; state.step = 0; render(); });
    ilr.addEventListener("input", render);
    render();
  };
})();
