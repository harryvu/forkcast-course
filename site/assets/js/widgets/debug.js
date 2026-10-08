/* Lesson 10 widgets: calibration you can distort, hooks you can attach, class weights you can read off. */
(function () {
  "use strict";
  const FC = window.FC;
  const S = FC.svgEl;
  function rng(seed) { return function () { seed |= 0; seed = (seed + 0x6D2B79F5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }

  /* ---------- widget: reliability diagram ---------- */
  FC.widgets.reliabilityDiagram = function (el) {
    const q = rng(10), N = 1500, base = [];
    for (let i = 0; i < N; i++) { const p = 0.5 + 0.5 * Math.pow(q(), 0.6); base.push({ p, correct: q() < p }); }   // calibrated to start with
    let shift = 0.0;
    el.innerHTML = `<div class="widget-head"><h3>Is 90% confident right 90% of the time?</h3></div>
      <div class="widget-body">
        <div class="widget-controls"><label style="display:grid;grid-template-columns:auto 1fr auto;gap:.6rem;align-items:center;min-width:320px">overconfidence <input type="range" min="-0.2" max="0.4" step="0.02" value="0" data-shift aria-label="overconfidence"><span class="shape" data-sv>+0.00</span></label><span class="muted">drag right: the model claims more than it delivers</span></div>
        <svg role="img" aria-label="reliability diagram: accuracy per confidence bin against the diagonal"></svg>
        <p class="readout" data-readout></p>
        <p class="muted" data-note style="max-width:68ch"></p>
      </div>`;
    const svg = el.querySelector("svg"), readout = el.querySelector("[data-readout]"), note = el.querySelector("[data-note]");
    function draw() {
      const preds = base.map((b) => ({ p: Math.min(0.999, Math.max(0.01, b.p + shift)), correct: b.correct }));
      const bins = Array.from({ length: 10 }, () => ({ conf: 0, acc: 0, n: 0 }));
      preds.forEach((b) => { const k = Math.min(9, Math.floor(b.p * 10)); bins[k].conf += b.p; bins[k].acc += b.correct ? 1 : 0; bins[k].n += 1; });
      let ece = 0; bins.forEach((b) => { if (b.n) { b.conf /= b.n; b.acc /= b.n; ece += (b.n / N) * Math.abs(b.acc - b.conf); } });
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      const W = 520, H = 320, L = 50, R = 14, T = 14, B = 40, size = Math.min(W - L - R, H - T - B);
      svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
      const sx = (v) => L + v * size, sy = (v) => T + (1 - v) * size;
      svg.appendChild(S("line", { x1: sx(0), y1: sy(0), x2: sx(1), y2: sy(1), style: "stroke: var(--ink-3); stroke-dasharray: 5 4" }));
      bins.forEach((b, k) => {
        if (!b.n) return;
        const x0 = sx(k / 10) + 2, w = size / 10 - 4;
        svg.appendChild(S("rect", { x: x0, y: sy(b.acc), width: w, height: sy(0) - sy(b.acc), style: "fill: var(--dim-n); fill-opacity: .55" }));
        const gapTop = Math.max(b.acc, b.conf), gapBottom = Math.min(b.acc, b.conf);
        svg.appendChild(S("rect", { x: x0 + w * 0.3, y: sy(gapTop), width: w * 0.4, height: sy(gapBottom) - sy(gapTop), style: `fill: ${b.conf > b.acc ? "var(--bad)" : "var(--ok)"}; fill-opacity: .55` }));
        svg.appendChild(S("text", { x: x0 + w / 2, y: sy(0) + 14, "text-anchor": "middle", class: "cell-text", style: "fill: var(--ink-3); font-size: 9px" }, String(b.n)));
      });
      [0, 0.5, 1].forEach((v) => { svg.appendChild(S("text", { x: sx(v), y: H - 10, "text-anchor": "middle", class: "cell-text", style: "fill: var(--ink-3)" }, v.toFixed(1))); svg.appendChild(S("text", { x: L - 6, y: sy(v) + 4, "text-anchor": "end", class: "cell-text", style: "fill: var(--ink-3)" }, v.toFixed(1))); });
      svg.appendChild(S("text", { x: sx(0.5), y: H - 26, "text-anchor": "middle", class: "cell-text", style: "fill: var(--ink-2)" }, "confidence (max softmax probability), 10 bins; small numbers are bin counts"));
      svg.appendChild(S("text", { x: 14, y: sy(0.5), "text-anchor": "middle", class: "cell-text", style: "fill: var(--ink-2)", transform: `rotate(-90 14 ${sy(0.5)})` }, "accuracy in the bin"));
      readout.innerHTML = `expected_calibration_error = <b class="${ece > 0.08 ? "err" : "ok"}">${ece.toFixed(3)}</b>   <span class="c"># weighted mean of |accuracy − confidence| over the bins; red = the model promised more than it delivered</span>`;
      note.textContent = shift > 0.05 ? "Overconfident: bars fall below the diagonal. A Forkcast dish check that is 95% sure of a wrong label will overrule a restaurant owner who is right. Fixes: label smoothing, temperature scaling on the validation set, or simply reporting a probability the user can read as the gamble it is." : shift < -0.05 ? "Underconfident: bars above the diagonal. The model is right more often than it says. Harmless to safety, wasteful in practice: a dish the model gets right 90% of the time should not be sent for manual review as a coin toss." : "Well calibrated: each bin's accuracy sits on the diagonal. The probabilities mean what they say, which is what lets you build thresholds on them.";
    }
    el.querySelector("[data-shift]").addEventListener("input", (e) => { shift = +e.target.value; el.querySelector("[data-sv]").textContent = (shift >= 0 ? "+" : "") + shift.toFixed(2); draw(); });
    draw();
  };

  /* ---------- widget: hooks on SmallCNN ---------- */
  FC.widgets.hookShapes = function (el) {
    const layers = [["0", "Conv2d(3, 16)", [16, 32, 32]], ["1", "BatchNorm2d(16)", [16, 32, 32]], ["2", "ReLU", [16, 32, 32]], ["3", "MaxPool2d(2)", [16, 16, 16]], ["4", "Conv2d(16, 32)", [32, 16, 16]], ["5", "BatchNorm2d(32)", [32, 16, 16]], ["6", "ReLU", [32, 16, 16]], ["7", "MaxPool2d(2)", [32, 8, 8]]];
    let sel = 6;
    el.innerHTML = `<div class="widget-head"><h3>Attach a forward hook to a layer of SmallCNN</h3></div>
      <div class="widget-body"><svg role="img" aria-label="the eight layers of SmallCNN's feature extractor"></svg><p class="readout" data-readout></p>
      <p class="muted" style="max-width:68ch">A forward hook is a function the layer calls after it runs, with the module, its inputs and its output. Register one and you can read any intermediate tensor without changing the model's code; remove it when you are done or it keeps firing. Grad-CAM hooks a late layer because its grid is small enough to be a heatmap and its channels already mean something.</p></div>`;
    const svg = el.querySelector("svg"), readout = el.querySelector("[data-readout]");
    function draw() {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      svg.setAttribute("viewBox", "0 0 760 150");
      svg.appendChild(S("text", { x: 10, y: 16, class: "cell-text", style: "fill: var(--ink-2)" }, "x (8, 3, 32, 32) →   model.features[i]   → head → logits (8, 4). Click a layer."));
      layers.forEach(([i, name, shape], k) => {
        const x = 10 + k * 93, on = k === sel, h = 30 + shape[1] * 1.4;
        svg.appendChild(S("rect", { x, y: 120 - h, width: 84, height: h, rx: 5, style: `fill: ${on ? "var(--dim-n)" : "var(--panel)"}; fill-opacity: ${on ? 0.2 : 1}; stroke: ${on ? "var(--dim-n)" : "var(--line)"}; stroke-width: ${on ? 2 : 1}; cursor: pointer` })).addEventListener("click", () => { sel = k; draw(); });
        svg.appendChild(S("text", { x: x + 42, y: 134, "text-anchor": "middle", class: "cell-text", style: "fill: var(--ink); pointer-events: none" }, `[${i}] ${name.split("(")[0]}`));
        svg.appendChild(S("text", { x: x + 42, y: 120 - h - 6, "text-anchor": "middle", class: "cell-text", style: "fill: var(--ink-2); font-size: 9.5px; pointer-events: none" }, `(8, ${shape.join(", ")})`));
      });
      const [i, name, shape] = layers[sel];
      readout.innerHTML = `handle = model.features[${i}].register_forward_hook(hook)   <span class="c"># hook(module, inputs, output)</span><br>model(x)   <span class="c"># output seen by the hook: torch.Size([8, ${shape.join(", ")}])</span><br>handle.remove()<br><span class="c"># GradCAM(model, model.features[${i}]) would give heatmaps of ${shape[1]}×${shape[2]}, from ${shape[0]} channels${sel === 6 ? "; this is the layer the lesson uses" : sel <= 2 ? "; too early: edges and colours, not evidence for a class" : ""}</span>`;
    }
    draw();
  };

  /* ---------- widget: class balance and weights ---------- */
  FC.widgets.classBalance = function (el) {
    const classes = ["pizza", "ramen", "pho", "sashimi", "bibimbap", "laksa"], counts = [400, 350, 300, 80, 60, 40];
    const total = counts.reduce((s, v) => s + v, 0), weights = counts.map((c) => total / (classes.length * c));
    el.innerHTML = `<div class="widget-head"><h3>Class balance, and the weights that fix it</h3></div>
      <div class="widget-body"><svg role="img" aria-label="training images per class and the resulting loss weights"></svg>
      <p class="readout">weights = class_weights(class_counts(labels, 6))   <span class="c"># total / (n_classes × count): every class now carries the same total weight</span><br>loss_fn = nn.CrossEntropyLoss(weight=weights)</p>
      <p class="muted" style="max-width:68ch">With 400 pizzas and 40 laksas, a model that never predicts laksa loses almost nothing. Inverse-frequency weights make one laksa mistake cost as much as ten pizza mistakes, so the rare class is worth learning. Check per-class accuracy before and after: the common classes give up a little, the rare ones gain a lot.</p></div>`;
    const svg = el.querySelector("svg"); svg.setAttribute("viewBox", "0 0 760 190");
    svg.appendChild(S("text", { x: 10, y: 16, class: "cell-text", style: "fill: var(--ink-2)" }, "training images per class"));
    svg.appendChild(S("text", { x: 400, y: 16, class: "cell-text", style: "fill: var(--ink-2)" }, "loss weight per class (count × weight = 205 for every class)"));
    classes.forEach((c, i) => {
      const y = 30 + i * 25;
      svg.appendChild(S("text", { x: 70, y: y + 13, "text-anchor": "end", class: "cell-text", style: "fill: var(--ink)" }, c));
      svg.appendChild(S("rect", { x: 78, y, width: counts[i] * 0.6, height: 18, rx: 3, style: "fill: var(--ink-3)" }));
      svg.appendChild(S("text", { x: 84 + counts[i] * 0.6, y: y + 13, class: "cell-text", style: "fill: var(--ink)" }, String(counts[i])));
      svg.appendChild(S("text", { x: 460, y: y + 13, "text-anchor": "end", class: "cell-text", style: "fill: var(--ink)" }, c));
      svg.appendChild(S("rect", { x: 468, y, width: weights[i] * 50, height: 18, rx: 3, style: "fill: var(--dim-w)" }));
      svg.appendChild(S("text", { x: 474 + weights[i] * 50, y: y + 13, class: "cell-text", style: "fill: var(--ink)" }, weights[i].toFixed(2)));
    });
  };

  /* ---------- widget: an error gallery as a table ---------- */
  FC.widgets.errorGallery = function (el) {
    const rows = [[354, "cross", "circle", 0.896], [94, "cross", "circle", 0.827], [163, "cross", "circle", 0.812], [6, "cross", "circle", 0.769], [150, "cross", "circle", 0.756], [211, "triangle", "square", 0.61], [77, "square", "cross", 0.52], [302, "circle", "cross", 0.41]];
    el.innerHTML = `<div class="widget-head"><h3>misclassified(logits, y, k=8): most confident first</h3></div>
      <div class="widget-body"><table class="bc-table" style="font-size:.92rem"><tr><th>index</th><th>true</th><th>predicted</th><th>confidence</th><th>what it tells you</th></tr>${rows.map((r) => `<tr><td>${r[0]}</td><td>${r[1]}</td><td>${r[2]}</td><td class="${r[3] > 0.7 ? "clash" : r[3] > 0.5 ? "stretch" : ""}">${r[3].toFixed(3)}</td><th style="text-align:left;font-weight:400">${r[3] > 0.7 ? "sure and wrong: a systematic blind spot, or a mislabelled image" : r[3] > 0.5 ? "a hard case; look at it" : "a coin toss: expected noise"}</th></tr>`).join("")}</table>
      <p class="muted" style="max-width:68ch;margin:.6rem 0 0">These are the Lesson 7 network's mistakes on 400 held-out shapes (92.5% accuracy). Five of the top five are crosses called circles: not random noise, one specific confusion. Show those five images with <code>gallery()</code> and the cause is usually visible within a minute. The confident mistakes are where your time goes; the coin tosses are where the noise lives.</p></div>`;
  };
})();
