/* Lesson 9 widgets: what you freeze and what you train, how to read a confusion matrix,
   and what top-k accuracy forgives. Parameter counts are ResNet-18's real ones. */
(function () {
  "use strict";
  const FC = window.FC;
  const S = FC.svgEl;
  const fmt = (n) => n.toLocaleString("en-US");

  /* ---------- widget: the stages of ResNet-18 ---------- */
  FC.widgets.featureHierarchy = function (el) {
    const stages = [
      { name: "stem (conv1, bn1)", params: 9536, sees: "edges, colour blobs", block: null },
      { name: "layer1", params: 147968, sees: "textures, simple patterns", block: "layer1" },
      { name: "layer2", params: 525568, sees: "parts: rims, grains, leaves", block: "layer2" },
      { name: "layer3", params: 2099712, sees: "object parts, arrangements", block: "layer3" },
      { name: "layer4", params: 8393728, sees: "whole objects, scenes", block: "layer4" },
      { name: "avgpool + fc (new head)", params: 51813, sees: "our 101 dishes", block: "fc" },
    ];
    const order = ["layer4", "layer3", "layer2", "layer1"];
    let k = 0;
    el.innerHTML = `
      <div class="widget-head"><h3>ResNet-18: what to freeze, what to train</h3></div>
      <div class="widget-body">
        <div class="widget-controls"><label style="display:grid;grid-template-columns:auto 1fr auto;gap:.6rem;align-items:center;min-width:300px">trainable_blocks <input type="range" min="0" max="4" step="1" value="0" data-k aria-label="trainable blocks"><span class="shape" data-kv>0</span></label></div>
        <svg role="img" aria-label="the stages of ResNet-18 with their parameter counts, frozen or trainable"></svg>
        <p class="readout" data-readout></p>
        <p class="muted" data-note style="max-width:68ch"></p>
      </div>`;
    const svg = el.querySelector("svg"), readout = el.querySelector("[data-readout]"), note = el.querySelector("[data-note]");
    function draw() {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      svg.setAttribute("viewBox", "0 0 760 250");
      const unfrozen = new Set(order.slice(0, k).concat(["fc"]));
      const maxP = Math.log10(8393728);
      let trainable = 0, x = 20;
      stages.forEach((s, i) => {
        const on = unfrozen.has(s.block), w = 108, h = 30 + 100 * (Math.log10(s.params) / maxP);
        if (on) trainable += s.params;
        svg.appendChild(S("rect", { x, y: 170 - h, width: w, height: h, rx: 6, style: `fill: ${on ? "var(--dim-w)" : "var(--line)"}; stroke: ${on ? "var(--dim-w)" : "var(--ink-3)"}` }));
        svg.appendChild(S("text", { x: x + w / 2, y: 166 - h, "text-anchor": "middle", class: "cell-text", style: `fill: ${on ? "var(--dim-w)" : "var(--ink-2)"}; font-weight: 600` }, fmt(s.params)));
        svg.appendChild(S("text", { x: x + w / 2, y: 188, "text-anchor": "middle", class: "cell-text", style: "fill: var(--ink)" }, s.name.split(" (")[0]));
        svg.appendChild(S("text", { x: x + w / 2, y: 204, "text-anchor": "middle", class: "cell-text", style: "fill: var(--ink-3); font-size: 9.5px" }, s.sees));
        svg.appendChild(S("text", { x: x + w / 2, y: 222, "text-anchor": "middle", class: "cell-text", style: `fill: ${on ? "var(--dim-w)" : "var(--ink-3)"}; font-size: 10px` }, on ? "trains" : "frozen"));
        if (i < stages.length - 1) svg.appendChild(S("text", { x: x + w + 8, y: 150, class: "axis-label", style: "fill: var(--ink-3)" }, "→"));
        x += w + 16;
      });
      svg.appendChild(S("text", { x: 20, y: 18, class: "cell-text", style: "fill: var(--ink-2)" }, "parameters per stage (bar height is logarithmic); image flows left to right, 224×224 in, 101 logits out"));
      readout.innerHTML = `build_transfer_model(101, trainable_blocks=${k})   <span class="c"># ${fmt(trainable)} trainable of 11,228,325 parameters</span>${k ? `<br>param_groups(model, lr_head=1e-3, lr_backbone=1e-4)   <span class="c"># the unfrozen blocks move ten times more carefully than the new head</span>` : ""}`;
      note.textContent = k === 0
        ? "Only the new head trains: 51,813 numbers. The backbone is a fixed feature extractor, so its output for each image never changes, and you can compute it once and train the head on the cached 512-d vectors in seconds."
        : k <= 2 ? "The top blocks respond to whole objects and object parts, which is where ImageNet's photos and our dish photos differ most, so they are the first to benefit from adapting. The low-level edge and texture detectors are the same for any natural image and stay frozen." : "Unfreezing nearly everything turns this into ordinary training with a good initialisation. It needs more data than we have and a small learning rate everywhere; with a large one the pretrained features are overwritten in the first epoch and the advantage is gone.";
    }
    el.querySelector("[data-k]").addEventListener("input", (e) => { k = +e.target.value; el.querySelector("[data-kv]").textContent = String(k); draw(); });
    draw();
  };

  /* ---------- widget: a confusion matrix you can click ---------- */
  FC.widgets.confusionExplorer = function (el) {
    const classes = ["pizza", "lasagna", "ramen", "pho", "sushi", "sashimi"];
    const cm = [[44, 4, 0, 0, 1, 1], [6, 41, 0, 1, 1, 1], [0, 0, 36, 13, 1, 0], [0, 1, 11, 37, 1, 0], [0, 0, 1, 0, 33, 16], [1, 0, 0, 0, 18, 31]];
    let sel = [2, 3];
    el.innerHTML = `<div class="widget-head"><h3>Reading a confusion matrix</h3></div>
      <div class="widget-body"><div class="two-col"><div><svg data-cm role="img" aria-label="a six by six confusion matrix"></svg></div><div><svg data-bars role="img" aria-label="per-class accuracy"></svg><p class="readout" data-readout></p></div></div>
      <p class="muted" style="max-width:68ch">Fifty validation images per dish. Overall accuracy is one number, 74%, and it hides the story: pizza and lasagna are nearly solved, while ramen and pho are confused with each other a quarter of the time and sushi and sashimi a third. The next step is not "more training"; it is looking at those pairs. Illustrative counts.</p></div>`;
    const svgCm = el.querySelector("[data-cm]"), svgBars = el.querySelector("[data-bars]"), readout = el.querySelector("[data-readout]");
    const C = 44, L = 70, T = 60;
    function draw() {
      while (svgCm.firstChild) svgCm.removeChild(svgCm.firstChild);
      svgCm.setAttribute("viewBox", `0 0 ${L + 6 * C + 10} ${T + 6 * C + 10}`);
      svgCm.appendChild(S("text", { x: L + 3 * C, y: 14, "text-anchor": "middle", class: "cell-text", style: "fill: var(--ink-2)" }, "predicted →"));
      svgCm.appendChild(S("text", { x: 14, y: T + 3 * C, "text-anchor": "middle", class: "cell-text", style: "fill: var(--ink-2)", transform: `rotate(-90 14 ${T + 3 * C})` }, "true ↓"));
      classes.forEach((c, j) => { svgCm.appendChild(S("text", { x: L + j * C + C / 2, y: T - 8, "text-anchor": "middle", class: "cell-text", style: "fill: var(--ink-2)", transform: `rotate(-35 ${L + j * C + C / 2} ${T - 8})` }, c)); svgCm.appendChild(S("text", { x: L - 6, y: T + j * C + C / 2 + 4, "text-anchor": "end", class: "cell-text", style: "fill: var(--ink-2)" }, c)); });
      cm.forEach((row, i) => row.forEach((v, j) => {
        const isSel = i === sel[0] && j === sel[1];
        svgCm.appendChild(S("rect", { x: L + j * C, y: T + i * C, width: C - 2, height: C - 2, rx: 3, style: `fill: ${i === j ? "var(--dim-y)" : "var(--dim-n)"}; fill-opacity: ${0.06 + 0.9 * (v / 50)}; stroke: ${isSel ? "var(--ink)" : "none"}; stroke-width: 2; cursor: pointer` })).addEventListener("click", () => { sel = [i, j]; draw(); });
        svgCm.appendChild(S("text", { x: L + j * C + C / 2 - 1, y: T + i * C + C / 2 + 3, "text-anchor": "middle", class: "cell-text", style: `fill: var(--ink); pointer-events: none; font-weight: ${isSel ? 700 : 400}` }, String(v)));
      }));
      while (svgBars.firstChild) svgBars.removeChild(svgBars.firstChild);
      svgBars.setAttribute("viewBox", "0 0 340 190");
      svgBars.appendChild(S("text", { x: 10, y: 14, class: "cell-text", style: "fill: var(--ink-2)" }, "per-class accuracy: diagonal / row sum"));
      classes.forEach((c, i) => { const acc = cm[i][i] / cm[i].reduce((s, v) => s + v, 0), y = 28 + i * 26; svgBars.appendChild(S("text", { x: 60, y: y + 13, "text-anchor": "end", class: "cell-text", style: "fill: var(--ink)" }, c)); svgBars.appendChild(S("rect", { x: 68, y, width: 220 * acc, height: 18, rx: 3, style: `fill: ${acc > 0.8 ? "var(--dim-y)" : "var(--dim-n)"}` })); svgBars.appendChild(S("text", { x: 74 + 220 * acc, y: y + 13, class: "cell-text", style: "fill: var(--ink)" }, (100 * acc).toFixed(0) + "%")); });
      const [i, j] = sel, v = cm[i][j];
      readout.innerHTML = i === j ? `cm[${i}, ${i}] = ${v}: <b>${v} of 50 ${classes[i]}</b> images were recognised.` : `cm[${i}, ${j}] = ${v}: <b>${v} ${classes[i]}</b> images were called <b>${classes[j]}</b>.${v >= 10 ? ` <span class="c"># most_confused() would list this pair</span>` : ""}`;
    }
    draw();
  };

  /* ---------- widget: top-k accuracy ---------- */
  FC.widgets.topkDemo = function (el) {
    const classes = ["pho", "ramen", "udon", "laksa", "tom yum", "miso soup", "bibimbap", "pad thai"];
    const examples = [
      { truth: 0, scores: [3.1, 2.9, 1.2, 0.4, 0.1, -0.5, -1, -1.5] },
      { truth: 1, scores: [2.8, 2.2, 1.9, 0.3, 0.0, -0.2, -0.9, -1.1] },
      { truth: 3, scores: [1.0, 0.8, 0.5, 2.4, 1.9, -0.3, -0.8, -1.0] },
      { truth: 2, scores: [2.6, 2.4, 1.1, 0.9, 0.7, -0.1, -0.6, -1.2] },
      { truth: 6, scores: [0.2, 0.1, -0.3, 0.0, 0.4, 0.1, 2.9, 1.1] },
      { truth: 7, scores: [0.5, 0.4, 0.2, 1.8, 1.2, -0.1, 0.9, 2.5] },
    ];
    let k = 1;
    el.innerHTML = `<div class="widget-head"><h3>Top-k accuracy</h3></div>
      <div class="widget-body"><div class="widget-controls"><label style="display:grid;grid-template-columns:auto 1fr auto;gap:.6rem;align-items:center;min-width:260px">k <input type="range" min="1" max="5" step="1" value="1" data-k aria-label="k"><span class="shape" data-kv>1</span></label></div>
      <svg role="img" aria-label="six validation images with their ranked predictions"></svg><p class="readout" data-readout></p>
      <p class="muted" style="max-width:68ch">Each row is one validation photo; its predictions are sorted by logit, true class marked. Top-1 asks whether the first guess is right; top-5 asks whether the truth is anywhere in the first five. With 101 dishes and many near-twins (pho, ramen, udon), top-5 is the number papers quote, and the gap between the two tells you how often the model is close but not exact.</p></div>`;
    const svg = el.querySelector("svg"), readout = el.querySelector("[data-readout]");
    function draw() {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      svg.setAttribute("viewBox", "0 0 760 200");
      let hits = 0;
      examples.forEach((ex, r) => {
        const ranked = ex.scores.map((s, c) => [s, c]).sort((a, b) => b[0] - a[0]).map((p) => p[1]);
        const rank = ranked.indexOf(ex.truth) + 1, hit = rank <= k; if (hit) hits += 1;
        const y = 10 + r * 30;
        svg.appendChild(S("text", { x: 10, y: y + 14, class: "cell-text", style: `fill: ${hit ? "var(--ok)" : "var(--bad)"}; font-weight: 600` }, hit ? "✓" : "✗"));
        svg.appendChild(S("text", { x: 28, y: y + 14, class: "cell-text", style: "fill: var(--ink-2)" }, `true: ${classes[ex.truth]}`));
        ranked.forEach((c, pos) => {
          const x = 150 + pos * 74, isTruth = c === ex.truth, inK = pos < k;
          svg.appendChild(S("rect", { x, y, width: 70, height: 20, rx: 3, style: `fill: ${inK ? "var(--dim-n)" : "var(--line-2)"}; fill-opacity: ${inK ? 0.18 : 1}; stroke: ${isTruth ? "var(--dim-y)" : "none"}; stroke-width: 2` }));
          svg.appendChild(S("text", { x: x + 35, y: y + 14, "text-anchor": "middle", class: "cell-text", style: `fill: var(--ink); font-size: 9.5px; font-weight: ${isTruth ? 700 : 400}` }, classes[c]));
        });
      });
      readout.innerHTML = `top-${k} accuracy on these six: <b>${hits}/6 = ${(100 * hits / 6).toFixed(0)}%</b>   <span class="c"># top_k_accuracy(logits, y, k=${k})</span>`;
    }
    el.querySelector("[data-k]").addEventListener("input", (e) => { k = +e.target.value; el.querySelector("[data-kv]").textContent = String(k); draw(); });
    draw();
  };
})();
