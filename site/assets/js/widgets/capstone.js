/* Lesson 16 widgets: the fusion model with modality switches and the measured ablation, a model-card builder,
   and a map of which lesson built which part of the final system. */
(function () {
  "use strict";
  const FC = window.FC;
  const S = FC.svgEl;

  /* ---------- widget: fusion lab ---------- */
  FC.widgets.fusionLab = function (el) {
    const ACC = { "image": 0.750, "text": 0.715, "tabular": 0.623, "image+text": 0.790, "image+tabular": 0.748, "text+tabular": 0.725, "image+text+tabular": 0.873 };
    const st = { image: true, text: true, tabular: true };
    el.innerHTML = `<div class="widget-head"><h3>Three encoders, one head: switch modalities off</h3>
        <div class="presets btn-row"><button type="button" class="chip" data-m="image" aria-pressed="true">image</button><button type="button" class="chip" data-m="text" aria-pressed="true">text</button><button type="button" class="chip" data-m="tabular" aria-pressed="true">tabular</button></div></div>
      <div class="widget-body"><svg role="img" aria-label="the fusion model's three branches, the concatenation and the head"></svg><p class="readout" data-readout></p>
      <p class="muted" style="max-width:68ch">Each branch turns its input into a 32-number vector; the three are concatenated into 96 numbers and classified. Switching a branch off replaces its vector with zeros, so the same class trains every ablation. The accuracies are the ones measured in the sandbox on the synthetic listings (12 epochs, 400 held-out listings, chance 0.5): no single modality gets past 0.75, and the three together reach 0.87, because the demand of a listing really does depend on all three.</p></div>`;
    const svg = el.querySelector("svg"), readout = el.querySelector("[data-readout]");
    function draw() {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      svg.setAttribute("viewBox", "0 0 760 230");
      const rows = [["image", "photo (3, 32, 32)", "conv blocks → Linear", "var(--dim-c)"], ["text", "description (T,) ids", "embedding → masked mean", "var(--dim-d)"], ["tabular", "price, prep, rating (3,)", "Linear → ReLU → Linear", "var(--dim-n)"]];
      rows.forEach(([m, inp, enc, col], r) => {
        const y = 20 + r * 62, on = st[m];
        svg.appendChild(S("rect", { x: 10, y, width: 190, height: 42, rx: 6, style: `fill: var(--panel); stroke: ${on ? col : "var(--line)"}; opacity: ${on ? 1 : 0.4}` })); svg.appendChild(S("text", { x: 105, y: y + 26, "text-anchor": "middle", class: "cell-text", style: "fill: var(--ink)" }, inp));
        svg.appendChild(S("text", { x: 215, y: y + 26, class: "axis-label", style: "fill: var(--ink-3)" }, "→"));
        svg.appendChild(S("rect", { x: 240, y, width: 200, height: 42, rx: 6, style: `fill: var(--panel); stroke: ${on ? col : "var(--line)"}; opacity: ${on ? 1 : 0.4}` })); svg.appendChild(S("text", { x: 340, y: y + 26, "text-anchor": "middle", class: "cell-text", style: "fill: var(--ink)" }, enc));
        svg.appendChild(S("line", { x1: 440, y1: y + 21, x2: 480, y2: 112, style: `stroke: ${on ? col : "var(--line)"}; stroke-width: 2; opacity: ${on ? 1 : 0.4}` }));
        svg.appendChild(S("text", { x: 452, y: y + 10, class: "cell-text", style: `fill: ${on ? col : "var(--ink-3)"}; font-size: 9.5px` }, on ? "(N, 32)" : "zeros (N, 32)"));
      });
      svg.appendChild(S("rect", { x: 480, y: 70, width: 90, height: 84, rx: 6, style: "fill: var(--dim-w); fill-opacity: .15; stroke: var(--dim-w)" })); svg.appendChild(S("text", { x: 525, y: 108, "text-anchor": "middle", class: "cell-text", style: "fill: var(--ink)" }, "cat →")); svg.appendChild(S("text", { x: 525, y: 124, "text-anchor": "middle", class: "cell-text", style: "fill: var(--ink)" }, "(N, 96)"));
      svg.appendChild(S("text", { x: 585, y: 116, class: "axis-label", style: "fill: var(--ink-3)" }, "→"));
      svg.appendChild(S("rect", { x: 610, y: 70, width: 140, height: 84, rx: 6, style: "fill: var(--panel); stroke: var(--dim-y)" })); svg.appendChild(S("text", { x: 680, y: 100, "text-anchor": "middle", class: "cell-text", style: "fill: var(--ink)" }, "Linear(96, 64)")); svg.appendChild(S("text", { x: 680, y: 116, "text-anchor": "middle", class: "cell-text", style: "fill: var(--ink)" }, "ReLU, Linear(64, 2)")); svg.appendChild(S("text", { x: 680, y: 138, "text-anchor": "middle", class: "cell-text", style: "fill: var(--dim-y)" }, "(N, 2) logits"));
      const key = ["image", "text", "tabular"].filter((m) => st[m]).join("+");
      const acc = ACC[key];
      readout.innerHTML = key ? `FusionModel(len(vocab), modalities=(${key.split("+").map((m) => `"${m}"`).join(", ")}))   <span class="c"># validation accuracy ${acc.toFixed(3)}${key === "image+text+tabular" ? ", the best" : acc < 0.76 ? "; add a modality" : ""}</span>` : `<span class="err">no inputs: the head sees only zeros and can do no better than guessing the majority class</span>`;
    }
    el.querySelectorAll("[data-m]").forEach((b) => b.addEventListener("click", () => { const m = b.getAttribute("data-m"); st[m] = !st[m]; b.setAttribute("aria-pressed", String(st[m])); draw(); }));
    draw();
  };

  /* ---------- widget: model card builder ---------- */
  FC.widgets.modelCardBuilder = function (el) {
    el.innerHTML = `<div class="widget-head"><h3>Write the model card as you go</h3></div>
      <div class="widget-body"><div class="two-col">
        <div class="setup-form">
          <label>Name <input type="text" data-f="name" value="Forkcast listing demand v1"></label>
          <label>Intended use <input type="text" data-f="use" value="Rank new restaurant listings for review priority. Not for setting prices."></label>
          <label>Training data <input type="text" data-f="data" value="1,600 synthetic listings (shapes, templated descriptions, simulated prices)."></label>
          <label>Fused accuracy <input type="text" data-f="acc" value="0.873"></label>
          <label>One limitation <input type="text" data-f="lim" value="Trained on synthetic data; real photos and reviews will shift the distribution."></label>
        </div>
        <pre class="no-copy" data-out style="margin:0;max-height:22rem;overflow:auto"></pre></div>
      <p class="muted" style="max-width:68ch;margin-top:.6rem">A model card travels with the artifact and answers the questions a colleague will have in six months: what is this for, what was it trained on, how good is it and on what, where does it fail, and which code produced it. <code>write_model_card</code> renders exactly these sections; the fields above are what you fill in.</p></div>`;
    const out = el.querySelector("[data-out]"), f = (k) => el.querySelector(`[data-f="${k}"]`).value;
    function draw() {
      out.textContent = `# ${f("name")}\n\n## Intended use\n\n${f("use")}\n\n## Training data\n\n${f("data")}\n\n## Evaluation\n\n| metric | value |\n|---|---|\n| accuracy (fused) | ${f("acc")} |\n| accuracy (best single modality) | 0.750 |\n\n## Limitations\n\n- ${f("lim")}\n\n## Provenance\n\n- Lesson 7: image encoder (conv blocks)\n- Lesson 11: text encoder (embedding, masked mean)\n- Lesson 4: tabular encoder (MLP)\n- Lesson 8: training loop v2\n- Lesson 15: artifact format and serving\n`;
    }
    el.querySelectorAll("[data-f]").forEach((i) => i.addEventListener("input", draw)); draw();
  };

  /* ---------- widget: the course as one system ---------- */
  FC.widgets.courseMap = function (el) {
    const parts = [
      ["tensors, shapes, broadcasting", "0", "every line of every forward pass"], ["gradient descent, autograd", "1, 2", "what fit_v2 does on every batch"], ["nn.Module, optimizers, normalization", "3", "the model classes and the Normalizer in the artifact"],
      ["hidden layers and ReLU", "4", "the tabular encoder and every head"], ["splits, early stopping, regularization", "5", "the validation accuracy you report"], ["Dataset, DataLoader, embeddings", "6", "the loaders and every lookup table"],
      ["convolutions", "7", "the image encoder"], ["devices, AMP, checkpoints", "8", "fit_v2, the loop that trained all of it"], ["pretrained backbones, confusion matrices", "9", "the real dish model, and how you evaluated it"],
      ["hooks, Grad-CAM, calibration", "10", "how you debugged the image branch"], ["tokens, padding, masks", "11", "the text encoder"], ["attention, Transformers", "12", "what replaces the text encoder at scale"],
      ["pretrained language models", "13", "the sentiment endpoint and menu search"], ["recommendations", "14", "the fourth service, user and item embeddings"], ["artifacts, export, FastAPI", "15", "how the capstone ships"],
    ];
    el.innerHTML = `<div class="widget-head"><h3>Where every part came from</h3></div>
      <div class="widget-body"><table class="bc-table" style="font-size:.9rem"><tr><th>idea</th><th>lesson</th><th>where it lives in the capstone</th></tr>${parts.map((p) => `<tr><th style="text-align:left;font-weight:400">${p[0]}</th><td>${p[1]}</td><th style="text-align:left;font-weight:400">${p[2]}</th></tr>`).join("")}</table></div>`;
  };
})();
