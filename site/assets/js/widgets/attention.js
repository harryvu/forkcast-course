/* Lesson 12 widgets: an attention matrix you can read, the sentence pair a bag cannot tell apart,
   and the encoder layer block by block. */
(function () {
  "use strict";
  const FC = window.FC;
  const S = FC.svgEl;

  /* ---------- widget: attention weights over a sentence ---------- */
  FC.widgets.attentionHeatmap = function (el) {
    const tokens = ["the", "pho", "was", "bad", ",", "not", "good", "."];
    // hand-made 4-d token vectors: [sentiment, negation, noun-ness, punctuation]
    const content = { the: [0, 0, 0.3, 0], pho: [0, 0, 1, 0], was: [0.1, 0, 0.2, 0], bad: [-1, 0.2, 0, 0], ",": [0, 0, 0, 1], not: [0, 1.2, 0, 0], good: [1, 0.2, 0, 0], ".": [0, 0, 0, 1] };
    const st = { positions: true, scale: true, query: 5 };
    el.innerHTML = `<div class="widget-head"><h3>Who looks at whom</h3>
        <div class="presets btn-row"><button type="button" class="chip" data-pos aria-pressed="true">positional encoding</button><button type="button" class="chip" data-scale aria-pressed="true">divide by √d</button></div></div>
      <div class="widget-body"><svg role="img" aria-label="attention weights between the tokens of a sentence"></svg><p class="readout" data-readout></p>
      <p class="muted" data-note style="max-width:68ch"></p></div>`;
    const svg = el.querySelector("svg"), readout = el.querySelector("[data-readout]"), note = el.querySelector("[data-note]");
    function vectors() {
      return tokens.map((t, i) => { const c = content[t].map((v) => 2.2 * v); if (st.positions) { c.push(1.4 * Math.sin(i / 2), 1.4 * Math.cos(i / 2)); } else { c.push(0, 0); } return c; });
    }
    function attention() {
      const v = vectors(), d = v[0].length, W = [];
      for (let i = 0; i < v.length; i++) {
        const scores = v.map((kj) => kj.reduce((s, x, n) => s + x * v[i][n], 0) / (st.scale ? Math.sqrt(d) : 1));
        const m = Math.max(...scores), ex = scores.map((s) => Math.exp(s - m)), Z = ex.reduce((s, x) => s + x, 0);
        W.push(ex.map((x) => x / Z));
      }
      return W;
    }
    function draw() {
      const W = attention(), n = tokens.length, C = 44, L = 70, T = 40;
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      svg.setAttribute("viewBox", `0 0 ${L + n * C + 230} ${T + n * C + 10}`);
      svg.appendChild(S("text", { x: L + (n * C) / 2, y: 14, "text-anchor": "middle", class: "cell-text", style: "fill: var(--ink-2)" }, "keys: the tokens being looked at →"));
      svg.appendChild(S("text", { x: 12, y: T + (n * C) / 2, "text-anchor": "middle", class: "cell-text", style: "fill: var(--ink-2)", transform: `rotate(-90 12 ${T + (n * C) / 2})` }, "queries: the token looking ↓"));
      tokens.forEach((t, j) => { svg.appendChild(S("text", { x: L + j * C + C / 2, y: T - 6, "text-anchor": "middle", class: "cell-text", style: "fill: var(--ink)" }, t)); svg.appendChild(S("text", { x: L - 6, y: T + j * C + C / 2 + 4, "text-anchor": "end", class: "cell-text", style: `fill: var(--ink); font-weight: ${j === st.query ? 700 : 400}` }, t)); });
      W.forEach((row, i) => row.forEach((w, j) => {
        svg.appendChild(S("rect", { x: L + j * C, y: T + i * C, width: C - 2, height: C - 2, rx: 3, style: `fill: var(--dim-n); fill-opacity: ${0.05 + 0.9 * w}; stroke: ${i === st.query ? "var(--ink)" : "none"}; cursor: pointer` })).addEventListener("click", () => { st.query = i; draw(); });
        svg.appendChild(S("text", { x: L + j * C + C / 2 - 1, y: T + i * C + C / 2 + 4, "text-anchor": "middle", class: "cell-text", style: `fill: ${w > 0.45 ? "var(--paper)" : "var(--ink)"}; font-size: 10px; pointer-events: none` }, w.toFixed(2)));
      }));
      const bx = L + n * C + 30, row = W[st.query];
      svg.appendChild(S("text", { x: bx, y: T - 6, class: "cell-text", style: "fill: var(--ink-2)" }, `what "${tokens[st.query]}" attends to`));
      row.forEach((w, j) => { svg.appendChild(S("rect", { x: bx, y: T + j * C + 10, width: 180 * w, height: 22, rx: 3, style: "fill: var(--dim-y)" })); svg.appendChild(S("text", { x: bx + 180 * w + 6, y: T + j * C + 26, class: "cell-text", style: "fill: var(--ink)" }, `${tokens[j]} ${w.toFixed(2)}`)); });
      readout.innerHTML = `weights = softmax(q @ k.T${st.scale ? " / sqrt(d)" : ""}, dim=-1)   <span class="c"># (1, 8, 8); row "${tokens[st.query]}" sums to ${row.reduce((s, x) => s + x, 0).toFixed(2)}</span><br>output[${st.query}] = Σ_j weights[${st.query}, j] · v[j]   <span class="c"># a mix of the value vectors, mostly ${tokens[row.indexOf(Math.max(...row))]}</span>`;
      note.textContent = !st.positions
        ? "Without positional encoding the matrix depends only on which tokens are present, not where they are: shuffle the sentence and every row stays the same. That is a bag of words with extra steps."
        : !st.scale ? "Without the √d division, dot products grow with the vector size and the softmax saturates: one weight near 1, the rest near 0, and the gradients through the softmax vanish. The scaling keeps the weights soft while training."
          : "Each row is a probability distribution: the query token's view of the sentence. Here \"not\" attends to itself and then to the sentiment words beside it, \"good\" and \"bad\", which is exactly the information a later layer needs to turn \"good\" into \"not good\". Hand-made vectors; a trained model learns its own.";
    }
    el.querySelector("[data-pos]").addEventListener("click", (e) => { st.positions = !st.positions; e.currentTarget.setAttribute("aria-pressed", String(st.positions)); draw(); });
    el.querySelector("[data-scale]").addEventListener("click", (e) => { st.scale = !st.scale; e.currentTarget.setAttribute("aria-pressed", String(st.scale)); draw(); });
    draw();
  };

  /* ---------- widget: the twin sentences ---------- */
  FC.widgets.orderDemo = function (el) {
    const a = "the pho was good , not bad .", b = "the pho was bad , not good .";
    const words = ["the", "pho", "was", "good", ",", "not", "bad", "."];
    el.innerHTML = `<div class="widget-head"><h3>Same words, opposite meaning</h3></div>
      <div class="widget-body"><svg role="img" aria-label="word counts of two sentences and what each model sees"></svg>
      <p class="muted" style="max-width:68ch">A bag of embeddings averages the token vectors, and the average of the same eight vectors is the same vector: both sentences produce identical input to the classifier, so it must get one of them wrong. Any model that pools before it looks at order has this ceiling. Attention with positional encoding lets each token's vector depend on its neighbours first, so "good" after "not" and "good" before "," become different vectors before the pooling.</p></div>`;
    const svg = el.querySelector("svg"); svg.setAttribute("viewBox", "0 0 760 190");
    [[a, 1, 30], [b, 0, 110]].forEach(([s, label, y]) => {
      svg.appendChild(S("text", { x: 10, y, class: "axis-label", style: `fill: ${label ? "var(--ok)" : "var(--bad)"}` }, `"${s}"  →  ${label ? "positive" : "negative"}`));
      words.forEach((w, j) => { const x = 10 + j * 56; svg.appendChild(S("rect", { x, y: y + 12, width: 50, height: 22, rx: 3, style: "fill: var(--dim-n); fill-opacity: .18; stroke: var(--dim-n)" })); svg.appendChild(S("text", { x: x + 25, y: y + 27, "text-anchor": "middle", class: "cell-text" }, `${w}: 1`)); });
      svg.appendChild(S("text", { x: 470, y: y + 27, class: "cell-text", style: "fill: var(--ink-2)" }, "bag of embeddings: identical pooled vector"));
      svg.appendChild(S("text", { x: 470, y: y + 45, class: "cell-text", style: "fill: var(--dim-y)" }, label ? "transformer: \"not\" ← \"bad\", \"good\" stands → positive" : "transformer: \"not\" ← \"good\", \"bad\" stands → negative"));
    });
    svg.appendChild(S("text", { x: 10, y: 180, class: "cell-text", style: "fill: var(--ink-3)" }, "measured in the sandbox: BagOfEmbeddings stays at 50% (loss ln 2 = 0.693), TinyTransformer reaches 100% in five epochs"));
  };

  /* ---------- widget: one encoder layer ---------- */
  FC.widgets.encoderStack = function (el) {
    const blocks = [
      { name: "input x", shape: "(N, T, d_model)", note: "Embedded tokens plus positional codes. Every block below keeps this shape, which is what lets you stack layers." },
      { name: "multi-head self-attention", shape: "(N, T, d_model)", note: "q, k, v are three Linear maps of x. nhead heads each attend over d_model / nhead dimensions in parallel, then their outputs are concatenated and mixed by one more Linear. Padded keys are masked out with src_key_padding_mask." },
      { name: "add & LayerNorm", shape: "(N, T, d_model)", note: "x + attention(x): the residual connection keeps the original token and adds what attention found. LayerNorm rescales each token vector to mean 0, variance 1, which keeps deep stacks trainable." },
      { name: "feed-forward", shape: "(N, T, d_model)", note: "Lesson 4's MLP, applied to every token position separately: Linear(d_model, dim_feedforward), ReLU, Linear back. This is where most of the layer's parameters live." },
      { name: "add & LayerNorm", shape: "(N, T, d_model)", note: "Second residual and normalisation. The output has the input's shape and goes to the next layer, or to the pooling and head." },
    ];
    let sel = 1, nhead = 2, d = 32;
    el.innerHTML = `<div class="widget-head"><h3>nn.TransformerEncoderLayer, block by block</h3>
        <div class="presets btn-row"><label class="muted" style="font-size:.9rem">d_model <select data-d><option>32</option><option>64</option><option>128</option></select></label><label class="muted" style="font-size:.9rem">nhead <select data-h><option>1</option><option selected>2</option><option>4</option><option>8</option></select></label></div></div>
      <div class="widget-body"><svg role="img" aria-label="the blocks of a Transformer encoder layer"></svg><p class="readout" data-readout></p><p class="muted" data-note style="max-width:68ch"></p></div>`;
    const svg = el.querySelector("svg"), readout = el.querySelector("[data-readout]"), note = el.querySelector("[data-note]");
    function params() { const ff = 2 * d; return { attn: 4 * d * d + 4 * d, ff: d * ff + ff + ff * d + d, norm: 4 * d }; }
    function draw() {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      svg.setAttribute("viewBox", "0 0 760 150");
      blocks.forEach((b, k) => {
        const x = 10 + k * 150, on = k === sel;
        svg.appendChild(S("rect", { x, y: 30, width: 138, height: 70, rx: 8, style: `fill: ${on ? "var(--dim-n)" : "var(--panel)"}; fill-opacity: ${on ? 0.18 : 1}; stroke: ${on ? "var(--dim-n)" : "var(--line)"}; stroke-width: ${on ? 2 : 1}; cursor: pointer` })).addEventListener("click", () => { sel = k; draw(); });
        svg.appendChild(S("text", { x: x + 69, y: 60, "text-anchor": "middle", class: "cell-text", style: "fill: var(--ink); font-weight: 600; pointer-events: none" }, b.name));
        svg.appendChild(S("text", { x: x + 69, y: 82, "text-anchor": "middle", class: "cell-text", style: "fill: var(--ink-2); pointer-events: none" }, b.shape));
        if (k < blocks.length - 1) svg.appendChild(S("text", { x: x + 144, y: 70, "text-anchor": "middle", class: "axis-label", style: "fill: var(--ink-3)" }, "→"));
      });
      svg.appendChild(S("path", { d: "M 79 100 C 79 135, 379 135, 379 100", fill: "none", style: "stroke: var(--dim-w); stroke-dasharray: 4 3" }));
      svg.appendChild(S("text", { x: 230, y: 143, "text-anchor": "middle", class: "cell-text", style: "fill: var(--dim-w)" }, "residual connections: the input is added back after each block"));
      const p = params();
      readout.innerHTML = `nn.TransformerEncoderLayer(d_model=${d}, nhead=${nhead}, dim_feedforward=${2 * d}, batch_first=True)   <span class="c"># head size ${d / nhead}; ${(p.attn + p.ff + p.norm).toLocaleString("en-US")} parameters: attention ${p.attn.toLocaleString("en-US")}, feed-forward ${p.ff.toLocaleString("en-US")}, norms ${p.norm}</span>`;
      note.textContent = blocks[sel].note + (sel === 1 ? ` With d_model=${d} and nhead=${nhead}, each head works in ${d / nhead} dimensions.` : "");
    }
    el.querySelector("[data-d]").addEventListener("change", (e) => { d = +e.target.value; if (d % nhead) nhead = 1; el.querySelector("[data-h]").value = String(nhead); draw(); });
    el.querySelector("[data-h]").addEventListener("change", (e) => { nhead = +e.target.value; if (d % nhead) { nhead = 1; e.target.value = "1"; } draw(); });
    draw();
  };
})();
