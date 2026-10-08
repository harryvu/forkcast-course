/* Lesson 11 widgets: text becomes ids, ids become a padded batch with a mask, the batch becomes a prediction. */
(function () {
  "use strict";
  const FC = window.FC;
  const S = FC.svgEl;
  const TOKEN = /[a-z0-9']+|[!?.,]/g;
  const tokenize = (t) => (t.toLowerCase().match(TOKEN) || []);
  // a small vocabulary in the Lesson 11 order: <pad>, <unk>, then by frequency in the templated corpus
  const ITOS = ["<pad>", "<unk>", ".", "the", "was", "and", "delivery", "again", "hot", "cold", "rider", "late", "pad", "thai", "pizza", "ramen", "pho", "order", "will", "delicious", "fresh", "fantastic", "bland", "soggy", "stale", "early", "quick", "hour", "took", "over", "an", "never", "arrived", "star", "stars", "five", "one", "!", ",", "loved", "it", "still", "terrible", "tasted", "friendly", "rude", "refund", "value", "great", "recommend", "highly", "tiny", "greasy"];
  const STOI = Object.fromEntries(ITOS.map((t, i) => [t, i]));
  const encode = (toks) => toks.map((t) => (t in STOI ? STOI[t] : 1));

  /* ---------- widget: text -> tokens -> ids ---------- */
  FC.widgets.tokenizerDemo = function (el) {
    el.innerHTML = `<div class="widget-head"><h3>From a review to a tensor of ids</h3></div>
      <div class="widget-body">
        <div class="widget-controls"><label style="flex:1;min-width:260px">review <input type="text" data-text style="width:100%" value="The ramen was delicious, but the delivery took over an HOUR!"></label></div>
        <svg role="img" aria-label="tokens and their ids"></svg>
        <p class="readout" data-readout></p>
        <p class="muted" style="max-width:68ch">Tokenizing is just string handling: lower-case, split on words and punctuation. The vocabulary is a lookup table built from the training reviews, with two reserved ids: 0 for padding, 1 for any token the training set never contained. Those <code>&lt;unk&gt;</code> hits are information lost; a bigger training set, or subword tokens (Lesson 13), shrinks them.</p>
      </div>`;
    const input = el.querySelector("[data-text]"), svg = el.querySelector("svg"), readout = el.querySelector("[data-readout]");
    function draw() {
      const toks = tokenize(input.value), ids = encode(toks);
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      const perRow = 9, rows = Math.max(1, Math.ceil(toks.length / perRow)), W = 760, H = 20 + rows * 66;
      svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
      toks.forEach((t, i) => {
        const x = 10 + (i % perRow) * 82, y = 10 + Math.floor(i / perRow) * 66, unk = ids[i] === 1;
        svg.appendChild(S("rect", { x, y, width: 76, height: 26, rx: 4, style: `fill: var(--panel); stroke: ${unk ? "var(--bad)" : "var(--dim-d)"}` }));
        svg.appendChild(S("text", { x: x + 38, y: y + 17, "text-anchor": "middle", class: "cell-text", style: `fill: ${unk ? "var(--bad)" : "var(--ink)"}` }, t.length > 11 ? t.slice(0, 10) + "…" : t));
        svg.appendChild(S("text", { x: x + 38, y: y + 44, "text-anchor": "middle", class: "axis-label", style: "fill: var(--ink-3); font-size: 10px" }, "↓"));
        svg.appendChild(S("text", { x: x + 38, y: y + 60, "text-anchor": "middle", class: "cell-text", style: `fill: ${unk ? "var(--bad)" : "var(--dim-n)"}; font-weight: 600` }, String(ids[i])));
      });
      const unk = ids.filter((i) => i === 1).length;
      readout.innerHTML = `tokenize(text)   <span class="c"># ${toks.length} tokens</span><br>torch.tensor(vocab.encode(tokens))   <span class="c"># tensor([${ids.join(", ")}]), shape (${toks.length},)${unk ? `; ${unk} token${unk > 1 ? "s" : ""} not in the vocabulary → id 1` : ""}</span>`;
    }
    input.addEventListener("input", draw);
    draw();
  };

  /* ---------- widget: padding and the mask ---------- */
  FC.widgets.padMask = function (el) {
    const reviews = ["Loved it, will order again!", "Cold and late.", "The pho was fresh and the rider was friendly."];
    let showMask = true;
    el.innerHTML = `<div class="widget-head"><h3>Three reviews, one batch</h3><div class="presets btn-row"><button type="button" class="chip" data-toggle aria-pressed="true">show the mask</button></div></div>
      <div class="widget-body"><svg role="img" aria-label="three padded id sequences and their mask"></svg><p class="readout" data-readout></p>
      <p class="muted" style="max-width:68ch">Tensors are rectangular, reviews are not. <code>pad_sequence</code> fills the short ones with id 0 up to the longest in the batch, and the mask remembers which positions are real. Everything downstream has to honour it: the embedding row for id 0 is kept at zero with <code>padding_idx=0</code>, and the mean over tokens divides by the number of real tokens, not by T. Forget the mask and a short review's meaning is diluted by its padding.</p></div>`;
    const svg = el.querySelector("svg"), readout = el.querySelector("[data-readout]");
    function draw() {
      const seqs = reviews.map((r) => encode(tokenize(r))), T = Math.max(...seqs.map((s) => s.length));
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      svg.setAttribute("viewBox", `0 0 ${20 + T * 56} 150`);
      seqs.forEach((s, i) => {
        const y = 14 + i * 42;
        for (let t = 0; t < T; t++) {
          const real = t < s.length, v = real ? s[t] : 0, x = 10 + t * 56;
          svg.appendChild(S("rect", { x, y, width: 52, height: 26, rx: 4, style: `fill: ${real ? "var(--dim-n)" : "var(--line-2)"}; fill-opacity: ${real ? 0.18 : 1}; stroke: ${real ? "var(--dim-n)" : "var(--line)"}; stroke-dasharray: ${real ? "none" : "3 2"}` }));
          svg.appendChild(S("text", { x: x + 26, y: y + 17, "text-anchor": "middle", class: "cell-text", style: `fill: ${real ? "var(--ink)" : "var(--ink-3)"}` }, String(v)));
          if (showMask) svg.appendChild(S("text", { x: x + 26, y: y + 38, "text-anchor": "middle", class: "cell-text", style: `fill: ${real ? "var(--ok)" : "var(--bad)"}; font-size: 10px` }, real ? "1" : "0"));
        }
      });
      readout.innerHTML = `ids, mask = pad_batch([…])   <span class="c"># ids (3, ${T}) int64, mask (3, ${T}) bool</span><br>mask.sum(dim=1)   <span class="c"># tensor([${seqs.map((s) => s.length).join(", ")}]): the real lengths</span>`;
    }
    el.querySelector("[data-toggle]").addEventListener("click", (e) => { showMask = !showMask; e.currentTarget.setAttribute("aria-pressed", String(showMask)); draw(); });
    draw();
  };

  /* ---------- widget: the shape pipeline ---------- */
  FC.widgets.bagPipeline = function (el) {
    const st = { N: 32, T: 120, D: 64 };
    el.innerHTML = `<div class="widget-head"><h3>BagOfEmbeddings, shape by shape</h3></div>
      <div class="widget-body">
        <div class="widget-controls">
          <label>batch N <select data-n><option>8</option><option selected>32</option><option>64</option></select></label>
          <label>longest review T <select data-t><option>20</option><option selected>120</option><option>200</option></select></label>
          <label>embedding dim D <select data-d><option>16</option><option selected>64</option><option>300</option></select></label>
        </div>
        <svg role="img" aria-label="the tensor shapes through the model"></svg>
        <p class="readout" data-readout></p>
        <p class="muted" style="max-width:68ch">The batch is a table of ids, each id becomes a row of D numbers, the T rows of each review are averaged into one, and that one row is classified. Everything after the pooling is Lesson 4's MLP; everything before it is Lesson 6's embedding table with a vocabulary of words instead of taxi zones. The pooling is what makes it a "bag": word order is thrown away, which is the limitation Lesson 12 removes.</p>
      </div>`;
    const svg = el.querySelector("svg"), readout = el.querySelector("[data-readout]");
    const sp = (n, role) => `<tspan style="fill: var(--dim-${role})">${n}</tspan>`;
    function draw() {
      const { N, T, D } = st;
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      svg.setAttribute("viewBox", "0 0 760 150");
      const stages = [["ids", `(${sp(N, "n")}, ${sp(T, "s")})`, "int64"], ["embedding(ids)", `(${sp(N, "n")}, ${sp(T, "s")}, ${sp(D, "d")})`, "float32"], ["masked mean over T", `(${sp(N, "n")}, ${sp(D, "d")})`, ""], ["head", `(${sp(N, "n")}, ${sp(2, "y")})`, "logits"]];
      stages.forEach(([name, shape, note], k) => {
        const x = 10 + k * 190;
        svg.appendChild(S("rect", { x, y: 30, width: 170, height: 70, rx: 8, style: "fill: var(--panel); stroke: var(--line)" }));
        svg.appendChild(S("text", { x: x + 85, y: 56, "text-anchor": "middle", class: "axis-label", style: "fill: var(--ink)" }, name));
        const t = S("text", { x: x + 85, y: 80, "text-anchor": "middle", class: "cell-text", style: "font-size: 13px" }); t.innerHTML = shape; svg.appendChild(t);
        if (note) svg.appendChild(S("text", { x: x + 85, y: 118, "text-anchor": "middle", class: "cell-text", style: "fill: var(--ink-3)" }, note));
        if (k < 3) svg.appendChild(S("text", { x: x + 180, y: 70, "text-anchor": "middle", class: "axis-label", style: "fill: var(--ink-3)" }, "→"));
      });
      readout.innerHTML = `nn.Embedding(len(vocab), ${D}, padding_idx=0)   <span class="c"># weight (vocab, ${D}); a review of 7 real tokens padded to ${T} still averages 7 rows</span><br><span class="c"># numbers in the embedded batch: ${N} × ${T} × ${D} = ${(N * T * D).toLocaleString("en-US")}</span>`;
    }
    el.querySelector("[data-n]").addEventListener("change", (e) => { st.N = +e.target.value; draw(); });
    el.querySelector("[data-t]").addEventListener("change", (e) => { st.T = +e.target.value; draw(); });
    el.querySelector("[data-d]").addEventListener("change", (e) => { st.D = +e.target.value; draw(); });
    draw();
  };
})();
