/* Lesson 13 widgets: subword pieces, gradient accumulation as a budget, and cosine search over a menu. */
(function () {
  "use strict";
  const FC = window.FC;
  const S = FC.svgEl;

  /* ---------- widget: WordPiece-style splitting ---------- */
  FC.widgets.subwordDemo = function (el) {
    const VOCAB = new Set(["[cls]", "[sep]", "the", "pizza", "ramen", "pho", "was", "cold", "soggy", "over", "##priced", "##cooked", "fork", "##cast", "fri", "##es", "deliver", "##y", "##ed", "late", "un", "##believ", "##ably", "good", "bad", "not", "fast", "slow", "egg", "##s", "noodle", "chicken", "spicy", "!", ",", ".", "fresh", "cheese", "##burger", "ham", "yester", "##day", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]);
    function wordpiece(word) {
      const pieces = []; let start = 0;
      while (start < word.length) {
        let end = word.length, found = null;
        while (end > start) { const cand = (start ? "##" : "") + word.slice(start, end); if (VOCAB.has(cand)) { found = cand; break; } end -= 1; }
        if (!found) return ["[UNK]"];
        pieces.push(found); start = end;
      }
      return pieces;
    }
    el.innerHTML = `<div class="widget-head"><h3>Subwords: no word is ever unknown</h3></div>
      <div class="widget-body"><div class="widget-controls"><label style="flex:1;min-width:260px">text <input type="text" data-text style="width:100%" value="Overpriced forkcast pizza, unbelievably soggy fries!"></label></div>
      <svg role="img" aria-label="words split into subword pieces"></svg><p class="readout" data-readout></p>
      <p class="muted" style="max-width:68ch">Lesson 11's vocabulary mapped every unseen word to one &lt;unk&gt; id. A subword tokenizer splits an unseen word into pieces it does know, longest match first; the ## prefix marks a piece that continues a word. Rare words cost a few extra tokens instead of their meaning, and the vocabulary stays small (30,522 for DistilBERT). A pretrained model's tokenizer is part of the model: use the one it shipped with.</p></div>`;
    const input = el.querySelector("[data-text]"), svg = el.querySelector("svg"), readout = el.querySelector("[data-readout]");
    function draw() {
      const words = input.value.toLowerCase().match(/[a-z]+|[!?.,]/g) || [];
      const all = [["[CLS]", "special"]];
      words.forEach((w) => wordpiece(w).forEach((p, i) => all.push([p, p === "[UNK]" ? "unk" : i ? "cont" : "start"])));
      all.push(["[SEP]", "special"]);
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      const perRow = 10, rows = Math.ceil(all.length / perRow); svg.setAttribute("viewBox", `0 0 760 ${12 + rows * 40}`);
      all.forEach(([p, kind], i) => {
        const x = 8 + (i % perRow) * 75, y = 8 + Math.floor(i / perRow) * 40;
        const stroke = kind === "special" ? "var(--ink-3)" : kind === "unk" ? "var(--bad)" : kind === "cont" ? "var(--dim-w)" : "var(--dim-d)";
        svg.appendChild(S("rect", { x, y, width: 70, height: 26, rx: 4, style: `fill: var(--panel); stroke: ${stroke}; stroke-width: ${kind === "cont" ? 1.5 : 1}` }));
        svg.appendChild(S("text", { x: x + 35, y: y + 17, "text-anchor": "middle", class: "cell-text", style: `fill: ${kind === "unk" ? "var(--bad)" : "var(--ink)"}` }, p.length > 10 ? p.slice(0, 9) + "…" : p));
      });
      readout.innerHTML = `tokenizer(text)["input_ids"]   <span class="c"># ${all.length} ids for ${words.length} words; ${all.filter((p) => p[1] === "cont").length} continuation pieces (##), ${all.filter((p) => p[1] === "unk").length} unknown</span>`;
    }
    input.addEventListener("input", draw); draw();
  };

  /* ---------- widget: gradient accumulation ---------- */
  FC.widgets.accumulationDemo = function (el) {
    let micro = 8, acc = 4;
    el.innerHTML = `<div class="widget-head"><h3>Gradient accumulation: a big batch in small pieces</h3></div>
      <div class="widget-body"><div class="widget-controls">
        <label>micro-batch <select data-m><option>4</option><option selected>8</option><option>16</option><option>32</option></select></label>
        <label>accumulation_steps <select data-a><option>1</option><option>2</option><option selected>4</option><option>8</option></select></label></div>
      <svg role="img" aria-label="forward and backward passes grouped into optimizer steps"></svg><p class="readout" data-readout></p>
      <p class="muted" style="max-width:68ch">Each forward and backward pass only has to hold one micro-batch's activations in GPU memory, while the gradients from several passes add up before a single optimizer step, exactly as Lesson 2 showed they would when nothing zeroes them. Dividing each loss by the number of steps keeps the sum equal to the mean over the whole effective batch. Memory scales with the micro-batch; the optimizer sees the effective batch.</p></div>`;
    const svg = el.querySelector("svg"), readout = el.querySelector("[data-readout]");
    function draw() {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      svg.setAttribute("viewBox", "0 0 760 130");
      const total = 8, w = 84;
      for (let i = 0; i < total; i++) {
        const x = 10 + i * (w + 8), stepEnd = (i + 1) % acc === 0 || i === total - 1;
        svg.appendChild(S("rect", { x, y: 20, width: w, height: 40, rx: 5, style: "fill: var(--dim-n); fill-opacity: .18; stroke: var(--dim-n)" }));
        svg.appendChild(S("text", { x: x + w / 2, y: 37, "text-anchor": "middle", class: "cell-text" }, `forward+backward`));
        svg.appendChild(S("text", { x: x + w / 2, y: 52, "text-anchor": "middle", class: "cell-text", style: "fill: var(--ink-2)" }, `${micro} reviews`));
        if (stepEnd) { svg.appendChild(S("rect", { x: x + w - 30, y: 70, width: 30, height: 22, rx: 4, style: "fill: var(--dim-y)" })); svg.appendChild(S("text", { x: x + w - 15, y: 85, "text-anchor": "middle", class: "cell-text", style: "fill: var(--paper); font-size: 9px" }, "step")); }
      }
      svg.appendChild(S("text", { x: 10, y: 115, class: "cell-text", style: "fill: var(--ink-2)" }, `peak activation memory ∝ ${micro}   |   effective batch ${micro * acc}   |   ${Math.ceil(total / acc)} optimizer steps for these ${total} micro-batches`));
      readout.innerHTML = `fit_lm(model, loader, ..., accumulation_steps=${acc})   <span class="c"># loader batch_size=${micro}: gradients of ${acc} micro-batch${acc > 1 ? "es" : ""} add up, then clip_and_step</span>`;
    }
    el.querySelector("[data-m]").addEventListener("change", (e) => { micro = +e.target.value; draw(); });
    el.querySelector("[data-a]").addEventListener("change", (e) => { acc = +e.target.value; draw(); });
    draw();
  };

  /* ---------- widget: cosine search over a menu ---------- */
  FC.widgets.cosineSearch = function (el) {
    const menu = [["spicy chicken ramen with soft egg", 20], ["beef pho with lime and herbs", 40], ["green curry with jasmine rice", 75], ["margherita pizza with fresh basil", 160], ["four cheese lasagna", 185], ["crispy fish tacos", 260], ["classic cheeseburger and fries", 230]];
    const queries = [["something spicy with noodles", 30], ["a hot soup", 45], ["italian comfort food", 172], ["crunchy and fried", 245]];
    let qi = 0;
    const vec = (deg) => [Math.cos((deg * Math.PI) / 180), Math.sin((deg * Math.PI) / 180)];
    el.innerHTML = `<div class="widget-head"><h3>Search by meaning: cosine similarity</h3><div class="presets btn-row" data-q></div></div>
      <div class="widget-body"><div class="two-col"><div><svg role="img" aria-label="menu items and a query as directions on a circle"></svg></div><div><svg data-bars role="img" aria-label="cosine similarity of each menu item to the query"></svg></div></div>
      <p class="readout" data-readout></p>
      <p class="muted" style="max-width:68ch">After mean pooling and normalisation every text is a unit vector, a direction. The cosine of the angle between two directions is their dot product: 1 for the same direction, 0 for unrelated, −1 for opposite. Search is one matrix-vector product over the whole menu, which is why a few hundred thousand dishes can be searched in milliseconds. Positions here are illustrative; the real vectors have 384 dimensions.</p></div>`;
    const svg = el.querySelector("svg"), bars = el.querySelector("[data-bars]"), readout = el.querySelector("[data-readout]"), qb = el.querySelector("[data-q]");
    function draw() {
      const [qtext, qdeg] = queries[qi], q = vec(qdeg);
      const scored = menu.map(([t, d]) => ({ t, d, s: vec(d)[0] * q[0] + vec(d)[1] * q[1] })).sort((a, b) => b.s - a.s);
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      svg.setAttribute("viewBox", "0 0 320 300"); const cx = 160, cy = 150, R = 110;
      svg.appendChild(S("circle", { cx, cy, r: R, style: "fill: none; stroke: var(--line)" }));
      menu.forEach(([t, d]) => { const v = vec(d), hit = scored.slice(0, 3).some((s) => s.t === t); svg.appendChild(S("line", { x1: cx, y1: cy, x2: cx + R * v[0], y2: cy - R * v[1], style: `stroke: ${hit ? "var(--dim-y)" : "var(--ink-3)"}; stroke-width: ${hit ? 2 : 1}` })); svg.appendChild(S("text", { x: cx + (R + 12) * v[0], y: cy - (R + 12) * v[1] + 4, "text-anchor": v[0] < -0.1 ? "end" : v[0] > 0.1 ? "start" : "middle", class: "cell-text", style: `fill: ${hit ? "var(--ink)" : "var(--ink-3)"}; font-size: 9.5px` }, t.split(" ").slice(0, 2).join(" "))); });
      svg.appendChild(S("line", { x1: cx, y1: cy, x2: cx + R * q[0], y2: cy - R * q[1], style: "stroke: var(--dim-n); stroke-width: 3", "stroke-linecap": "round" }));
      svg.appendChild(S("text", { x: cx + (R + 12) * q[0], y: cy - (R + 12) * q[1] + 4, "text-anchor": q[0] < -0.1 ? "end" : "start", class: "axis-label", style: "fill: var(--dim-n)" }, "query"));
      while (bars.firstChild) bars.removeChild(bars.firstChild);
      bars.setAttribute("viewBox", "0 0 380 300");
      bars.appendChild(S("text", { x: 10, y: 16, class: "cell-text", style: "fill: var(--ink-2)" }, "embeddings @ q, sorted"));
      scored.forEach((s, i) => { const y = 30 + i * 36; bars.appendChild(S("text", { x: 10, y: y + 12, class: "cell-text", style: `fill: var(--ink); font-weight: ${i < 3 ? 600 : 400}` }, s.t.length > 30 ? s.t.slice(0, 29) + "…" : s.t)); bars.appendChild(S("rect", { x: 200, y, width: Math.max(0, s.s) * 150, height: 16, rx: 3, style: `fill: ${i < 3 ? "var(--dim-y)" : "var(--ink-3)"}` })); bars.appendChild(S("text", { x: 206 + Math.max(0, s.s) * 150, y: y + 12, class: "cell-text" }, s.s.toFixed(2))); });
      readout.innerHTML = `search("${qtext}", menu, embeddings, encoder, tokenizer, k=3)   <span class="c"># ${scored.slice(0, 3).map((s) => `("${s.t.split(" ").slice(0, 3).join(" ")}…", ${s.s.toFixed(2)})`).join(", ")}</span>`;
    }
    queries.forEach(([t], i) => { const b = document.createElement("button"); b.type = "button"; b.className = "chip"; b.textContent = t; b.setAttribute("aria-pressed", String(i === 0)); b.addEventListener("click", () => { qi = i; qb.querySelectorAll(".chip").forEach((c) => c.setAttribute("aria-pressed", "false")); b.setAttribute("aria-pressed", "true"); draw(); }); qb.appendChild(b); });
    draw();
  };
})();
