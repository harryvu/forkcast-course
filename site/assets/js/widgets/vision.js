/* Lesson 7 widgets: a kernel sliding over pixels, pooling, and what cross-entropy punishes. */
(function () {
  "use strict";
  const FC = window.FC;
  const S = FC.svgEl;
  const f = (v, n = 1) => (Number.isFinite(v) ? v.toFixed(n) : "∞");

  /* ---------- widget: convolution, one position at a time ---------- */
  FC.widgets.convSlide = function (el) {
    // an 8x8 single-channel image: a bright vertical bar on a dark background, plus a dot
    const IMG = Array.from({ length: 8 }, (_, i) => Array.from({ length: 8 }, (_, j) => (j === 3 || j === 4 ? 9 : 0) + (i === 1 && j === 6 ? 6 : 0)));
    const KERNELS = {
      vertical: { name: "vertical edge", k: [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]] },
      horizontal: { name: "horizontal edge", k: [[-1, -1, -1], [0, 0, 0], [1, 1, 1]] },
      blur: { name: "blur (mean)", k: [[1, 1, 1], [1, 1, 1], [1, 1, 1]].map((r) => r.map((v) => +(v / 9).toFixed(2))) },
      identity: { name: "identity", k: [[0, 0, 0], [0, 1, 0], [0, 0, 0]] },
    };
    const st = { kernel: "vertical", stride: 1, pad: 0, pos: 0 };
    el.innerHTML = `
      <div class="widget-head"><h3>A 3×3 kernel slides over an 8×8 image</h3>
        <div class="presets btn-row"><button type="button" class="chip" data-step>slide one step</button><button type="button" class="chip" data-reset>back to start</button></div></div>
      <div class="widget-body">
        <div class="widget-controls">
          <label>kernel <select data-k>${Object.entries(KERNELS).map(([id, K]) => `<option value="${id}">${K.name}</option>`).join("")}</select></label>
          <label>stride <select data-s><option>1</option><option>2</option></select></label>
          <label>padding <select data-p><option>0</option><option>1</option></select></label>
        </div>
        <svg role="img" aria-label="input image, kernel and output feature map"></svg>
        <p class="readout" data-readout></p>
        <p class="muted" data-note style="max-width:68ch"></p>
      </div>`;
    const svg = el.querySelector("svg"), readout = el.querySelector("[data-readout]"), note = el.querySelector("[data-note]");
    function padded() { const n = 8 + 2 * st.pad, P = Array.from({ length: n }, () => new Array(n).fill(0)); for (let i = 0; i < 8; i++) for (let j = 0; j < 8; j++) P[i + st.pad][j + st.pad] = IMG[i][j]; return P; }
    function outSize() { return Math.floor((8 + 2 * st.pad - 3) / st.stride) + 1; }
    function conv() { const P = padded(), n = outSize(), K = KERNELS[st.kernel].k, O = []; for (let a = 0; a < n; a++) { O.push([]); for (let b = 0; b < n; b++) { let s = 0; for (let u = 0; u < 3; u++) for (let v = 0; v < 3; v++) s += P[a * st.stride + u][b * st.stride + v] * K[u][v]; O[a].push(s); } } return O; }
    function draw() {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      svg.setAttribute("viewBox", "0 0 760 250");
      const P = padded(), n = outSize(), O = conv(), total = n * n;
      st.pos = Math.min(st.pos, total - 1);
      const a = Math.floor(st.pos / n), b = st.pos % n, C = 22;
      const x0 = 20, y0 = 30, np = P.length;
      svg.appendChild(S("text", { x: x0, y: 18, class: "cell-text", style: "fill: var(--ink-2)" }, `input (1, ${np}, ${np})${st.pad ? " with zero padding" : ""}`));
      for (let i = 0; i < np; i++) for (let j = 0; j < np; j++) {
        const v = P[i][j], inWin = i >= a * st.stride && i < a * st.stride + 3 && j >= b * st.stride && j < b * st.stride + 3;
        const isPad = st.pad && (i === 0 || j === 0 || i === np - 1 || j === np - 1);
        svg.appendChild(S("rect", { x: x0 + j * C, y: y0 + i * C, width: C - 1, height: C - 1, style: `fill: var(--ink); fill-opacity: ${0.05 + 0.85 * (v / 9)}; stroke: ${inWin ? "var(--dim-n)" : isPad ? "var(--line)" : "none"}; stroke-width: ${inWin ? 2 : 1}; stroke-dasharray: ${isPad && !inWin ? "3 2" : "none"}` }));
        svg.appendChild(S("text", { x: x0 + j * C + C / 2, y: y0 + i * C + C / 2 + 4, "text-anchor": "middle", class: "cell-text", style: `fill: ${v > 4 ? "var(--paper)" : "var(--ink-2)"}; font-size: 10px` }, String(v)));
      }
      const kx = x0 + np * C + 40, ky = y0 + 40, K = KERNELS[st.kernel].k;
      svg.appendChild(S("text", { x: kx, y: ky - 12, class: "cell-text", style: "fill: var(--dim-w)" }, "kernel (3, 3): the weights"));
      for (let u = 0; u < 3; u++) for (let v = 0; v < 3; v++) { svg.appendChild(S("rect", { x: kx + v * 34, y: ky + u * 34, width: 33, height: 33, rx: 3, style: "fill: var(--panel); stroke: var(--dim-w)" })); svg.appendChild(S("text", { x: kx + v * 34 + 16, y: ky + u * 34 + 21, "text-anchor": "middle", class: "cell-text", style: "fill: var(--dim-w); font-weight: 600" }, String(K[u][v]))); }
      const ox = kx + 3 * 34 + 50, oC = Math.min(C, Math.floor(180 / n));
      svg.appendChild(S("text", { x: ox, y: 18, class: "cell-text", style: "fill: var(--dim-y)" }, `output (1, ${n}, ${n})`));
      const mx = Math.max(1, ...O.flat().map(Math.abs));
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
        const v = O[i][j], done = i * n + j <= st.pos;
        svg.appendChild(S("rect", { x: ox + j * oC, y: y0 + i * oC, width: oC - 1, height: oC - 1, style: `fill: ${done ? (v >= 0 ? "var(--dim-y)" : "var(--dim-n)") : "var(--line-2)"}; fill-opacity: ${done ? 0.15 + 0.8 * Math.abs(v) / mx : 1}; stroke: ${i === a && j === b ? "var(--ink)" : "none"}; stroke-width: 2` }));
        if (done && oC >= 18) svg.appendChild(S("text", { x: ox + j * oC + oC / 2, y: y0 + i * oC + oC / 2 + 4, "text-anchor": "middle", class: "cell-text", style: "fill: var(--ink); font-size: 10px" }, String(Math.round(v))));
      }
      const terms = []; for (let u = 0; u < 3; u++) for (let v = 0; v < 3; v++) if (K[u][v] !== 0 && P[a * st.stride + u][b * st.stride + v] !== 0) terms.push(`${P[a * st.stride + u][b * st.stride + v]}×${K[u][v]}`);
      readout.innerHTML = `output[${a}, ${b}] = ${terms.length ? terms.join(" + ") : "0 (every product is zero)"} = <b>${f(O[a][b], 0)}</b><br><span class="c">size: floor((8 + 2×${st.pad} − 3) / ${st.stride}) + 1 = ${n}      nn.Conv2d(1, 1, kernel_size=3, stride=${st.stride}, padding=${st.pad})</span>`;
      note.textContent = st.kernel === "vertical" ? "The same nine weights are applied at every position: that is what makes a convolution cheap and translation-aware. This kernel responds where brightness changes left to right, so it lights up on both edges of the bar and ignores the flat middle." : st.kernel === "horizontal" ? "The same picture, different weights: now the top-to-bottom changes matter, and the vertical bar is invisible. A conv layer learns many kernels at once, one per output channel." : st.kernel === "blur" ? "Averaging the neighbourhood smooths noise and softens edges. Early layers of trained networks usually contain a few kernels like this and many like the edge detectors." : "Only the centre weight is 1: the output copies the input. Padding of 1 with a 3×3 kernel keeps the size, which is why padding=1 is the default choice in this course.";
    }
    el.querySelector("[data-step]").addEventListener("click", () => { st.pos = (st.pos + 1) % (outSize() ** 2); draw(); });
    el.querySelector("[data-reset]").addEventListener("click", () => { st.pos = 0; draw(); });
    el.querySelector("[data-k]").addEventListener("change", (e) => { st.kernel = e.target.value; draw(); });
    el.querySelector("[data-s]").addEventListener("change", (e) => { st.stride = +e.target.value; st.pos = 0; draw(); });
    el.querySelector("[data-p]").addEventListener("change", (e) => { st.pad = +e.target.value; st.pos = 0; draw(); });
    draw();
  };

  /* ---------- widget: pooling ---------- */
  FC.widgets.poolDemo = function (el) {
    const A = [[1, 3, 2, 0], [4, 8, 1, 2], [0, 2, 7, 5], [1, 1, 3, 6]];
    el.innerHTML = `<div class="widget-head"><h3>MaxPool2d(2): keep the strongest response in each 2×2 patch</h3></div>
      <div class="widget-body"><svg role="img" aria-label="a 4 by 4 map pooled to 2 by 2 by max and by mean"></svg>
      <p class="muted" style="max-width:68ch;margin:.6rem 0 0">Pooling has no weights. It halves height and width, so the next layer's kernels cover twice as much of the original image, and it makes the network indifferent to exactly which pixel an edge sat on. Max is the usual choice between layers; average pooling appears at the very end of modern networks to collapse a map to one number per channel.</p></div>`;
    const svg = el.querySelector("svg"); svg.setAttribute("viewBox", "0 0 760 150");
    const C = 30;
    const grid = (x, y, M, title, hl) => {
      svg.appendChild(S("text", { x, y: y - 8, class: "cell-text", style: "fill: var(--ink-2)" }, title));
      M.forEach((row, i) => row.forEach((v, j) => {
        const q = hl ? ((Math.floor(i / 2) * 2 + Math.floor(j / 2)) % 4) : 0;
        svg.appendChild(S("rect", { x: x + j * C, y: y + i * C, width: C - 1, height: C - 1, style: `fill: var(--dim-n); fill-opacity: ${hl ? 0.08 + 0.14 * q : 0.08 + 0.07 * v}; stroke: var(--line)` }));
        svg.appendChild(S("text", { x: x + j * C + C / 2, y: y + i * C + C / 2 + 4, "text-anchor": "middle", class: "cell-text", style: `fill: var(--ink); font-weight: ${hl && v === Math.max(...[A[Math.floor(i/2)*2][Math.floor(j/2)*2], A[Math.floor(i/2)*2][Math.floor(j/2)*2+1], A[Math.floor(i/2)*2+1][Math.floor(j/2)*2], A[Math.floor(i/2)*2+1][Math.floor(j/2)*2+1]]) ? 700 : 400}` }, String(Number.isInteger(v) ? v : v.toFixed(2))));
      }));
    };
    grid(20, 30, A, "input (4, 4): one channel's responses", true);
    const mx = [[0, 0], [0, 0]], mean = [[0, 0], [0, 0]];
    for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) { const p = [A[2 * i][2 * j], A[2 * i][2 * j + 1], A[2 * i + 1][2 * j], A[2 * i + 1][2 * j + 1]]; mx[i][j] = Math.max(...p); mean[i][j] = p.reduce((s, v) => s + v, 0) / 4; }
    svg.appendChild(S("text", { x: 170, y: 95, class: "axis-label" }, "→"));
    grid(210, 30, mx, "MaxPool2d(2) → (2, 2)", false);
    grid(330, 30, mean, "AvgPool2d(2) → (2, 2)", false);
    svg.appendChild(S("text", { x: 470, y: 60, class: "cell-text", style: "fill: var(--ink-2)" }, "each 2×2 patch (shaded) becomes one number"));
    svg.appendChild(S("text", { x: 470, y: 80, class: "cell-text", style: "fill: var(--ink-2)" }, "32×32 → 16×16 → 8×8 after the two pools in SmallCNN"));
  };

  /* ---------- widget: softmax and cross-entropy ---------- */
  FC.widgets.softmaxCE = function (el) {
    const classes = ["circle", "square", "triangle", "cross"];
    const st = { logits: [2.0, 0.5, -1.0, 0.0], truth: 0 };
    el.innerHTML = `
      <div class="widget-head"><h3>From logits to a loss</h3><div class="presets btn-row" data-truth></div></div>
      <div class="widget-body">
        <div class="two-col">
          <div>${classes.map((c, i) => `<label style="display:grid;grid-template-columns:4.5rem 1fr 3rem;gap:.6rem;align-items:center;margin:.3rem 0"><span class="shape dim-y">${c}</span><input type="range" min="-5" max="5" step="0.1" value="${st.logits[i]}" data-l="${i}" aria-label="logit for ${c}"><span class="shape" data-lv="${i}">${st.logits[i].toFixed(1)}</span></label>`).join("")}
          <p class="muted" style="font-size:.9rem;margin:.6rem 0 0">logits: the four raw outputs of the last Linear layer, any real numbers</p></div>
          <div><svg role="img" aria-label="probabilities after softmax and the loss for the true class"></svg></div>
        </div>
        <p class="readout" data-readout></p>
      </div>`;
    const svg = el.querySelector("svg"), readout = el.querySelector("[data-readout]"), tb = el.querySelector("[data-truth]");
    function draw() {
      const m = Math.max(...st.logits), ex = st.logits.map((v) => Math.exp(v - m)), Z = ex.reduce((s, v) => s + v, 0), p = ex.map((v) => v / Z), loss = -Math.log(p[st.truth]);
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      svg.setAttribute("viewBox", "0 0 380 190");
      svg.appendChild(S("text", { x: 10, y: 14, class: "cell-text", style: "fill: var(--ink-2)" }, "softmax(logits): probabilities, they sum to 1"));
      p.forEach((v, i) => {
        const y = 26 + i * 34, truth = i === st.truth;
        svg.appendChild(S("text", { x: 70, y: y + 15, "text-anchor": "end", class: "cell-text", style: `fill: var(--ink); font-weight: ${truth ? 700 : 400}` }, classes[i] + (truth ? " ✓" : "")));
        svg.appendChild(S("rect", { x: 80, y, width: 240 * v, height: 22, rx: 3, style: `fill: ${truth ? "var(--dim-y)" : "var(--ink-3)"}` }));
        svg.appendChild(S("text", { x: 86 + 240 * v, y: y + 15, class: "cell-text", style: "fill: var(--ink)" }, v.toFixed(3)));
      });
      svg.appendChild(S("text", { x: 10, y: 180, class: "axis-label", style: `fill: ${loss > 1.5 ? "var(--bad)" : "var(--ok)"}` }, `loss = −log(${p[st.truth].toFixed(3)}) = ${loss.toFixed(3)}`));
      readout.innerHTML = `true class: <b class="dim-y">${classes[st.truth]}</b>   <span class="c"># nn.CrossEntropyLoss()(logits, y) = −log p[true]; chance level is −log(1/4) = 1.386</span><br><span class="c"># ${p[st.truth] > 0.9 ? "confident and right: almost no loss, almost no gradient" : p[st.truth] < 0.1 ? "confident and wrong: the loss grows without bound, and so does the push to fix it" : "unsure: a moderate loss; the gradient pushes the true logit up and the others down"}</span>`;
    }
    classes.forEach((c, i) => { const b = document.createElement("button"); b.type = "button"; b.className = "chip"; b.textContent = "truth: " + c; b.setAttribute("aria-pressed", String(i === 0)); b.addEventListener("click", () => { st.truth = i; tb.querySelectorAll(".chip").forEach((x) => x.setAttribute("aria-pressed", "false")); b.setAttribute("aria-pressed", "true"); draw(); }); tb.appendChild(b); });
    el.querySelectorAll("[data-l]").forEach((inp) => inp.addEventListener("input", () => { const i = +inp.getAttribute("data-l"); st.logits[i] = +inp.value; el.querySelector(`[data-lv="${i}"]`).textContent = st.logits[i].toFixed(1); draw(); }));
    draw();
  };
})();
