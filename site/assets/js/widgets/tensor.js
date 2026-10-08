/* Tensor widgets. Used in Lesson 0 and reused whenever a later lesson shows
   a shape. Pure SVG, no dependencies. Colours come from the dim-* tokens. */
(function () {
  "use strict";
  const FC = window.FC;
  const S = FC.svgEl;

  const CELL = 16, GAP = 2, PITCH = CELL + GAP, OFF = 6;
  const MAX = { cols: 8, rows: 8, depth: 4, groups: 3 };

  const DEFAULT_ROLES = { 0: [], 1: ["d"], 2: ["n", "d"], 3: ["n", "s", "d"], 4: ["n", "c", "s", "s"] };
  const DEFAULT_LABELS = {
    1: ["features d"],
    2: ["batch N", "features d"],
    3: ["batch N", "steps T", "features d"],
    4: ["batch N", "channels C", "height H", "width W"],
  };

  function color(role) { return FC.dimColors[role] || "var(--dim-s)"; }
  function fmtShape(shape) { return `torch.Size([${shape.join(", ")}])`; }
  function shapeHtml(shape, roles) {
    const inner = shape.map((d, i) => `<b class="dim-${roles[i] || "s"}">${d}</b>`).join(", ");
    return `<span class="shape">(${inner}${shape.length === 1 ? "," : ""})</span>`;
  }

  /* Draw a grid of rows×cols cells at (x, y). Returns {w, h}. */
  function grid(parent, x, y, rows, cols, opts) {
    opts = opts || {};
    const r = Math.min(rows, MAX.rows), c = Math.min(cols, MAX.cols);
    for (let i = 0; i < r; i++) {
      for (let j = 0; j < c; j++) {
        parent.appendChild(S("rect", { x: x + j * PITCH, y: y + i * PITCH, width: CELL, height: CELL, rx: 2, class: "cell" }));
        if (opts.numbers) {
          parent.appendChild(S("text", { x: x + j * PITCH + CELL / 2, y: y + i * PITCH + CELL / 2 + 4, "text-anchor": "middle", class: "cell-text" }, String(opts.numbers[i * cols + j])));
        }
      }
    }
    if (cols > MAX.cols) dots(parent, x + c * PITCH + 2, y + (r * PITCH) / 2, "h");
    if (rows > MAX.rows) dots(parent, x + (c * PITCH) / 2, y + r * PITCH + 2, "v");
    return { w: c * PITCH - GAP + (cols > MAX.cols ? 14 : 0), h: r * PITCH - GAP + (rows > MAX.rows ? 14 : 0) };
  }
  function dots(parent, x, y, dir) {
    for (let k = 0; k < 3; k++) {
      const cx = dir === "h" ? x + 4 + k * 4 : x;
      const cy = dir === "h" ? y : y + 4 + k * 4;
      parent.appendChild(S("circle", { cx, cy, r: 1.2, class: "dots" }));
    }
  }
  function bar(parent, x1, y1, x2, y2, role) {
    parent.appendChild(S("line", { x1, y1, x2, y2, style: `stroke:${color(role)}`, "stroke-width": 3, "stroke-linecap": "round" }));
  }
  function label(parent, x, y, text, role, anchor) {
    parent.appendChild(S("text", { x, y, class: "axis-label", style: `fill:${color(role)}`, "text-anchor": anchor || "start" }, text));
  }

  /* Render a shape into an <svg>. roles/labels are per-dimension arrays. */
  function drawShape(svg, shape, roles, labels, opts) {
    opts = opts || {};
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    const g = S("g");
    const nd = shape.length;
    const longest = Math.max(0, ...(labels || []).map((l, i) => String(`${l} = ${shape[i]}`).length));
    const PAD_L = Math.max(70, longest * 7.4 + 24), PAD_T = 34, PAD_B = 34, PAD_R = 50;
    let contentW = 0, contentH = 0;

    if (nd === 0) {
      g.appendChild(S("rect", { x: PAD_L, y: PAD_T, width: CELL, height: CELL, rx: 2, class: "cell" }));
      g.appendChild(S("circle", { cx: PAD_L + CELL / 2, cy: PAD_T + CELL / 2, r: 3, style: "fill: var(--ink)" }));
      label(g, PAD_L + CELL + 10, PAD_T + CELL - 3, "a single number, shape ()", "s");
      contentW = 200; contentH = CELL;
    } else if (nd === 1) {
      const d = shape[0];
      const { w, h } = grid(g, PAD_L, PAD_T, 1, d, opts);
      bar(g, PAD_L, PAD_T - 8, PAD_L + w, PAD_T - 8, roles[0]);
      label(g, PAD_L + w / 2, PAD_T - 14, `${labels[0]} = ${d}`, roles[0], "middle");
      contentW = w; contentH = h;
    } else {
      const rows = shape[nd - 2], cols = shape[nd - 1];
      const depth = nd >= 3 ? shape[nd - 3] : 1;
      const groups = nd === 4 ? shape[0] : 1;
      const dVis = Math.min(depth, MAX.depth), gVis = Math.min(groups, MAX.groups);
      const gridW = Math.min(cols, MAX.cols) * PITCH - GAP + (cols > MAX.cols ? 14 : 0);
      const gridH = Math.min(rows, MAX.rows) * PITCH - GAP + (rows > MAX.rows ? 14 : 0);
      const stackX = (dVis - 1) * OFF, stackY = (dVis - 1) * OFF;
      const groupW = gridW + stackX + (depth > MAX.depth ? 16 : 0);
      const groupGap = 28;
      const baseY = PAD_T + stackY;
      for (let gi = 0; gi < gVis; gi++) {
        const gx = PAD_L + gi * (groupW + groupGap);
        for (let k = dVis - 1; k >= 0; k--) {
          grid(g, gx + k * OFF, baseY - k * OFF, rows, cols, k === 0 ? opts : {});
        }
        if (depth > MAX.depth) dots(g, gx + stackX + gridW + 4, baseY - stackY + 6, "h");
        if (gi === 0) {
          // axis bars on the front grid of the first group
          bar(g, gx, baseY - 8 - (nd >= 3 ? 0 : 0), gx + gridW, baseY - 8, roles[nd - 1]);                 // columns
          label(g, gx + gridW / 2, baseY - 14, `${labels[nd - 1]} = ${cols}`, roles[nd - 1], "middle");
          bar(g, gx - 8, baseY, gx - 8, baseY + gridH, roles[nd - 2]);                                       // rows
          label(g, gx - 14, baseY + gridH / 2 + 4, `${labels[nd - 2]} = ${rows}`, roles[nd - 2], "end");
          if (nd >= 3) {                                                                                     // depth, along the stack offset
            const x1 = gx + gridW + 6, y1 = baseY + gridH + 6;
            bar(g, x1, y1, x1 + stackX, y1 - stackY, roles[nd - 3]);
            label(g, x1 + stackX + 8, y1 - stackY + 4, `${labels[nd - 3]} = ${depth}`, roles[nd - 3]);
          }
        }
      }
      if (groups > MAX.groups) dots(g, PAD_L + gVis * (groupW + groupGap) - groupGap + 6, baseY + gridH / 2, "h");
      contentW = gVis * (groupW + groupGap) - groupGap + (groups > MAX.groups ? 20 : 0) + 60;
      contentH = gridH + stackY + 10;
      if (nd === 4) {                                                                                        // groups, below everything
        const y = baseY + gridH + 22;
        bar(g, PAD_L, y, PAD_L + gVis * (groupW + groupGap) - groupGap, y, roles[0]);
        label(g, PAD_L, y + 16, `${labels[0]} = ${groups}`, roles[0]);
        contentH += 36;
      }
    }
    svg.appendChild(g);
    const W = PAD_L + contentW + PAD_R, H = PAD_T + contentH + PAD_B;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.style.maxWidth = Math.round(Math.max(W, 260) * 1.5) + "px"; // cells read better a little larger than 1:1
  }
  FC.drawShape = drawShape;

  /* ---------- widget: shape builder ---------- */
  FC.widgets.shapeBuilder = function (el) {
    const presets = [
      { name: "One order", shape: [4], roles: ["d"], labels: ["features d"], code: "torch.tensor([2.3, 1, 18, 0])" },
      { name: "A batch of orders", shape: [32, 4], roles: ["n", "d"], labels: ["batch N", "features d"], code: "torch.randn(32, 4)" },
      { name: "A batch of images", shape: [8, 1, 28, 28], roles: ["n", "c", "s", "s"], labels: ["batch N", "channels C", "height H", "width W"], code: "torch.zeros(8, 1, 28, 28)" },
      { name: "A batch of sentences", shape: [2, 5, 8], roles: ["n", "s", "d"], labels: ["batch N", "tokens T", "features d"], code: "torch.zeros(2, 5, 8)" },
      { name: "A scalar", shape: [], roles: [], labels: [], code: "torch.tensor(3.5)" },
    ];
    el.innerHTML = `
      <div class="widget-head"><h3>Build a tensor and look at its shape</h3>
        <div class="presets btn-row" data-presets></div></div>
      <div class="widget-body">
        <div class="widget-controls">
          <label>shape <input type="text" data-shape value="(32, 4)" aria-label="shape, comma separated"></label>
          <span class="muted" data-hint>Type up to four sizes, like 8, 3, 28, 28.</span>
        </div>
        <svg role="img" aria-label="drawing of the tensor"></svg>
        <p class="readout" data-readout></p>
        <ul class="legend" data-legend></ul>
      </div>`;
    const input = el.querySelector("[data-shape]");
    const svg = el.querySelector("svg");
    const readout = el.querySelector("[data-readout]");
    const legend = el.querySelector("[data-legend]");
    const presetBox = el.querySelector("[data-presets]");
    let current = presets[1];

    function render(shape, roles, labels, code) {
      drawShape(svg, shape, roles, labels);
      const numel = shape.reduce((a, b) => a * b, 1);
      const words = shape.length === 0 ? "0 dimensions: a scalar." : `${shape.length} dimension${shape.length > 1 ? "s" : ""} holding ${numel.toLocaleString()} number${numel === 1 ? "" : "s"}.`;
      const shown = code || (shape.length ? `torch.zeros(${shape.join(", ")})` : "torch.tensor(3.5)");
      readout.innerHTML = `<span class="c">x = </span>${shown}<br>x.shape = ${shapeHtml(shape, roles)} <span class="muted">${words}</span>`;
      legend.innerHTML = shape.map((d, i) => `<li><span class="swatch" style="background:${color(roles[i])}"></span>dim ${i}: ${labels[i]} = ${d}</li>`).join("");
    }
    function fromInput() {
      const shape = FC.parseShape(input.value);
      if (shape === null || shape.length > 4) {
        readout.innerHTML = `<span class="err">Could not read that shape. Use up to four whole numbers separated by commas, for example 8, 3, 28, 28.</span>`;
        return;
      }
      const roles = DEFAULT_ROLES[shape.length], labels = DEFAULT_LABELS[shape.length] || [];
      presetBox.querySelectorAll(".chip").forEach((c) => c.setAttribute("aria-pressed", "false"));
      render(shape, roles, labels, null);
    }
    presets.forEach((p, i) => {
      const b = document.createElement("button");
      b.type = "button"; b.className = "chip"; b.textContent = p.name;
      b.setAttribute("aria-pressed", String(i === 1));
      b.addEventListener("click", () => {
        presetBox.querySelectorAll(".chip").forEach((c) => c.setAttribute("aria-pressed", "false"));
        b.setAttribute("aria-pressed", "true");
        input.value = `(${p.shape.join(", ")})`;
        render(p.shape, p.roles, p.labels, p.code);
      });
      presetBox.appendChild(b);
    });
    input.addEventListener("input", fromInput);
    render(current.shape, current.roles, current.labels, current.code);
  };

  /* ---------- widget: static figure (index hero) ---------- */
  FC.widgets.shapeFigure = function (el) {
    const shape = FC.parseShape(el.getAttribute("data-shape") || "(32, 4)");
    const roles = (el.getAttribute("data-roles") || "n,d").split(",");
    const labels = (el.getAttribute("data-labels") || "batch N,features d").split(",");
    const svg = S("svg", { role: "img", "aria-label": `a tensor of shape ${shape.join(" by ")}` });
    el.appendChild(svg);
    drawShape(svg, shape, roles, labels);
    svg.style.width = "100%";
  };

  /* ---------- widget: broadcasting checker ---------- */
  FC.widgets.broadcast = function (el) {
    const presets = [
      { name: "Add a bias to every order", a: "(32, 4)", b: "(4,)", story: "One bias per feature, applied to all 32 orders." },
      { name: "Scale each order", a: "(32, 4)", b: "(32, 1)", story: "One number per order, applied to all 4 features." },
      { name: "The classic mistake", a: "(32, 4)", b: "(32,)", story: "You meant one number per order, but a (32,) vector lines up with the features axis, not the batch axis." },
      { name: "Pairwise differences", a: "(32, 1, 4)", b: "(1, 100, 4)", story: "Every one of 32 orders against every one of 100 others. This trick appears again in Lesson 13." },
      { name: "Per-channel scaling", a: "(8, 3, 28, 28)", b: "(3, 1, 1)", story: "One number per colour channel. The two 1s stretch over height and width." },
    ];
    el.innerHTML = `
      <div class="widget-head"><h3>Will these two shapes broadcast?</h3>
        <div class="presets btn-row" data-presets></div></div>
      <div class="widget-body">
        <div class="widget-controls">
          <label>a <input type="text" data-a value="(32, 4)" aria-label="shape of a"></label>
          <span class="muted">+</span>
          <label>b <input type="text" data-b value="(4,)" aria-label="shape of b"></label>
        </div>
        <table class="bc-table" aria-live="polite"><tbody data-rows></tbody></table>
        <p class="readout" data-readout></p>
      </div>`;
    const ia = el.querySelector("[data-a]"), ib = el.querySelector("[data-b]");
    const rows = el.querySelector("[data-rows]"), readout = el.querySelector("[data-readout]");
    const presetBox = el.querySelector("[data-presets]");
    let story = presets[0].story;

    function compute() {
      const a = FC.parseShape(ia.value), b = FC.parseShape(ib.value);
      if (a === null || b === null) {
        rows.innerHTML = "";
        readout.innerHTML = `<span class="err">Could not read one of the shapes. Use whole numbers separated by commas.</span>`;
        return;
      }
      const n = Math.max(a.length, b.length);
      const ra = [], rb = [], rr = [];
      let clash = null;
      for (let i = 0; i < n; i++) {
        const ai = a[a.length - n + i], bi = b[b.length - n + i];
        const ca = { v: ai === undefined ? "–" : ai, cls: ai === undefined ? "miss" : "" };
        const cb = { v: bi === undefined ? "–" : bi, cls: bi === undefined ? "miss" : "" };
        let out;
        if (ai === undefined) out = bi;
        else if (bi === undefined) out = ai;
        else if (ai === bi) out = ai;
        else if (ai === 1) { out = bi; ca.cls = "stretch"; ca.v = `1→${bi}`; }
        else if (bi === 1) { out = ai; cb.cls = "stretch"; cb.v = `1→${ai}`; }
        else { out = "✗"; ca.cls = cb.cls = "clash"; if (clash === null) clash = { i, ai, bi }; }
        ra.push(ca); rb.push(cb); rr.push(out);
      }
      const cell = (c) => `<td class="${c.cls}">${c.v}</td>`;
      rows.innerHTML = `
        <tr><th>dimension, read from the right</th>${rr.map((_, i) => `<th style="text-align:center">${i}</th>`).join("")}</tr>
        <tr><th>a ${ia.value.trim()}</th>${ra.map(cell).join("")}</tr>
        <tr><th>b ${ib.value.trim()}</th>${rb.map(cell).join("")}</tr>
        <tr class="result"><th>result</th>${rr.map((v) => `<td class="${v === "✗" ? "clash" : ""}">${v}</td>`).join("")}</tr>`;
      if (clash) {
        readout.innerHTML = `<span class="err">RuntimeError: The size of tensor a (${clash.ai}) must match the size of tensor b (${clash.bi}) at non-singleton dimension ${clash.i}</span><br><span class="muted">${story}</span>`;
      } else {
        readout.innerHTML = `<span class="ok">(a + b).shape = ${fmtShape(rr)}</span><br><span class="muted">${story}</span>`;
      }
    }
    presets.forEach((p, i) => {
      const btn = document.createElement("button");
      btn.type = "button"; btn.className = "chip"; btn.textContent = p.name;
      btn.setAttribute("aria-pressed", String(i === 0));
      btn.addEventListener("click", () => {
        presetBox.querySelectorAll(".chip").forEach((c) => c.setAttribute("aria-pressed", "false"));
        btn.setAttribute("aria-pressed", "true");
        ia.value = p.a; ib.value = p.b; story = p.story; compute();
      });
      presetBox.appendChild(btn);
    });
    [ia, ib].forEach((inp) => inp.addEventListener("input", () => {
      presetBox.querySelectorAll(".chip").forEach((c) => c.setAttribute("aria-pressed", "false"));
      story = "Rule: line the shapes up from the right. Each pair must be equal, or one of them must be 1 (or missing). The 1 is stretched to match.";
      compute();
    }));
    compute();
  };

  /* ---------- widget: view vs permute ---------- */
  FC.widgets.viewPermute = function (el) {
    const base = { shape: [2, 3], values: [0, 1, 2, 3, 4, 5] };
    const ops = [
      { code: "x.view(3, 2)", shape: [3, 2], order: [0, 1, 2, 3, 4, 5], note: "Same six numbers in the same order, regrouped into three rows of two. A view costs nothing: no numbers are copied." },
      { code: "x.permute(1, 0)", shape: [3, 2], order: [0, 3, 1, 4, 2, 5], note: "Rows became columns. Read the result left to right and the numbers are in a new order: this is the difference between permute and view. The result is not contiguous, so x.permute(1, 0).view(6) raises an error; use .reshape(6) or .contiguous() first." },
      { code: "x.reshape(6)", shape: [6], order: [0, 1, 2, 3, 4, 5], note: "Back to one row. reshape gives a view when it can, and quietly copies when it can't, which is why it's the safe default." },
      { code: "x.unsqueeze(0)", shape: [1, 2, 3], order: [0, 1, 2, 3, 4, 5], note: "Nothing moved. A new dimension of size 1 was added at position 0, which is how a single example becomes a batch of one." },
      { code: "x.flatten()", shape: [6], order: [0, 1, 2, 3, 4, 5], note: "Flatten is reshape(-1): everything into one dimension, in memory order." },
    ];
    el.innerHTML = `
      <div class="widget-head"><h3>view, reshape and permute</h3><div class="presets btn-row" data-ops></div></div>
      <div class="widget-body">
        <div class="two-col">
          <div><p class="readout" style="margin:0 0 .4rem"><span class="c">x = </span>torch.arange(6).reshape(2, 3)<br>x.shape = ${shapeHtml([2, 3], ["n", "d"])}</p><svg aria-label="the original 2 by 3 tensor"></svg></div>
          <div><p class="readout" style="margin:0 0 .4rem" data-code></p><svg data-result aria-label="the result"></svg></div>
        </div>
        <p class="readout" data-order></p>
        <p class="muted" data-note style="max-width:68ch"></p>
      </div>`;
    const opsBox = el.querySelector("[data-ops]");
    const left = el.querySelector("svg"), right = el.querySelector("[data-result]");
    const codeEl = el.querySelector("[data-code]"), orderEl = el.querySelector("[data-order]"), noteEl = el.querySelector("[data-note]");

    function drawNumbered(svg, shape, order) {
      const roles = shape.length === 1 ? ["d"] : shape.length === 2 ? ["n", "d"] : ["s", "n", "d"];
      const labels = shape.length === 1 ? ["elements"] : shape.length === 2 ? ["rows", "cols"] : ["new dim", "rows", "cols"];
      drawShape(svg, shape, roles, labels, { numbers: order });
    }
    drawNumbered(left, base.shape, base.values);
    function show(op) {
      drawNumbered(right, op.shape, op.order);
      const roles = op.shape.length === 1 ? ["d"] : op.shape.length === 2 ? ["n", "d"] : ["s", "n", "d"];
      codeEl.innerHTML = `<span class="c">y = </span>${op.code}<br>y.shape = ${shapeHtml(op.shape, roles)}`;
      const same = op.order.every((v, i) => v === i);
      orderEl.innerHTML = `memory order of x: 0 1 2 3 4 5<br>reading order of y: <span class="${same ? "ok" : "err"}">${op.order.join(" ")}</span> ${same ? "(unchanged)" : "(changed)"}`;
      noteEl.textContent = op.note;
    }
    ops.forEach((op, i) => {
      const b = document.createElement("button");
      b.type = "button"; b.className = "chip"; b.textContent = op.code; b.setAttribute("aria-pressed", String(i === 0));
      b.addEventListener("click", () => {
        opsBox.querySelectorAll(".chip").forEach((c) => c.setAttribute("aria-pressed", "false"));
        b.setAttribute("aria-pressed", "true");
        show(op);
      });
      opsBox.appendChild(b);
    });
    show(ops[0]);
  };
})();
