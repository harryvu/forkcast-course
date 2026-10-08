/* Lesson 15 widgets: one request through the service, batching against latency, what goes in the artifact. */
(function () {
  "use strict";
  const FC = window.FC;
  const S = FC.svgEl;

  /* ---------- widget: a request, step by step ---------- */
  FC.widgets.requestFlow = function (el) {
    const stages = [
      { name: "HTTP request", code: 'POST /eta\n{"orders": [{"trip_distance": 1.95, "pickup_hour": 13, "weekday": 6,\n             "passenger_count": 3, "pu_zone": 99, "do_zone": 41}]}', note: "JSON from the app. Nothing here is a tensor yet, and nothing can be trusted: a client can send any string, any number, any shape." },
      { name: "validation (pydantic)", code: "class Order(BaseModel):\n    trip_distance: float = Field(gt=0, le=50)\n    pickup_hour: int = Field(ge=0, le=23)\n    pu_zone: int = Field(ge=0, lt=266)   # ...", note: "FastAPI parses the body into typed objects and answers 422 with the field and the reason when something is missing, the wrong type or out of range. The model only ever sees inputs it was trained on the range of." },
      { name: "features", code: "frame = pd.DataFrame(orders)\ncont = normalizer.transform(tensor(frame[CONTINUOUS].values))   # (N, 4)\ncat = tensor(frame[CATEGORICAL].values, dtype=long)              # (N, 2)", note: "The same column order and the same normalizer statistics as training, loaded from the artifact. Most serving bugs live here: a reordered column or a freshly fitted normalizer gives confident nonsense with no error." },
      { name: "model, inference mode", code: "with torch.inference_mode():\n    minutes = model(cont, cat).tolist()", note: "eval mode, no autograd graph, no gradient buffers: faster and lighter than training-time forward. The whole batch of orders goes through in one call." },
      { name: "HTTP response", code: '200 OK\n{"minutes": [5.41]}', note: "Plain floats back out. Round-trip inside one process: about 3 ms for one order on a CPU, most of it HTTP and parsing, not the network." },
    ];
    let i = 0;
    el.innerHTML = `<div class="widget-head"><h3>One request through the service</h3><div class="presets btn-row"><button type="button" class="chip" data-next>next</button><button type="button" class="chip" data-reset>reset</button></div></div>
      <div class="widget-body"><svg role="img" aria-label="the stages of one request"></svg><pre class="no-copy" data-code style="margin:.6rem 0"></pre><p class="muted" data-note style="max-width:68ch"></p></div>`;
    const svg = el.querySelector("svg"), code = el.querySelector("[data-code]"), note = el.querySelector("[data-note]");
    function draw() {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      svg.setAttribute("viewBox", "0 0 760 70");
      stages.forEach((s, k) => { const x = 10 + k * 150, on = k === i; svg.appendChild(S("rect", { x, y: 14, width: 138, height: 42, rx: 7, style: `fill: ${on ? "var(--dim-n)" : "var(--panel)"}; fill-opacity: ${on ? 0.18 : 1}; stroke: ${on ? "var(--dim-n)" : "var(--line)"}; stroke-width: ${on ? 2 : 1}; cursor: pointer` })).addEventListener("click", () => { i = k; draw(); }); svg.appendChild(S("text", { x: x + 69, y: 40, "text-anchor": "middle", class: "cell-text", style: "fill: var(--ink); pointer-events: none" }, s.name)); if (k < stages.length - 1) svg.appendChild(S("text", { x: x + 144, y: 40, "text-anchor": "middle", class: "axis-label", style: "fill: var(--ink-3)" }, "→")); });
      code.textContent = stages[i].code; note.textContent = stages[i].note;
    }
    el.querySelector("[data-next]").addEventListener("click", () => { i = (i + 1) % stages.length; draw(); });
    el.querySelector("[data-reset]").addEventListener("click", () => { i = 0; draw(); });
    draw();
  };

  /* ---------- widget: batching against latency ---------- */
  FC.widgets.batchingLatency = function (el) {
    const st = { rps: 200, window: 5 };
    const FIXED = 1.0, PER_ITEM = 0.02;   // ms: measured in the sandbox for the ETA model
    el.innerHTML = `<div class="widget-head"><h3>Batching: throughput against latency</h3></div>
      <div class="widget-body"><div class="widget-controls" style="flex-direction:column;align-items:stretch;gap:.5rem;max-width:460px">
        <label style="display:grid;grid-template-columns:9rem 1fr 4rem;gap:.6rem;align-items:center"><span>requests / second</span><input type="range" min="10" max="2000" step="10" value="200" data-rps aria-label="requests per second"><span class="shape" data-rpsv>200</span></label>
        <label style="display:grid;grid-template-columns:9rem 1fr 4rem;gap:.6rem;align-items:center"><span>batch window (ms)</span><input type="range" min="0" max="20" step="1" value="5" data-win aria-label="batch window"><span class="shape" data-winv>5</span></label></div>
        <svg role="img" aria-label="per-request latency and model time per request for one-at-a-time against batched"></svg><p class="readout" data-readout></p>
        <p class="muted" style="max-width:68ch">A forward pass costs about 1 ms of fixed work plus 0.02 ms per order on this CPU (measured in Lesson 15's tests). Serving one order per call spends almost all of it on the fixed part. Collecting the orders that arrive within a short window and running them as one batch cuts the model cost per order by that factor, at the price of each request waiting up to the window. Past about 1000 requests per second the one-at-a-time service cannot keep up: requests queue and latency grows without bound.</p></div>`;
    const svg = el.querySelector("svg"), readout = el.querySelector("[data-readout]");
    function draw() {
      const { rps, window: win } = st;
      const perRequestSingle = FIXED + PER_ITEM, capacitySingle = 1000 / perRequestSingle;
      const perBatch = Math.max(1, Math.round((rps * win) / 1000));
      const batchCost = FIXED + PER_ITEM * perBatch, perRequestBatched = batchCost / perBatch, batchedLatency = win + batchCost;
      const overloaded = rps > capacitySingle;
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      svg.setAttribute("viewBox", "0 0 760 150");
      const rows = [["one request per forward", perRequestSingle, overloaded ? "unbounded (queue)" : perRequestSingle.toFixed(2) + " ms", overloaded ? "var(--bad)" : "var(--ink-3)"], [`batched (${perBatch} per forward)`, perRequestBatched, batchedLatency.toFixed(2) + " ms with wait", "var(--dim-y)"]];
      svg.appendChild(S("text", { x: 10, y: 16, class: "cell-text", style: "fill: var(--ink-2)" }, "model time per request (bar) and what the client waits (label)"));
      rows.forEach(([name, cost, label, col], r) => { const y = 34 + r * 44; svg.appendChild(S("text", { x: 10, y: y + 15, class: "cell-text", style: "fill: var(--ink)" }, name)); svg.appendChild(S("rect", { x: 230, y, width: Math.min(260, cost * 200), height: 22, rx: 3, style: `fill: ${col}` })); svg.appendChild(S("text", { x: 236 + Math.min(260, cost * 200), y: y + 15, class: "cell-text" }, `${cost.toFixed(3)} ms → latency ${label}`)); });
      svg.appendChild(S("text", { x: 10, y: 140, class: "cell-text", style: "fill: var(--ink-2)" }, `capacity one-at-a-time: ${Math.round(capacitySingle)} requests/s   |   batched: ${Math.round(1000 / perRequestBatched)} requests/s`));
      readout.innerHTML = `${rps} req/s, window ${win} ms   <span class="c"># ${overloaded ? "one-at-a-time cannot keep up; " : ""}batching ${perBatch} order${perBatch === 1 ? "" : "s"} costs ${batchCost.toFixed(2)} ms per forward, ${perRequestBatched.toFixed(3)} ms per order${perBatch === 1 ? " (the window is too short for this load to form a batch)" : ""}</span>`;
    }
    el.querySelector("[data-rps]").addEventListener("input", (e) => { st.rps = +e.target.value; el.querySelector("[data-rpsv]").textContent = String(st.rps); draw(); });
    el.querySelector("[data-win]").addEventListener("input", (e) => { st.window = +e.target.value; el.querySelector("[data-winv]").textContent = String(st.window); draw(); });
    draw();
  };

  /* ---------- widget: the artifact ---------- */
  FC.widgets.artifactBundle = function (el) {
    const items = [
      { name: "model.state_dict()", need: true, why: "The weights. 27 kB for the ETA model, 45 MB for ResNet-18.", off: "Nothing to serve." },
      { name: "normalizer mean and std", need: true, why: "The statistics the weights were trained with. Fitting new ones on live traffic changes every input.", off: "Predictions drift by minutes with no error anywhere." },
      { name: "architecture config (hidden sizes, vocab, classes)", need: true, why: "To rebuild the same shapes before load_state_dict, and to name the outputs.", off: "size mismatch on load, or label 2 with no idea what 2 means." },
      { name: "tokenizer / vocabulary", need: true, why: "For text models: the id of every token must match training (Lessons 11 and 13).", off: "Every word becomes <unk>; the model answers the same thing for everything." },
      { name: "a version and the training commit", need: true, why: "So a bad prediction can be traced to the run that produced the weights.", off: "No way to roll back or reproduce." },
      { name: "optimizer state", need: false, why: "Needed to resume training (Lesson 8), useless for inference and often larger than the weights.", off: "" },
      { name: "training data", need: false, why: "Never ships. Privacy, size, and the service must not depend on it.", off: "" },
      { name: "the training script", need: false, why: "The artifact must load with inference code only; that is what makes it portable to a Docker image or an export format.", off: "" },
    ];
    let sel = 1;
    el.innerHTML = `<div class="widget-head"><h3>What goes into the artifact</h3></div>
      <div class="widget-body"><svg role="img" aria-label="checklist of what an inference artifact contains"></svg><p class="readout" data-readout></p></div>`;
    const svg = el.querySelector("svg"), readout = el.querySelector("[data-readout]");
    function draw() {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      svg.setAttribute("viewBox", "0 0 760 240");
      items.forEach((it, k) => { const y = 10 + k * 28, on = k === sel; svg.appendChild(S("rect", { x: 10, y, width: 740, height: 24, rx: 4, style: `fill: ${on ? "var(--dim-n)" : "var(--panel)"}; fill-opacity: ${on ? 0.15 : 1}; stroke: ${on ? "var(--dim-n)" : "var(--line-2)"}; cursor: pointer` })).addEventListener("click", () => { sel = k; draw(); }); svg.appendChild(S("text", { x: 20, y: y + 16, class: "cell-text", style: `fill: ${it.need ? "var(--ok)" : "var(--bad)"}; font-weight: 700; pointer-events: none` }, it.need ? "✓ ship" : "✗ leave out")); svg.appendChild(S("text", { x: 120, y: y + 16, class: "cell-text", style: "fill: var(--ink); pointer-events: none" }, it.name)); });
      const it = items[sel];
      readout.innerHTML = `<b>${it.name}</b>: ${it.why}${it.off ? `<br><span class="c"># if it is missing: ${it.off}</span>` : ""}`;
    }
    draw();
  };
})();
