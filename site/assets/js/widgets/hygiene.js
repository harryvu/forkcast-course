/* Lesson 17 widgets: where randomness enters a run, an experiment log you can group by config, and seed noise against a real difference. */
(function () {
  "use strict";
  const FC = window.FC;
  const S = FC.svgEl;

  /* ---------- widget: sources of randomness ---------- */
  FC.widgets.seedSources = function (el) {
    const sources = [
      { name: "weight initialisation", how: "torch.manual_seed(seed)", on: true },
      { name: "DataLoader shuffle", how: "DataLoader(..., generator=torch.Generator().manual_seed(seed))", on: true },
      { name: "dropout masks", how: "the same torch generator, once the model is seeded", on: true },
      { name: "data generation / splits", how: "make_orders(seed=), train_val_test_split(seed=): the data_seed field", on: true },
      { name: "Python random, NumPy", how: "random.seed(seed), np.random.seed(seed): augmentation and sampling code often uses them", on: true },
      { name: "GPU kernels", how: "torch.use_deterministic_algorithms(True), cudnn.benchmark = False; some ops have no deterministic version", on: false },
    ];
    el.innerHTML = `<div class="widget-head"><h3>Where randomness enters a run</h3></div>
      <div class="widget-body"><svg role="img" aria-label="six sources of randomness, each seeded or not"></svg><p class="readout" data-readout></p>
      <p class="muted" style="max-width:68ch">Click a row to toggle whether it is seeded. A run reproduces only when every source is pinned; one unseeded source is enough to make two "identical" runs differ, and the usual culprits are the loader's shuffle and a stray <code>random.random()</code> in augmentation code. GPU kernels are the one source you cannot always pin: some operations trade determinism for speed, which is why the course's definition-of-done numbers are measured on the CPU.</p></div>`;
    const svg = el.querySelector("svg"), readout = el.querySelector("[data-readout]");
    function draw() {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      svg.setAttribute("viewBox", "0 0 760 200");
      sources.forEach((s, k) => { const y = 8 + k * 31; svg.appendChild(S("rect", { x: 10, y, width: 740, height: 26, rx: 4, style: `fill: ${s.on ? "var(--ok-bg)" : "var(--bad-bg)"}; stroke: var(--line-2); cursor: pointer` })).addEventListener("click", () => { s.on = !s.on; draw(); }); svg.appendChild(S("text", { x: 20, y: y + 17, class: "cell-text", style: `fill: ${s.on ? "var(--ok)" : "var(--bad)"}; font-weight: 700; pointer-events: none` }, s.on ? "seeded" : "unseeded")); svg.appendChild(S("text", { x: 100, y: y + 17, class: "cell-text", style: "fill: var(--ink); pointer-events: none" }, s.name)); svg.appendChild(S("text", { x: 300, y: y + 17, class: "cell-text", style: "fill: var(--ink-2); font-size: 9.5px; pointer-events: none" }, s.how.length > 72 ? s.how.slice(0, 70) + "…" : s.how)); });
      const open = sources.filter((s) => !s.on);
      readout.innerHTML = open.length ? `<span class="err">run twice, get two results:</span> ${open.map((s) => s.name).join(", ")} ${open.length > 1 ? "are" : "is"} not pinned` : `<span class="ok">run twice, get the same digits.</span> seed_everything(seed) pins the first five; the sixth is the GPU's to keep.`;
    }
    draw();
  };

  /* ---------- widget: the experiment log ---------- */
  FC.widgets.runLog = function (el) {
    const runs = [
      ["1cf7a8b0", "mlp", "(32, 32)", 0.01, 0.0, 0, 7.561], ["1cf7a8b0", "mlp", "(32, 32)", 0.01, 0.0, 0, 7.561], ["e2a9f10c", "mlp", "(32, 32)", 0.01, 0.0, 1, 7.432], ["7b31d6e4", "mlp", "(32, 32)", 0.01, 0.0, 2, 7.759], ["a0c5e8b2", "mlp", "(32, 32)", 0.01, 0.0, 3, 7.868], ["5d9f2c41", "mlp", "(32, 32)", 0.01, 0.0, 4, 8.021],
      ["9f4b2d77", "mlp", "(64, 64)", 0.01, 0.0, 0, 6.583], ["c81e3a05", "mlp", "(64, 64)", 0.01, 0.0, 1, 7.486], ["3e6d0b9a", "mlp", "(64, 64)", 0.01, 0.0, 2, 7.390], ["b7a4f1c3", "mlp", "(64, 64)", 0.01, 0.0, 3, 6.547], ["d2c8e05f", "mlp", "(64, 64)", 0.01, 0.0, 4, 6.464],
      ["48ab7e12", "mlp", "(32, 32)", 0.01, 0.001, 0, 7.553], ["6c2f9d80", "mlp", "(32, 32)", 0.01, 0.001, 1, 7.418], ["f3e17a2b", "mlp", "(32, 32)", 0.01, 0.001, 2, 7.742], ["0a9c4d6e", "mlp", "(32, 32)", 0.01, 0.001, 3, 7.846], ["8e5b3f17", "mlp", "(32, 32)", 0.01, 0.001, 4, 7.997],
    ];
    let grouped = false;
    el.innerHTML = `<div class="widget-head"><h3>experiments.jsonl, one line per run</h3><div class="presets btn-row"><button type="button" class="chip" data-g>group by config</button></div></div>
      <div class="widget-body"><div data-table style="overflow-x:auto"></div>
      <p class="muted" style="max-width:68ch;margin-top:.6rem">The raw log answers "what did I run"; grouping by every field except the seed answers "what did I learn". Two identical lines at the top are the reproducibility check: same fingerprint, same digits. Numbers measured in the sandbox: 150 epochs on 2,000 nonlinear orders, validation MSE.</p></div>`;
    const table = el.querySelector("[data-table]");
    function draw() {
      if (!grouped) {
        table.innerHTML = `<table class="bc-table" style="font-size:.85rem"><tr><th>fingerprint</th><th>model</th><th>hidden</th><th>lr</th><th>weight_decay</th><th>seed</th><th>best_val</th></tr>${runs.map((r) => `<tr>${r.map((v, i) => `<td style="text-align:${i < 3 ? "left" : "right"}">${v}</td>`).join("")}</tr>`).join("")}</table>`;
      } else {
        const groups = {};
        runs.forEach((r) => { const k = r.slice(1, 5).join("|"); (groups[k] = groups[k] || []).push(r[6]); });
        const rows = Object.entries(groups).map(([k, vals]) => { const uniq = vals; const mean = uniq.reduce((s, v) => s + v, 0) / uniq.length; const std = Math.sqrt(uniq.reduce((s, v) => s + (v - mean) ** 2, 0) / (uniq.length - 1)); return [...k.split("|"), uniq.length, mean, std]; }).sort((a, b) => a[5] - b[5]);
        table.innerHTML = `<table class="bc-table" style="font-size:.85rem"><tr><th>model</th><th>hidden</th><th>lr</th><th>weight_decay</th><th>runs</th><th>mean_val</th><th>std_val</th></tr>${rows.map((r) => `<tr><td>${r[0]}</td><td>${r[1]}</td><td>${r[2]}</td><td>${r[3]}</td><td>${r[4]}</td><td><b>${r[5].toFixed(3)}</b></td><td>${r[6].toFixed(3)}</td></tr>`).join("")}</table><p class="readout" style="margin-top:.5rem"><span class="c"># (64, 64) beats (32, 32) by 0.84 with a std of 0.5: a real difference. weight_decay=0.001 changes the mean by 0.02 with a std of 0.23: seed noise, not an effect.</span></p>`;
      }
    }
    el.querySelector("[data-g]").addEventListener("click", (e) => { grouped = !grouped; e.currentTarget.setAttribute("aria-pressed", String(grouped)); draw(); });
    draw();
  };

  /* ---------- widget: is the difference real? ---------- */
  FC.widgets.seedNoise = function (el) {
    const st = { diff: 0.3, std: 0.25, n: 5 };
    el.innerHTML = `<div class="widget-head"><h3>Seed noise against a real difference</h3></div>
      <div class="widget-body"><div class="widget-controls" style="flex-direction:column;align-items:stretch;gap:.5rem;max-width:460px">
        <label style="display:grid;grid-template-columns:11rem 1fr 3.5rem;gap:.6rem;align-items:center"><span>difference of means</span><input type="range" min="0" max="1.2" step="0.02" value="0.3" data-d aria-label="difference"><span class="shape" data-dv>0.30</span></label>
        <label style="display:grid;grid-template-columns:11rem 1fr 3.5rem;gap:.6rem;align-items:center"><span>std across seeds</span><input type="range" min="0.05" max="0.8" step="0.01" value="0.25" data-s aria-label="std"><span class="shape" data-sv>0.25</span></label>
        <label style="display:grid;grid-template-columns:11rem 1fr 3.5rem;gap:.6rem;align-items:center"><span>seeds per config</span><input type="range" min="2" max="10" step="1" value="5" data-n aria-label="seeds"><span class="shape" data-nv>5</span></label></div>
        <svg role="img" aria-label="two distributions of results across seeds"></svg><p class="readout" data-readout></p>
        <p class="muted" style="max-width:68ch">Each config's result varies from seed to seed; the std measures by how much. A difference between two configs is believable when it is large compared with the spread of the means, roughly std × √(2/n). One seed per config cannot tell you anything about that spread, which is why single-run comparisons so often reverse when someone reruns them.</p></div>`;
    const svg = el.querySelector("svg"), readout = el.querySelector("[data-readout]");
    function draw() {
      const { diff, std, n } = st, se = std * Math.sqrt(2 / n), t = diff / se;
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      svg.setAttribute("viewBox", "0 0 640 170"); const L = 40, W = 560, H = 120, base = 140;
      const x = (v) => L + ((v + 1.5) / 4) * W;
      [["config A", 0, "var(--ink-3)"], ["config B", diff, "var(--dim-y)"]].forEach(([name, mu, col]) => {
        let d = ""; for (let k = 0; k <= 120; k++) { const v = -1.5 + (4 * k) / 120, y = Math.exp(-0.5 * ((v - mu) / std) ** 2); d += (k ? " L" : "M") + x(v) + " " + (base - y * H * 0.9); }
        svg.appendChild(S("path", { d, fill: "none", style: `stroke: ${col}; stroke-width: 2.5` }));
        svg.appendChild(S("text", { x: x(mu), y: 18, "text-anchor": "middle", class: "cell-text", style: `fill: ${col}` }, name));
      });
      svg.appendChild(S("line", { x1: L, y1: base, x2: L + W, y2: base, style: "stroke: var(--line)" }));
      svg.appendChild(S("text", { x: L + W / 2, y: 162, "text-anchor": "middle", class: "cell-text", style: "fill: var(--ink-2)" }, "best validation loss, shifted so config A sits at 0 (each curve: the result's spread across seeds)"));
      readout.innerHTML = `difference ${diff.toFixed(2)}, spread of a mean over ${n} seeds ≈ ${se.toFixed(3)}, ratio ${t.toFixed(1)}   <span class="${t > 3 ? "ok" : t > 1.5 ? "" : "err"}"># ${t > 3 ? "clearly real" : t > 1.5 ? "suggestive; run more seeds" : "indistinguishable from seed noise"}</span>`;
    }
    el.querySelector("[data-d]").addEventListener("input", (e) => { st.diff = +e.target.value; el.querySelector("[data-dv]").textContent = st.diff.toFixed(2); draw(); });
    el.querySelector("[data-s]").addEventListener("input", (e) => { st.std = +e.target.value; el.querySelector("[data-sv]").textContent = st.std.toFixed(2); draw(); });
    el.querySelector("[data-n]").addEventListener("input", (e) => { st.n = +e.target.value; el.querySelector("[data-nv]").textContent = String(st.n); draw(); });
    draw();
  };
})();
