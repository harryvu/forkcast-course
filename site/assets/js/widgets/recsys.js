/* Lesson 14 widgets: tastes as directions, implicit feedback and negatives, hit rate at k. */
(function () {
  "use strict";
  const FC = window.FC;
  const S = FC.svgEl;
  function rng(seed) { return function () { seed |= 0; seed = (seed + 0x6D2B79F5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }

  /* ---------- widget: the taste plane ---------- */
  FC.widgets.latentMap = function (el) {
    const items = [["Pho Saigon", 0.9, 0.2], ["Ramen Lab", 0.8, -0.3], ["Curry House", 0.7, 0.7], ["Taco Truck", 0.4, -0.8], ["Pizza Napoli", -0.6, 0.1], ["Burger Barn", -0.8, -0.6], ["Lasagna Nonna", -0.7, 0.6], ["Salad Bar", -0.2, 0.9], ["Sushi Ko", 0.2, 0.5], ["Kebab Corner", 0.3, -0.5]];
    const st = { x: 0.6, y: 0.4 };
    el.innerHTML = `<div class="widget-head"><h3>Tastes as directions: move the user</h3></div>
      <div class="widget-body"><div class="widget-controls" style="flex-direction:column;align-items:stretch;gap:.5rem;max-width:420px">
        <label style="display:grid;grid-template-columns:7rem 1fr 3rem;gap:.6rem;align-items:center"><span>spicy ⟷ mild</span><input type="range" min="-1" max="1" step="0.05" value="0.6" data-x aria-label="user taste axis 1"><span class="shape" data-xv>0.60</span></label>
        <label style="display:grid;grid-template-columns:7rem 1fr 3rem;gap:.6rem;align-items:center"><span>light ⟷ hearty</span><input type="range" min="-1" max="1" step="0.05" value="0.4" data-y aria-label="user taste axis 2"><span class="shape" data-yv>0.40</span></label></div>
        <div class="two-col"><div><svg data-plane role="img" aria-label="restaurants and one user in a two-dimensional taste space"></svg></div><div><svg data-rank role="img" aria-label="restaurants ranked by dot product with the user"></svg></div></div>
        <p class="readout" data-readout></p>
        <p class="muted" style="max-width:68ch">Matrix factorization learns a vector for every user and every restaurant so that their dot product is high where orders happened. Nobody labels the axes; "spicy" and "hearty" are our reading of what the model found, and in the real model there are sixteen of them. Move the user and the ranking follows: that is the whole recommender.</p></div>`;
    const plane = el.querySelector("[data-plane]"), rank = el.querySelector("[data-rank]"), readout = el.querySelector("[data-readout]");
    function draw() {
      const scored = items.map(([n, x, y]) => ({ n, x, y, s: x * st.x + y * st.y })).sort((a, b) => b.s - a.s);
      while (plane.firstChild) plane.removeChild(plane.firstChild);
      plane.setAttribute("viewBox", "0 0 370 320"); const cx = 170, cy = 160, R = 120;
      plane.appendChild(S("line", { x1: 20, y1: cy, x2: 300, y2: cy, style: "stroke: var(--line)" })); plane.appendChild(S("line", { x1: cx, y1: 20, x2: cx, y2: 300, style: "stroke: var(--line)" }));
      plane.appendChild(S("text", { x: 300, y: cy - 6, "text-anchor": "end", class: "cell-text", style: "fill: var(--ink-3)" }, "spicy")); plane.appendChild(S("text", { x: 20, y: cy - 6, class: "cell-text", style: "fill: var(--ink-3)" }, "mild"));
      plane.appendChild(S("text", { x: cx + 6, y: 28, class: "cell-text", style: "fill: var(--ink-3)" }, "hearty")); plane.appendChild(S("text", { x: cx + 6, y: 300, class: "cell-text", style: "fill: var(--ink-3)" }, "light"));
      scored.forEach((it, i) => { const px = cx + R * it.x, py = cy - R * it.y, top = i < 3; plane.appendChild(S("circle", { cx: px, cy: py, r: top ? 7 : 5, style: `fill: ${top ? "var(--dim-y)" : "var(--ink-3)"}` })); plane.appendChild(S("text", { x: px + 9, y: py + 4, class: "cell-text", style: `fill: ${top ? "var(--ink)" : "var(--ink-3)"}; font-size: 9.5px` }, it.n)); });
      plane.appendChild(S("line", { x1: cx, y1: cy, x2: cx + R * st.x, y2: cy - R * st.y, style: "stroke: var(--dim-n); stroke-width: 3", "stroke-linecap": "round" }));
      plane.appendChild(S("circle", { cx: cx + R * st.x, cy: cy - R * st.y, r: 7, style: "fill: var(--dim-n); stroke: var(--panel); stroke-width: 2" }));
      plane.appendChild(S("text", { x: cx + R * st.x + 10, y: cy - R * st.y - 8, class: "axis-label", style: "fill: var(--dim-n)" }, "user"));
      while (rank.firstChild) rank.removeChild(rank.firstChild);
      rank.setAttribute("viewBox", "0 0 380 320");
      rank.appendChild(S("text", { x: 10, y: 16, class: "cell-text", style: "fill: var(--ink-2)" }, "score_all(user): user_vec · item_vec, sorted"));
      scored.forEach((it, i) => { const y = 28 + i * 28, w = Math.max(0, it.s) * 140; rank.appendChild(S("text", { x: 10, y: y + 13, class: "cell-text", style: `fill: var(--ink); font-weight: ${i < 3 ? 600 : 400}` }, it.n)); rank.appendChild(S("rect", { x: 130, y: y + 1, width: w, height: 16, rx: 3, style: `fill: ${i < 3 ? "var(--dim-y)" : "var(--ink-3)"}` })); if (it.s < 0) rank.appendChild(S("rect", { x: 130 + it.s * 60, y: y + 1, width: -it.s * 60, height: 16, rx: 3, style: "fill: var(--line)" })); rank.appendChild(S("text", { x: 136 + w, y: y + 13, class: "cell-text" }, it.s.toFixed(2))); });
      readout.innerHTML = `user_vec = [${st.x.toFixed(2)}, ${st.y.toFixed(2)}]   <span class="c"># top 3: ${scored.slice(0, 3).map((it) => it.n).join(", ")}</span>`;
    }
    el.querySelector("[data-x]").addEventListener("input", (e) => { st.x = +e.target.value; el.querySelector("[data-xv]").textContent = st.x.toFixed(2); draw(); });
    el.querySelector("[data-y]").addEventListener("input", (e) => { st.y = +e.target.value; el.querySelector("[data-yv]").textContent = st.y.toFixed(2); draw(); });
    draw();
  };

  /* ---------- widget: implicit feedback and negative sampling ---------- */
  FC.widgets.negativeSampling = function (el) {
    const n = 16, ordered = new Set([1, 4, 5, 9, 12]);
    let q = rng(14), negatives = [];
    el.innerHTML = `<div class="widget-head"><h3>What the data says, and what it does not</h3><div class="presets btn-row"><button type="button" class="chip" data-sample>sample negatives</button></div></div>
      <div class="widget-body"><svg role="img" aria-label="one user's row of restaurants: ordered, never ordered, sampled as negatives"></svg><p class="readout" data-readout></p>
      <p class="muted" style="max-width:68ch">Orders are implicit feedback: a 1 where the customer ordered and silence everywhere else. Silence is not dislike; it is mostly restaurants they never saw. So training treats the ordered pairs as positives and draws a few random unordered restaurants per positive as weak negatives. Occasionally a sampled negative is a place they would love, which is why the signal is noisy and why millions of orders still add up to good recommendations.</p></div>`;
    const svg = el.querySelector("svg"), readout = el.querySelector("[data-readout]");
    function draw() {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      svg.setAttribute("viewBox", "0 0 760 110");
      svg.appendChild(S("text", { x: 10, y: 16, class: "cell-text", style: "fill: var(--ink-2)" }, "user 42, restaurants 0 to 15"));
      for (let i = 0; i < n; i++) {
        const x = 10 + i * 46, pos = ordered.has(i), neg = negatives.includes(i);
        svg.appendChild(S("rect", { x, y: 26, width: 40, height: 40, rx: 5, style: `fill: ${pos ? "var(--dim-y)" : neg ? "var(--dim-n)" : "var(--panel)"}; fill-opacity: ${pos || neg ? 0.25 : 1}; stroke: ${pos ? "var(--dim-y)" : neg ? "var(--dim-n)" : "var(--line)"}` }));
        svg.appendChild(S("text", { x: x + 20, y: 51, "text-anchor": "middle", class: "cell-text", style: "fill: var(--ink)" }, pos ? "1" : neg ? "0?" : "·"));
        svg.appendChild(S("text", { x: x + 20, y: 82, "text-anchor": "middle", class: "cell-text", style: "fill: var(--ink-3); font-size: 9px" }, String(i)));
      }
      svg.appendChild(S("text", { x: 10, y: 104, class: "cell-text", style: "fill: var(--ink-2)" }, `positives: ${[...ordered].join(", ")}   negatives this batch: ${negatives.length ? negatives.join(", ") : "none sampled yet"}`));
      readout.innerHTML = negatives.length ? `negatives = sample_negatives(users, n_items, seen)   <span class="c"># never one of ${[...ordered].join(", ")}; a fresh draw every batch</span>` : `loss = BCE(score(user, ordered), 1) + BCE(score(user, negative), 0)   <span class="c"># press the button to draw negatives</span>`;
    }
    el.querySelector("[data-sample]").addEventListener("click", () => { negatives = []; while (negatives.length < ordered.size) { const c = Math.floor(q() * n); if (!ordered.has(c) && !negatives.includes(c)) negatives.push(c); } draw(); });
    draw();
  };

  /* ---------- widget: hit rate at k ---------- */
  FC.widgets.hitRateDemo = function (el) {
    const users = [{ held: "Curry House", learned: ["Curry House", "Pho Saigon", "Kebab Corner", "Sushi Ko", "Ramen Lab", "Salad Bar"], pop: ["Pizza Napoli", "Burger Barn", "Ramen Lab", "Pho Saigon", "Taco Truck", "Curry House"] }, { held: "Burger Barn", learned: ["Taco Truck", "Burger Barn", "Kebab Corner", "Pizza Napoli", "Ramen Lab", "Pho Saigon"], pop: ["Pizza Napoli", "Burger Barn", "Ramen Lab", "Pho Saigon", "Taco Truck", "Curry House"] }, { held: "Salad Bar", learned: ["Sushi Ko", "Lasagna Nonna", "Curry House", "Salad Bar", "Pho Saigon", "Pizza Napoli"], pop: ["Pizza Napoli", "Burger Barn", "Ramen Lab", "Pho Saigon", "Taco Truck", "Curry House"] }, { held: "Pho Saigon", learned: ["Ramen Lab", "Pho Saigon", "Curry House", "Sushi Ko", "Kebab Corner", "Taco Truck"], pop: ["Pizza Napoli", "Burger Barn", "Ramen Lab", "Pho Saigon", "Taco Truck", "Curry House"] }, { held: "Kebab Corner", learned: ["Taco Truck", "Ramen Lab", "Pho Saigon", "Curry House", "Burger Barn", "Kebab Corner"], pop: ["Pizza Napoli", "Burger Barn", "Ramen Lab", "Pho Saigon", "Taco Truck", "Curry House"] }];
    let k = 3;
    el.innerHTML = `<div class="widget-head"><h3>Hit rate at k</h3></div>
      <div class="widget-body"><div class="widget-controls"><label style="display:grid;grid-template-columns:auto 1fr auto;gap:.6rem;align-items:center;min-width:260px">k <input type="range" min="1" max="6" step="1" value="3" data-k aria-label="k"><span class="shape" data-kv>3</span></label></div>
      <svg role="img" aria-label="five users' held-out orders against two ranked lists"></svg><p class="readout" data-readout></p>
      <p class="muted" style="max-width:68ch">For each user, hide their last order, rank every restaurant they have not tried, and ask whether the hidden one is in the top k. Averaged over users that is hit rate at k, a number you can compare across models and against the one baseline you must beat: recommending whatever is most popular. Illustrative lists.</p></div>`;
    const svg = el.querySelector("svg"), readout = el.querySelector("[data-readout]");
    function draw() {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      svg.setAttribute("viewBox", "0 0 760 190");
      let hitsL = 0, hitsP = 0;
      users.forEach((u, r) => {
        const y = 20 + r * 32;
        svg.appendChild(S("text", { x: 10, y: y + 14, class: "cell-text", style: "fill: var(--ink)" }, `held out: ${u.held}`));
        [["learned", u.learned, 150], ["popularity", u.pop, 460]].forEach(([name, list, x0]) => {
          const hit = list.slice(0, k).includes(u.held); if (hit) { if (name === "learned") hitsL += 1; else hitsP += 1; }
          list.forEach((it, i) => { const x = x0 + i * 48, inK = i < k, isHeld = it === u.held; svg.appendChild(S("rect", { x, y, width: 44, height: 22, rx: 3, style: `fill: ${inK ? (name === "learned" ? "var(--dim-y)" : "var(--ink-3)") : "var(--line-2)"}; fill-opacity: ${inK ? 0.25 : 1}; stroke: ${isHeld ? "var(--ink)" : "none"}; stroke-width: 2` })); svg.appendChild(S("text", { x: x + 22, y: y + 15, "text-anchor": "middle", class: "cell-text", style: "font-size: 8.5px; fill: var(--ink)" }, it.split(" ")[0])); });
        });
      });
      svg.appendChild(S("text", { x: 150, y: 12, class: "cell-text", style: "fill: var(--ink-2)" }, "learned model's ranking")); svg.appendChild(S("text", { x: 460, y: 12, class: "cell-text", style: "fill: var(--ink-2)" }, "popularity ranking"));
      svg.appendChild(S("text", { x: 10, y: 185, class: "cell-text", style: "fill: var(--ink-2)" }, "outlined cell = the held-out restaurant; shaded = inside the top k"));
      readout.innerHTML = `hit_rate_at_k(model, test, seen, k=${k})   <span class="c"># learned ${hitsL}/5 = ${(hitsL / 5).toFixed(2)}, popularity ${hitsP}/5 = ${(hitsP / 5).toFixed(2)}</span>`;
    }
    el.querySelector("[data-k]").addEventListener("input", (e) => { k = +e.target.value; el.querySelector("[data-kv]").textContent = String(k); draw(); });
    draw();
  };
})();
