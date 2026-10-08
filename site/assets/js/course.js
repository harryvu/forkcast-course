/* Forkcast course — shared page behaviour for the single-page course.
   The page holds the home view (#home) and one <article data-lesson="NN"> per
   lesson. A hash router shows one at a time: #lesson-02 opens a lesson,
   #lesson-02/build scrolls to the section with id l02-build.
   Browser storage holds "fc.progress" and "fc.theme"; every access is guarded. */
(function () {
  "use strict";

  const FC = (window.FC = window.FC || {});
  FC.widgets = FC.widgets || {};
  FC.lessons = {};

  /* ---------- storage ---------- */
  const store = {
    read(key, fallback) {
      try { const raw = localStorage.getItem(key); return raw === null ? fallback : JSON.parse(raw); } catch (_) { return fallback; }
    },
    write(key, value) {
      try { localStorage.setItem(key, JSON.stringify(value)); } catch (_) { /* storage unavailable: the page still works, it just won't remember */ }
    },
  };
  FC.progress = {
    all() { return store.read("fc.progress", {}); },
    get(lesson) { return this.all()[lesson] || { quizzes: {}, dod: {}, done: false }; },
    set(lesson, data) { const all = this.all(); all[lesson] = data; store.write("fc.progress", all); },
    reset(lesson) { const all = this.all(); delete all[lesson]; store.write("fc.progress", all); },
  };

  /* ---------- theme ---------- */
  function prefersDark() {
    try { return !!(window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches); } catch (_) { return false; }
  }
  function isDark(root) { return root.getAttribute("data-theme") === "dark" || (!root.hasAttribute("data-theme") && prefersDark()); }
  function applyTheme(theme) {
    const root = document.documentElement;
    if (theme === "light" || theme === "dark") root.setAttribute("data-theme", theme); else root.removeAttribute("data-theme");
    document.querySelectorAll("[data-theme-toggle]").forEach((btn) => {
      const dark = isDark(root);
      btn.setAttribute("aria-label", dark ? "Switch to light theme" : "Switch to dark theme");
      btn.textContent = dark ? "☀" : "☾";
    });
  }
  function initTheme() {
    applyTheme(store.read("fc.theme", null));
    document.querySelectorAll("[data-theme-toggle]").forEach((btn) => btn.addEventListener("click", () => {
      const next = isDark(document.documentElement) ? "light" : "dark";
      store.write("fc.theme", next); applyTheme(next);
    }));
  }

  /* ---------- code blocks: where does this run? ----------
     Lines starting with ! or % are Colab notebook cells; blocks written with a $ prompt are shell
     commands (in Colab, prefix them with !). A badge next to the copy button says which. */
  function initEnvBadges() {
    document.querySelectorAll("pre").forEach((pre) => {
      if (pre.closest(".no-copy") || pre.querySelector(".env-badge")) return;
      const code = pre.querySelector("code") || pre;
      const lines = [];
      code.childNodes.forEach((node) => { if (!(node.nodeType === 1 && node.classList.contains("out"))) lines.push(node.textContent); });
      const text = lines.join("");
      let label = null, title = null;
      if (/(^|\n)\s*[!%]\w/.test(text)) { label = "Colab cell"; title = "A notebook cell: ! runs a shell command, % is an IPython magic, {name} inserts a Python variable. In a terminal, drop the ! and the braces."; }
      else if (code.querySelector(".cmd")) { label = "shell command"; title = "Run in a terminal, or in a Colab cell with a leading !"; }
      if (!label) return;
      const badge = document.createElement("span");
      badge.className = "env-badge"; badge.textContent = label; badge.title = title;
      pre.appendChild(badge);
    });
  }

  /* ---------- copy buttons on code blocks ---------- */
  function initCopyButtons() {
    document.querySelectorAll("pre").forEach((pre) => {
      if (pre.closest(".no-copy") || pre.querySelector(".copy-btn")) return;
      const btn = document.createElement("button");
      btn.className = "copy-btn"; btn.type = "button"; btn.textContent = "Copy";
      btn.addEventListener("click", async (e) => {
        e.preventDefault(); e.stopPropagation();           // a code block inside a DoD label must not toggle its checkbox
        const code = pre.querySelector("code") || pre;
        const lines = [];
        code.childNodes.forEach((node) => { if (node.nodeType === 1 && node.classList.contains("out")) return; lines.push(node.textContent); });
        const text = lines.join("").replace(/\n{3,}/g, "\n\n").trim();
        try { await navigator.clipboard.writeText(text); btn.textContent = "Copied"; } catch (_) { btn.textContent = "Select and copy"; }
        setTimeout(() => (btn.textContent = "Copy"), 1600);
      });
      pre.appendChild(btn);
    });
  }

  /* ---------- GitHub target: whose repository do the links point at? ----------
     The viewer enters a GitHub user (and repo name) once; it is remembered in this browser.
     FC_CONFIG.repo is the course's own repository and is only used when it is not the placeholder. */
  FC.github = {
    get() { return store.read("fc.github", null); },
    set(user, repo) {
      user = String(user || "").trim().replace(/^@/, "").replace(/^https?:\/\/github\.com\//, "").split("/")[0];
      store.write("fc.github", { user, repo: String(repo || "").trim() || "forkcast-course" });
      applyRepoLinks();
    },
    clear() { try { localStorage.removeItem("fc.github"); } catch (_) { /* nothing to clear */ } applyRepoLinks(); },
    configRepo() { const r = (window.FC_CONFIG || {}).repo || ""; return !r || /YOUR_GITHUB_USER/.test(r) ? null : r; },
    repo() { const g = this.get(); if (g && g.user) return `${g.user}/${g.repo || "forkcast-course"}`; return this.configRepo(); },
    templateRepo() { return this.configRepo() || this.repo(); },   // the course's own repo, for "Use this template"
  };
  FC.colabUrl = function (notebookPath, repo) {
    const cfg = window.FC_CONFIG || {};
    return `https://colab.research.google.com/github/${repo || FC.github.repo() || "YOUR_GITHUB_USER/forkcast-course"}/blob/${cfg.branch || "main"}/${notebookPath}`;
  };
  function setLink(a, url) {
    if (url) { a.href = url; a.classList.remove("needs-setup"); a.target = "_blank"; a.rel = "noopener"; a.removeAttribute("title"); }
    else { a.href = "#"; a.classList.add("needs-setup"); a.removeAttribute("target"); a.title = "Enter your GitHub user first"; }
  }
  function applyRepoLinks() {
    const repo = FC.github.repo(), tmpl = FC.github.templateRepo();
    document.querySelectorAll("[data-colab]").forEach((a) => setLink(a, repo ? FC.colabUrl(a.getAttribute("data-colab"), repo) : null));
    document.querySelectorAll("[data-repo-url]").forEach((a) => setLink(a, repo ? `https://github.com/${repo}` : null));
    document.querySelectorAll("[data-template-url]").forEach((a) => setLink(a, tmpl ? `https://github.com/${tmpl}` : null));
    document.querySelectorAll("[data-repo-name]").forEach((el) => { el.textContent = repo || "your-github-user/forkcast-course"; });
    document.querySelectorAll("[data-github-label]").forEach((el) => { el.textContent = repo ? repo : "Set your GitHub user"; });
    const g = FC.github.get() || {};
    document.querySelectorAll("[data-github-setup]").forEach((box) => {
      const u = box.querySelector("input[name='user']"), r = box.querySelector("input[name='repo']"), st = box.querySelector("[data-setup-status]");
      if (u && document.activeElement !== u) u.value = g.user || "";
      if (r && document.activeElement !== r) r.value = g.repo || "forkcast-course";
      if (st) st.innerHTML = repo
        ? `Links point at <b>github.com/${repo}</b>.`
        : `Nothing set yet: repository and Colab links stay inactive until you enter a GitHub user.`;
    });
  }
  function renderSetup(box) {
    if (box.querySelector("form")) return;
    box.innerHTML = `
      <form class="setup-form">
        <label>GitHub user <input type="text" name="user" placeholder="your-github-user" autocomplete="off" spellcheck="false"></label>
        <label>Repository <input type="text" name="repo" value="forkcast-course" autocomplete="off" spellcheck="false"></label>
        <div class="btn-row"><button type="submit" class="btn small">Save</button><button type="button" class="btn secondary small" data-setup-clear>Clear</button></div>
      </form>
      <p class="muted setup-status" data-setup-status></p>`;
    box.querySelector("form").addEventListener("submit", (e) => { e.preventDefault(); FC.github.set(box.querySelector("input[name='user']").value, box.querySelector("input[name='repo']").value); });
    box.querySelector("[data-setup-clear]").addEventListener("click", () => FC.github.clear());
  }
  function openSetupPanel() {
    const panel = document.getElementById("github-setup");
    if (!panel) return;
    panel.hidden = false;
    const input = panel.querySelector("input[name='user']");
    if (input) input.focus();
  }
  function initGithub() {
    document.querySelectorAll("[data-github-setup]").forEach(renderSetup);
    document.querySelectorAll("[data-github-toggle]").forEach((btn) => btn.addEventListener("click", () => {
      const panel = document.getElementById("github-setup");
      if (!panel) return;
      if (panel.hidden) openSetupPanel(); else panel.hidden = true;
    }));
    document.querySelectorAll("[data-setup-close]").forEach((btn) => btn.addEventListener("click", () => { const p = document.getElementById("github-setup"); if (p) p.hidden = true; }));
    document.addEventListener("click", (e) => {
      const a = e.target.closest && e.target.closest("a.needs-setup");
      if (a) { e.preventDefault(); e.stopPropagation(); openSetupPanel(); }
    }, true);
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") { const p = document.getElementById("github-setup"); if (p) p.hidden = true; } });
    applyRepoLinks();
  }

  /* ---------- quiz answers ---------- */
  function normalizeAnswer(text) {
    const t = String(text).toLowerCase();
    if (/error|fail|raise|crash|exception|mismatch/.test(t)) return "error";
    if (/^\s*-?\d{1,3}(,\d{3})+(\.\d+)?\s*$/.test(t)) return String(Number(t.replace(/[,\s]/g, "")));   // 51,813 is one number, not a shape
    const nums = t.replace(/−/g, "-").match(/-?\d+(?:\.\d+)?/g);
    if (nums) return nums.map((n) => String(Number(n))).join(",");
    const bare = t.replace(/torch\.size|[\[\]\s]/g, "");
    if (bare === "()" || bare === "" || /scalar|0-?d/.test(bare)) return "()";
    return bare;
  }

  /* ---------- per lesson: definition of done ---------- */
  function createDod(article, lesson) {
    const panel = article.querySelector("[data-dod-panel]");
    const dod = {
      setAuto(name, value, label) {
        const box = panel && panel.querySelector(`[data-dod-auto="${name}"]`);
        if (!box) return;
        box.checked = !!value;
        const note = box.parentElement.querySelector(".auto");
        if (note && label) note.textContent = `Tracked automatically on this page: ${label}.`;
        dod.update();
      },
      update() {
        if (!panel) return;
        const boxes = Array.from(panel.querySelectorAll("input[type='checkbox']"));
        const checked = boxes.filter((b) => b.checked).length;
        const done = boxes.length > 0 && checked === boxes.length;
        panel.classList.toggle("complete", done);
        article.querySelectorAll("[data-dod-count]").forEach((el) => { el.textContent = `${checked}/${boxes.length}`; el.classList.toggle("done", done); });
        article.querySelectorAll("[data-dod-status]").forEach((el) => { el.textContent = done ? "Lesson complete" : checked === 0 ? "Not started" : "In progress"; });
        const current = FC.progress.get(lesson);
        current.dod = {};
        boxes.forEach((b) => { current.dod[b.getAttribute("data-dod") || ("auto:" + b.getAttribute("data-dod-auto"))] = b.checked; });
        current.done = done;
        FC.progress.set(lesson, current);
        FC.refreshSyllabus();
      },
    };
    if (!panel) return dod;
    const saved = FC.progress.get(lesson);
    panel.querySelectorAll("input[type='checkbox'][data-dod]").forEach((box) => {
      box.checked = !!(saved.dod && saved.dod[box.getAttribute("data-dod")]);
      box.addEventListener("change", dod.update);
    });
    panel.querySelectorAll("input[type='checkbox'][data-dod-auto]").forEach((box) => {
      box.disabled = true;
      box.checked = !!(saved.dod && saved.dod["auto:" + box.getAttribute("data-dod-auto")]);
    });
    const reset = panel.querySelector("[data-dod-reset]");
    if (reset) reset.addEventListener("click", () => {
      if (!confirm("Clear your progress for this lesson on this browser?")) return;
      FC.progress.reset(lesson); location.reload();
    });
    dod.update();
    return dod;
  }

  /* ---------- per lesson: quizzes ----------
     <div class="quiz" data-quiz="q1" data-answer="1,32,4" data-explain="…" data-hint="…">
       <input type="text"> <button data-check> <p class="quiz-feedback">   */
  function initQuizzes(article, lesson, dod) {
    const quizzes = Array.from(article.querySelectorAll(".quiz[data-quiz]"));
    if (!quizzes.length) return;
    const saved = FC.progress.get(lesson);
    function updateSummary() {
      const total = quizzes.length, correct = quizzes.filter((q) => q.classList.contains("correct")).length;
      article.querySelectorAll("[data-quiz-progress]").forEach((el) => { el.textContent = `${correct} of ${total} questions answered correctly`; });
      dod.setAuto("quiz", correct === total, `${correct} of ${total}`);
    }
    quizzes.forEach((q) => {
      const id = q.getAttribute("data-quiz"), expected = normalizeAnswer(q.getAttribute("data-answer"));
      const input = q.querySelector("input[type='text']"), btn = q.querySelector("[data-check]"), fb = q.querySelector(".quiz-feedback");
      const explain = q.getAttribute("data-explain") || "", hint = q.getAttribute("data-hint") || "";
      let wrongCount = 0;
      const pretty = () => (expected === "error" ? "error" : expected === "()" ? "()" : `(${expected.split(",").join(", ")})`);
      function markCorrect(fromSave) {
        q.classList.add("correct");
        if (fb) { fb.className = "quiz-feedback ok"; fb.textContent = (fromSave ? "Answered. " : "Right. ") + explain; }
        if (input) input.disabled = true;
        if (btn) btn.disabled = true;
        const current = FC.progress.get(lesson); current.quizzes[id] = true; FC.progress.set(lesson, current);
        updateSummary();
      }
      function check() {
        if (!input) return;
        const given = normalizeAnswer(input.value);
        if (!input.value.trim()) { fb.className = "quiz-feedback"; fb.textContent = "Type an answer first, like (1, 32, 4), a number, or the word error."; return; }
        if (given === expected) { markCorrect(false); return; }
        wrongCount += 1;
        fb.className = "quiz-feedback bad";
        fb.textContent = "Not yet. " + (hint || "Walk through it one step at a time.");
        if (wrongCount >= 2 && !q.querySelector(".reveal")) {
          const reveal = document.createElement("button");
          reveal.type = "button"; reveal.className = "reveal"; reveal.textContent = "Show the answer";
          reveal.addEventListener("click", () => {
            fb.className = "quiz-feedback";
            fb.textContent = `The answer is ${expected === "error" ? "that it raises an error" : pretty()}. ${explain} Type it in to record it.`;
            reveal.remove();
          });
          fb.insertAdjacentElement("afterend", reveal);
        }
      }
      if (btn) btn.addEventListener("click", check);
      if (input) input.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); check(); } });
      if (saved.quizzes && saved.quizzes[id]) { if (input) input.value = pretty(); markCorrect(true); }
    });
    updateSummary();
  }

  /* ---------- per lesson: which section is current in the rail ---------- */
  function initSectionNav(article, lesson) {
    const links = Array.from(article.querySelectorAll(".rail nav a[data-section]"));
    const targets = links.map((a) => ({ a, el: document.getElementById(`l${lesson}-${a.getAttribute("data-section")}`) })).filter((t) => t.el);
    return function update() {
      if (article.hidden || !targets.length) return;
      const line = 140;   // a section is current once its heading has passed this many px from the top
      let current = targets[0];
      targets.forEach((t) => { if (t.el.getBoundingClientRect().top <= line) current = t; });
      targets.forEach((t) => t.a.classList.toggle("active", t === current));
    };
  }

  /* ---------- home view: syllabus statuses ---------- */
  FC.refreshSyllabus = function () {
    const all = FC.progress.all();
    document.querySelectorAll("[data-lesson-status]").forEach((el) => {
      const p = all[el.getAttribute("data-lesson-status")];
      el.classList.remove("done", "started");
      if (!p) { el.textContent = "Not started"; return; }
      const dod = p.dod || {}, keys = Object.keys(dod), checked = keys.filter((k) => dod[k]).length;
      if (p.done) { el.textContent = "Done"; el.classList.add("done"); }
      else if (checked > 0 || Object.keys(p.quizzes || {}).length > 0) { el.textContent = `In progress (${checked}/${keys.length} DoD)`; el.classList.add("started"); }
      else el.textContent = "Not started";
    });
  };

  /* ---------- router ---------- */
  const router = {
    current: null,
    parse(hash) {
      const h = (hash || "").replace(/^#/, "");
      if (!h || h === "home") return { view: "home" };
      const m = h.match(/^lesson-(\d\d)(?:\/([\w-]+))?$/);
      return m ? { view: "lesson", lesson: m[1], section: m[2] } : { view: "home" };
    },
    show(hash) {
      const r = this.parse(hash);
      const home = document.getElementById("home");
      const articles = document.querySelectorAll("article[data-lesson]");
      const where = document.querySelector("[data-where]");
      const total = ((window.FC_CONFIG || {}).lessonCount || 18) - 1;
      if (r.view === "lesson" && FC.lessons[r.lesson]) {
        const entry = FC.lessons[r.lesson];
        if (home) home.hidden = true;
        articles.forEach((a) => { a.hidden = a.getAttribute("data-lesson") !== r.lesson; });
        this.current = r.lesson;
        document.title = `Lesson ${Number(r.lesson)}: ${entry.article.getAttribute("data-title")} – Forkcast course`;
        if (where) where.textContent = `Lesson ${Number(r.lesson)} of ${total}`;
        const target = r.section ? document.getElementById(`l${r.lesson}-${r.section}`) : null;
        requestAnimationFrame(() => { if (target) target.scrollIntoView(); else window.scrollTo(0, 0); entry.updateNav(); });
      } else {
        if (home) home.hidden = false;
        articles.forEach((a) => { a.hidden = true; });
        this.current = null;
        document.title = "Forkcast course: deep learning and PyTorch by building";
        if (where) where.textContent = "";
        const target = r.view === "home" && hash && hash !== "#home" ? document.getElementById(hash.slice(1)) : null;
        requestAnimationFrame(() => { if (target) target.scrollIntoView(); else window.scrollTo(0, 0); });
      }
    },
  };
  FC.router = router;

  /* ---------- helpers for widgets ---------- */
  FC.parseShape = function (text) {
    const cleaned = String(text).replace(/[()\[\]\s]|torch\.Size/g, "");
    if (cleaned === "") return [];
    const nums = cleaned.split(",").filter((p) => p !== "").map((p) => Number(p));
    if (nums.some((n) => !Number.isInteger(n) || n < 0)) return null;
    return nums;
  };
  FC.dimColors = { n: "var(--dim-n)", d: "var(--dim-d)", w: "var(--dim-w)", y: "var(--dim-y)", c: "var(--dim-c)", s: "var(--dim-s)" };
  FC.dimNames = { n: "batch", d: "features", w: "weights", y: "outputs", c: "channels", s: "spatial" };
  FC.svgEl = function (tag, attrs, text) {
    const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
    Object.entries(attrs || {}).forEach(([k, v]) => el.setAttribute(k, v));
    if (text !== undefined) el.textContent = text;
    return el;
  };

  /* ---------- init ---------- */
  FC.init = function () {
    initTheme(); initCopyButtons(); initEnvBadges(); initGithub();
    document.querySelectorAll("article[data-lesson]").forEach((article) => {
      const lesson = article.getAttribute("data-lesson");
      const dod = createDod(article, lesson);
      initQuizzes(article, lesson, dod);
      FC.lessons[lesson] = { article, dod, updateNav: initSectionNav(article, lesson) };
    });
    document.querySelectorAll("[data-widget]").forEach((el) => { const name = el.getAttribute("data-widget"); if (FC.widgets[name]) FC.widgets[name](el); });
    FC.refreshSyllabus();
    let ticking = false;
    window.addEventListener("scroll", () => {
      if (ticking) return;
      ticking = true;
      requestAnimationFrame(() => { ticking = false; const cur = router.current; if (cur && FC.lessons[cur]) FC.lessons[cur].updateNav(); });
    }, { passive: true });
    window.addEventListener("hashchange", () => router.show(location.hash));
    document.addEventListener("click", (e) => {   // clicking the link for the hash we are already on still navigates
      const a = e.target.closest && e.target.closest('a[href^="#"]');
      if (a && a.getAttribute("href") === location.hash) { e.preventDefault(); router.show(location.hash); }
    });
    router.show(location.hash);
  };
})();
