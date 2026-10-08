"""Assemble the whole course into one self-contained HTML file: docs/index.html.

Sources live under site/: template.html, home.html, lessons/NN-*.html (one
<article> per lesson), assets/css/course.css, assets/js/config.js,
assets/js/course.js and assets/js/widgets/*.js. The output is committed, so
GitHub Pages serves it as-is and the reference workflow checks it is current.

Usage:  python tools/build_site.py
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
OUT = ROOT / "docs" / "index.html"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8").rstrip("\n")


def main() -> None:
    template = read(SITE / "template.html")
    css = read(SITE / "assets/css/course.css")
    config = read(SITE / "assets/js/config.js")
    js_files = [SITE / "assets/js/course.js", *sorted((SITE / "assets/js/widgets").glob("*.js"))]
    js = "\n\n".join(read(p) for p in js_files)
    home = read(SITE / "home.html")
    lesson_files = sorted((SITE / "lessons").glob("*.html"))
    lessons = "\n\n".join(read(p) for p in lesson_files)
    html = (template.replace("{{CSS}}", css).replace("{{CONFIG}}", config)
            .replace("{{HOME}}", home).replace("{{LESSONS}}", lessons).replace("{{JS}}", js))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html + "\n", encoding="utf-8")
    ids = re.findall(r'\sid="([^"]+)"', html)
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise SystemExit(f"duplicate ids in the built page: {dupes}")
    print(f"{OUT.relative_to(ROOT)}: {len(lesson_files)} lessons, {len(html) // 1024} kB")


if __name__ == "__main__":
    main()
