"""Check the built course page: every internal link resolves and nothing local is missing.

Internal links are hashes: #home, #lesson-NN, or #lesson-NN/section (which
targets an element with id lNN-section). Runs in the reference workflow.

Usage:  python tools/check_site.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PAGE = Path(__file__).resolve().parent.parent / "docs" / "index.html"
SKIP = ("http://", "https://", "//", "mailto:", "data:", "javascript:")


def main() -> int:
    html = PAGE.read_text(encoding="utf-8")
    ids = set(re.findall(r'\sid="([^"]+)"', html))
    problems: list[str] = []
    for ref in re.findall(r'(?:href|src)="([^"]+)"', html):
        if ref.startswith(SKIP):
            continue
        if not ref.startswith("#"):
            if not (PAGE.parent / ref).exists():
                problems.append(f"missing local file: {ref}")
            continue
        target = ref[1:]
        if target in ("", "home") or target in ids:
            continue
        m = re.fullmatch(r"lesson-(\d\d)/([\w-]+)", target)
        if m and f"l{m.group(1)}-{m.group(2)}" in ids and f"lesson-{m.group(1)}" in ids:
            continue
        problems.append(f"unresolved link: #{target}")
    if problems:
        print("Problems in docs/index.html:")
        for p in sorted(set(problems)):
            print("  " + p)
        return 1
    lessons = len(re.findall(r'<article class="page" data-lesson=', html))
    print(f"docs/index.html: {lessons} lessons, all internal links resolve.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
