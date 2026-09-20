"""Port of the ASu skill's build-asu-resume.mjs, because node is not installed on this box.

The upstream script only inlines the shared frame — base.css (which itself @imports
toolbar.css), toolbar.html and editor.js — into a content shell so the delivered file is
self-contained. Same markers, same order, same result; verified against the repo's own
`--check` invariant by rebuilding the committed template and diffing.

    python scripts/build_asu_resume.py <content shell> <output html> [--repo <ASu-skills>]
"""
from __future__ import annotations

import argparse
import pathlib
import sys


def _strip_final_eol(text: str) -> str:
    return text.rstrip("\r\n")


def _require(text: str, marker: str, replacement: str) -> str:
    if marker not in text:
        raise SystemExit(f"ASu 模板缺少构建标记：{marker!r}")
    return text.replace(marker, replacement, 1)


def build(shell_path: pathlib.Path, repo: pathlib.Path) -> str:
    frame = repo / "assets" / "frame"
    asu = frame / "asu"
    shell = shell_path.read_text(encoding="utf-8")
    eol = "\r\n" if "\r\n" in shell else "\n"

    def part(name: str) -> str:
        return _strip_final_eol((asu / name).read_text(encoding="utf-8").replace("\r\n", "\n").replace("\n", eol))

    def shared(name: str) -> str:
        return (frame / name).read_text(encoding="utf-8").replace("\r\n", "\n").replace("\n", eol)

    css = part("base.css").replace('@import url("../toolbar.css");', shared("toolbar.css"))
    toolbar = shared("toolbar.html").replace("</div>", part("toolbar.html") + "</div>", 1)
    editor = part("editor.js") + eol + shared("editor.js")

    out = _require(shell, f"  <base href=\"../\">{eol}", "")
    out = _require(out, '  <link rel="stylesheet" href="frame/asu/base.css">',
                   f"  <style>{eol}{css}{eol}  </style>")
    out = _require(out, "  <!-- @ASU_TOOLBAR -->", toolbar)
    out = _require(out, "  <!-- @ASU_EDITOR -->", f"  <script>{eol}{editor}{eol}  </script>")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("shell"); ap.add_argument("out")
    ap.add_argument("--repo", default="/tmp/claude-0/-root-probesql/"
                    "fff621cb-6c6f-4dc8-be8b-9a2553a2182c/scratchpad/ASu-skills")
    ap.add_argument("--check", action="store_true",
                    help="rebuild the repo's own committed template and diff, to prove the port")
    a = ap.parse_args()
    repo = pathlib.Path(a.repo)
    generated = build(pathlib.Path(a.shell), repo)
    if a.check:
        current = pathlib.Path(a.out).read_text(encoding="utf-8")
        if generated != current:
            for i, (x, y) in enumerate(zip(generated.splitlines(), current.splitlines())):
                if x != y:
                    print(f"first difference at line {i+1}:\n  built:  {x[:120]}\n  repo:   {y[:120]}")
                    break
            sys.exit(f"MISMATCH: built {len(generated)} chars, committed {len(current)}")
        print("port verified: byte-identical to the repo's committed template")
    else:
        pathlib.Path(a.out).write_text(generated, encoding="utf-8")
        print(f"wrote {a.out}  ({len(generated)} chars)")
