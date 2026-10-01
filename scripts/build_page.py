"""
Assemble the deployable single-file page from its sources.

Two substitutions are performed on src/index.template.html:

  1. model/model.json  -> the placeholder in  const MODEL = ... /*__MODEL_DATA__*/ null
  2. src/app.css      -> the placeholder in  <style>/*__APP_CSS__*/</style>

The result, index.html, has no network dependencies of any kind: no CDN, no
build step, no backend. A student can double-click it from disk over file://.

Usage:  python3 scripts/build_page.py
"""

import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TPL = os.path.join(ROOT, "src", "index.template.html")
OUT = os.path.join(ROOT, "index.html")
MODEL = os.path.join(ROOT, "model", "model.json")
CSS = os.path.join(ROOT, "src", "app.css")

# Each entry: (placeholder that must exist verbatim in the template, file to inline)
SUBS = [
    ("/*__MODEL_DATA__*/ null", MODEL),
    ("/*__APP_CSS__*/", CSS),
]


def main():
    with open(TPL) as f:
        html = f.read()

    for marker, path in SUBS:
        if marker not in html:
            raise SystemExit(
                "Placeholder not found in src/index.template.html: %r\n"
                "The template and scripts/build_page.py are out of sync." % marker)
        with open(path) as f:
            html = html.replace(marker, f.read())

    out = html
    with open(OUT, "w") as f:
        f.write(out)

    # Sanity checks on the artefact we just produced. Cheap, and they catch the
    # failure modes that produce a page which *looks* fine but is broken.
    #
    # Comments are stripped before scanning. Otherwise these checks fire on the
    # explanatory prose in src/app.css, which names the CDN specifically to
    # document that it is no longer used -- and a guard that cries wolf gets
    # ignored, which is worse than having no guard at all.
    code = re.sub(r"/\*.*?\*/", " ", out, flags=re.S)      # CSS comments
    code = re.sub(r"<!--.*?-->", " ", code, flags=re.S)     # HTML comments

    problems = []
    remote = re.findall(r'<(?:script[^>]*\ssrc|link[^>]*\shref)\s*=\s*["\']https?://[^"\']+',
                        code, re.I)
    if remote:
        problems.append(
            "index.html loads %d remote resource(s), e.g. %s -- the page will not "
            "work offline or from file://." % (len(remote), remote[0]))
    if "/*__MODEL_DATA__*/" in code or "/*__APP_CSS__*/" in code:
        problems.append("an unsubstituted placeholder survived into index.html.")
    if problems:
        raise SystemExit("build produced a broken artefact:\n  - " + "\n  - ".join(problems))

    print("inlined %s (%.0f KB) and src/app.css (%.0f KB)"
          % (os.path.relpath(MODEL, ROOT), os.path.getsize(MODEL) / 1024,
             os.path.getsize(CSS) / 1024))
    print("wrote index.html (%.0f KB) — self-contained, zero network requests"
          % (os.path.getsize(OUT) / 1024))


if __name__ == "__main__":
    main()