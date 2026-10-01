"""
Verify that the built page still behaves exactly like the training code.

This is the guard that turns three claims that are easy to assert and hard to
believe into tests that fail loudly:

  1. The page makes ZERO network requests, so a student can double-click
     index.html and it works offline from disk.

  2. Every CSS class used by the markup is actually implemented in
     src/app.css. Because app.css is a vendored SNAPSHOT of what the Tailwind
     CDN used to generate, adding a new utility class without regenerating it
     produces an unstyled element and no error at all.

  3. The JavaScript in index.html produces bit-comparable features and
     predictions to an independent Python implementation, for the default
     inputs, all 60 sample records, and the corners of the input space.

  4. The page's whole script executes end to end under a stub DOM, and the
     numbers it displays are the correct ones. This exists because checks 1-3
     all passed once on a build where the page was dead on arrival: a missing
     `let windDir` made init() throw while every read-out stayed blank.

Check 3 works by extracting the region between the PURE-MODEL CORE markers in
index.html, running it under Node.js, and diffing against Python. It does not
import train_model.py on purpose: the value comes from the two sides being
INDEPENDENT reimplementations, not from one calling the other.

Also re-asserts the invariant that the hardcoded slider bounds used to break:
every sampled record must lie inside input_ranges, so no "real record" can be
silently clamped by a slider that is too narrow.

Usage:  python3 scripts/verify_page.py
Exit code 0 = all checks passed.
"""

import json
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGE = os.path.join(ROOT, "index.html")
TPL = os.path.join(ROOT, "src", "index.template.html")
CSS = os.path.join(ROOT, "src", "app.css")
MODEL = os.path.join(ROOT, "model", "model.json")

CORE_BEGIN = "PURE-MODEL CORE — BEGIN"
CORE_END = "PURE-MODEL CORE — END"

# Numeric tolerance for cross-language comparison. Both sides evaluate the same
# IEEE-754 doubles in the same order, so this should be exact; the slack only
# absorbs any platform-level libm difference in sin/cos.
TOL = 1e-9

failures = []
notes = []


def check(ok, label, detail=""):
    if ok:
        print("  PASS  %s%s" % (label, ("  — " + detail) if detail else ""))
    else:
        print("  FAIL  %s%s" % (label, ("\n        " + detail) if detail else ""))
        failures.append(label)
    return ok


# ---------------------------------------------------------------------------
# 1. Offline / self-contained
# ---------------------------------------------------------------------------

def check_offline(out):
    print("\n[1] self-contained artefact")
    code = re.sub(r"/\*.*?\*/", " ", out, flags=re.S)
    code = re.sub(r"<!--.*?-->", " ", code, flags=re.S)

    remote = re.findall(r'<(?:script[^>]*\ssrc|link[^>]*\shref|img[^>]*\ssrc)'
                        r'\s*=\s*["\']https?://[^"\']+', code, re.I)
    check(not remote, "no remote <script>/<link>/<img> resources",
          "found: %s" % remote[:3] if remote else "0 resource tags")

    check("@import" not in code, "no CSS @import")

    check("/*__MODEL_DATA__*/" not in code and "/*__APP_CSS__*/" not in code,
          "no unsubstituted placeholders")

    # fetch/XHR would also be an off-machine dependency at runtime.
    check("fetch(" not in code and "XMLHttpRequest" not in code,
          "no fetch() / XMLHttpRequest in page code")


# ---------------------------------------------------------------------------
# 2. CSS coverage — the vendored stylesheet must actually implement every class
#    the markup and the JS string-built class lists rely on.
# ---------------------------------------------------------------------------

UTIL = re.compile(r"^(?:(?:hover|sm|md|lg|focus|dark):)?"
                  r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*(?:/[a-z0-9-]+)*$")
UTIL_ARBITRARY = re.compile(r"^(?:(?:hover|sm|md|lg|focus|dark):)?"
                            r"[a-z][a-z0-9-]*-\[[^\]\s]+\]$")


def css_classes(css):
    """Every class selector present in the stylesheet, un-escaped."""
    return {m.group(1).replace("\\", "")
            for m in re.finditer(r"\.((?:\\.|[A-Za-z0-9_-])+)", css)}


def used_classes(tpl):
    """
    Classes the source relies on.

    From HTML: class="..." attributes (this also picks up class attributes
    inside JS-built HTML strings). Plus every quoted JS string containing a
    space whose whitespace-separated tokens ALL look like utility classes --
    that "all tokens" condition is what stops prose and getElementById() args
    from being mistaken for class lists. Plus explicit classList.* arguments.
    """
    found = set()
    for attrs in re.findall(r'class="([^"]*)"', tpl):
        # The naive regex also matches JavaScript-concatenated attributes such as
        # '<td class="' + (i ? ... ) + ' ...', where the captured text is code
        # rather than markup. Those characters never occur inside a real class
        # name, so their presence identifies the capture as an expression to be
        # skipped -- its actual classes are picked up by the quoted-literal scan
        # below.
        if re.search(r"""[+?()'",]""", attrs):
            continue
        found.update(attrs.split())

    # A quoted JS string counts as a class list only if at least one of its
    # tokens actually looks like a utility (contains '-', ':' or '['). Without
    # that test, ordinary interface copy such as "Load a random real record"
    # is indistinguishable from a class list and gets reported as unstyled.
    for lit in re.findall(r"'([^'\n]*)'|\"([^\"\n]*)\"", tpl):
        s = lit[0] or lit[1]
        toks = s.split()
        if len(toks) < 2:
            continue
        shaped = [UTIL.match(t) or UTIL_ARBITRARY.match(t) for t in toks]
        if not any(shaped):
            continue
        if not all(shaped):
            continue
        if not any(re.search(r"[-:\[]", t) for t in toks):
            continue
        found.update(toks)

    for arg in re.findall(r"classList\.(?:add|toggle|remove)\(\s*['\"]([^'\"]+)['\"]", tpl):
        found.update(arg.split())

    return {c for c in found if UTIL.match(c) or UTIL_ARBITRARY.match(c)}


def check_css(tpl, css):
    print("\n[2] stylesheet coverage")
    covered, used = css_classes(css), used_classes(tpl)
    check(len(used) > 0, "class extraction found something to check",
          "%d classes used" % len(used))
    missing = sorted(used - covered)
    check(not missing,
          "every class used is implemented in src/app.css",
          "unstyled classes: %s\n        src/app.css is a vendored snapshot; see the "
          "regeneration notes at the top of that file." % missing if missing
          else "%d/%d classes covered" % (len(used), len(used)))


# ---------------------------------------------------------------------------
# 3. JavaScript vs Python parity
# ---------------------------------------------------------------------------

# Independent Python reimplementation of the page's feature builder and tree
# traversal. Deliberately plain Python: no pandas, no sklearn, no import of
# train_model.py.
DIR_CODE = {"NW": 0, "NE": 1, "SE": 2, "cv": 3}


def py_features(g):
    import math
    tau = 2 * math.pi
    return [g["TEMP"], g["DEWP"], g["TEMP"] - g["DEWP"], g["PRES"], g["PRES"] - 1013.25,
            math.sin(tau * g["hour"] / 24), math.cos(tau * g["hour"] / 24),
            math.sin(tau * g["month"] / 12), math.cos(tau * g["month"] / 12),
            DIR_CODE[g["cbwd"]]]


def py_reg_predict(model, x):
    r = model["regressor"]
    out = r["init"]
    for a in r["trees"]:
        o = 0
        while a[o] >= 0:
            # int() matters: child pointers round-trip through JSON as floats,
            # and Python refuses a float subscript. JavaScript silently coerces,
            # which is exactly the kind of language difference that hides a bug.
            o = 5 * int(a[o + 2] if x[int(a[o])] <= a[o + 1] else a[o + 3])
        out += r["learning_rate"] * a[o + 4]
    return out


def py_clf_scores(model, x):
    c = model["classifier"]
    score = list(c["init"])
    for item in c["trees"]:
        a = item["tree"]
        o = 0
        while a[o] >= 0:
            o = 5 * int(a[o + 2] if x[int(a[o])] <= a[o + 1] else a[o + 3])
        score[item["cls"]] += c["learning_rate"] * a[o + 4]
    return score


def extract_core(out):
    i = out.find(CORE_BEGIN)
    j = out.find(CORE_END)
    if i < 0 or j < 0:
        raise SystemExit("PURE-MODEL CORE markers not found in index.html — "
                         "src/index.template.html must keep them verbatim.")
    # Start after the comment that *opens* the BEGIN marker, and finish after
    # the comment that *closes* the END marker. Slicing straight to the marker
    # text would leave a dangling "/*" and a syntax error under Node.
    start = out.find("*/", i) + 2
    stop = out.find("*/", j) + 2
    return out[start:stop]


NODE_DRIVER = """
const cases = JSON.parse(process.argv[2]);
const res = cases.map(c => {
  const x = buildFeatures(c);
  const s = clfScores(x);
  return { f: x, pm: regPredict(x), cls: s.indexOf(Math.max.apply(null, s)) };
});
process.stdout.write(JSON.stringify(res));
"""


def check_parity(model, out):
    print("\n[3] JavaScript vs Python parity")

    # Build the case list: defaults, every sample record, and the corners.
    meta = model["meta"]
    cases = [{"label": "defaults", **meta["defaults"], "cbwd": "cv"}]
    for i, s in enumerate(model["samples"]):
        cases.append({"label": "sample-%02d" % i, "TEMP": s["TEMP"], "DEWP": s["DEWP"],
                      "PRES": s["PRES"], "month": s["month"], "hour": s["hour"],
                      "cbwd": s["cbwd"]})

    R = meta["input_ranges"]
    corners = [
        ("corner-cold-dry",   R["TEMP"][0], R["DEWP"][0], R["PRES"][0], 1,  0,  "NW"),
        ("corner-hot-humid",  R["TEMP"][1], R["DEWP"][1], R["PRES"][1], 7,  13, "SE"),
        ("corner-low-press",  R["TEMP"][0], R["DEWP"][1], R["PRES"][0], 12, 23, "cv"),
        ("corner-high-press", R["TEMP"][1], R["DEWP"][0], R["PRES"][1], 6,  12, "NE"),
        # Beyond the training range on purpose: the trees must extrapolate to a
        # finite value rather than throw or return NaN.
        ("beyond-range",      R["TEMP"][1] + 15, R["DEWP"][1] + 10, R["PRES"][1] + 40, 3, 9, "cv"),
    ]
    for label, t, dp, pr, mo, hr, wd in corners:
        cases.append({"label": label, "TEMP": t, "DEWP": dp, "PRES": pr,
                      "month": mo, "hour": hr, "cbwd": wd})

    payload = json.dumps([{k: v for k, v in c.items() if k != "label"} for c in cases])

    with tempfile.NamedTemporaryFile("w", suffix=".mjs", delete=False) as f:
        f.write(extract_core(out) + NODE_DRIVER)
        script = f.name
    try:
        proc = subprocess.run(["node", script, payload], capture_output=True, text=True)
    finally:
        os.unlink(script)

    if proc.returncode != 0:
        check(False, "node executed the extracted core",
              "exit %d\n%s" % (proc.returncode, proc.stderr.strip()[:600]))
        return
    check(True, "node executed the extracted core", "%d cases" % len(cases))

    js = json.loads(proc.stdout)

    worst_f = worst_p = 0.0
    bad_cls = []
    for case, got in zip(cases, js):
        x = py_features(case)
        worst_f = max(worst_f, max(abs(a - b) for a, b in zip(x, got["f"])))
        worst_p = max(worst_p, abs(py_reg_predict(model, x) - got["pm"]))
        s = py_clf_scores(model, x)
        if s.index(max(s)) != got["cls"]:
            bad_cls.append(case["label"])

    check(worst_f < TOL, "feature vectors agree", "max abs diff = %.3g" % worst_f)
    check(worst_p < TOL, "regression predictions agree", "max abs diff = %.3g µg/m³" % worst_p)
    check(not bad_cls, "classification argmax agrees",
          "disagreed on: %s" % bad_cls if bad_cls else "all %d cases" % len(cases))


# ---------------------------------------------------------------------------
# 4. End-to-end render under a minimal DOM stub
#
#    Checks 1-3 are static and all passed on a build where the page was in fact
#    completely dead: `let windDir` had gone missing, init() threw, and every
#    readout stayed blank. Nothing about that failure is visible to a parity
#    check, because the broken variable only exists in the DOM-dependent half
#    of the script.
#
#    So run the WHOLE script -- not just the pure core -- against a stub
#    document. If init() throws, Node exits non-zero and the build fails. Then
#    confirm the numbers that reached the page are the right ones, and fire the
#    random-record button to exercise the sample path.
# ---------------------------------------------------------------------------

DOM_SHIM = r"""
const els = {};
function makeEl(id) {
  const el = {
    id: id, textContent: '', innerHTML: '', className: '', value: '',
    min: null, max: null, step: null, type: '', style: {}, dataset: {},
    _handlers: {},
    addEventListener(ev, fn) { (this._handlers[ev] = this._handlers[ev] || []).push(fn); },
    classList: { _s: new Set(), add(c) { this._s.add(c); }, remove(c) { this._s.delete(c); },
                 toggle(c) { this._s.has(c) ? this._s.delete(c) : this._s.add(c); } },
  };
  return el;
}
const windButtons = ['NW', 'NE', 'SE', 'cv'].map(v => {
  const b = makeEl('wd-' + v); b.dataset.v = v; return b;
});
globalThis.document = {
  getElementById(id) { return els[id] || (els[id] = makeEl(id)); },
  querySelectorAll(sel) { return sel === '#in-cbwd button' ? windButtons : []; },
};
"""

RENDER_DRIVER = """
function txt(id) { const e = els[id]; return e ? e.textContent : null; }
const out = {
  pm: txt('out-pm'), cls: txt('out-class'), aqi: txt('out-aqi'),
  lo: txt('out-lo'), hi: txt('out-hi'), markerLeft: els['marker-bar']?.style.left,
  metricsLen: (els['metrics-panel'] || {}).innerHTML.length,
  provLen: (els['provenance'] || {}).innerHTML.length,
  headerRows: txt('hdr-rows'), capDEWP: txt('r-DEWP'),
  bounds: ['TEMP','DEWP','PRES'].map(k => els['in-'+k].min + '..' + els['in-'+k].max),
  noteLen: (els['out-note'] || {}).innerHTML.length,
};
// Fire the random-record button; this is the only path that touches the
// per-sample fields (day, observed_pm25, observed_class).
const hs = (els['btn-random'] || {})._handlers || {};
if (hs.click && hs.click.length) hs.click.forEach(fn => fn());
out.sampleLine = (els['sample-line'] || {}).innerHTML || '';
process.stdout.write(JSON.stringify(out));
"""


def extract_full_script(out):
    i = out.find(CORE_BEGIN)
    if i < 0:
        raise SystemExit("PURE-MODEL CORE markers not found in index.html")
    start = out.rfind("<script>", 0, i)
    end = out.rfind("</script>")
    if start < 0 or end < 0:
        raise SystemExit("could not delimit the main <script> block in index.html")
    return out[start + len("<script>"):end]


def check_render(model, out):
    print("\n[4] end-to-end render (DOM stub)")

    with tempfile.NamedTemporaryFile("w", suffix=".mjs", delete=False) as f:
        f.write(DOM_SHIM + "\n" + extract_full_script(out) + RENDER_DRIVER)
        script = f.name
    try:
        proc = subprocess.run(["node", script], capture_output=True, text=True)
    finally:
        os.unlink(script)

    if proc.returncode != 0:
        err = proc.stderr.strip()
        check(False, "init() and the render path execute without throwing",
              "node exit %d\n        %s" % (proc.returncode, err.splitlines()[-1] if err else ""))
        return
    check(True, "init() and the render path execute without throwing")

    got = json.loads(proc.stdout)
    meta = model["meta"]

    # What the page displayed must equal what Python computes for the same inputs.
    d = dict(meta["defaults"]); d["cbwd"] = "cv"
    expect = max(0.0, py_reg_predict(model, py_features(d)))
    try:
        shown = float(got["pm"])
    except (TypeError, ValueError):
        shown = None
    check(shown is not None and abs(shown - expect) < 0.05,
          "page readout equals the Python prediction for the default inputs",
          "page %s vs python %.4f" % (got["pm"], expect) if shown is not None
          else "readout was %r" % got["pm"])

    check(got["cls"] in meta["class_names"],
          "page displays a known class name", repr(got["cls"]))

    check(got["metricsLen"] > 2000 and got["provLen"] > 500,
          "evaluation panel and provenance panel both populated",
          "metrics %d chars, provenance %d chars" % (got["metricsLen"], got["provLen"]))

    R = meta["input_ranges"]
    # Compare numerically: Python renders -19.0 where JavaScript renders -19,
    # and a string comparison would flag that cosmetic difference as a failure.
    keys = ["TEMP", "DEWP", "PRES"]
    try:
        bounds = [[float(v) for v in b.split("..")] for b in got["bounds"]]
        want = [list(R[k]) for k in keys]
    except ValueError:
        bounds, want = None, None
    check(bounds == want, "slider bounds came from the model metadata",
          "got %s, want %s" % (got["bounds"],
                               ["%s..%s" % (R[k][0], R[k][1]) for k in keys]))

    check(got["noteLen"] > 40, "prediction-interval note rendered", "%d chars" % got["noteLen"])
    check(got["markerLeft"] not in (None, "", "0%"),
          "concentration marker positioned on the scale", repr(got["markerLeft"]))

    # The random-record path is the only consumer of the per-sample fields, so a
    # missing `day` would otherwise only surface after a user click.
    check("Measured" in got["sampleLine"] and "µg/m³" in got["sampleLine"],
          "random-record panel shows the measured value",
          got["sampleLine"][:70] + "..." if got["sampleLine"] else "(empty)")
    check(bool(got["headerRows"]) and got["headerRows"] == "{:,}".format(meta["rows_used"]),
          "header record count injected from the model", repr(got["headerRows"]))


# ---------------------------------------------------------------------------
# 5. Structural invariants
# ---------------------------------------------------------------------------

def check_invariants(tpl, model):
    print("\n[5] structural invariants")
    meta = model["meta"]

    # The regression guard for the bug this file was written for: hardcoded
    # slider bounds that were narrower than the data.
    for k in ["TEMP", "DEWP", "PRES"]:
        m = re.search(r'id="in-%s"[^>]*\bmin="' % k, tpl)
        check(not m, "in-%s has no hardcoded min attribute (bounds come from model)" % k)
        m = re.search(r'id="in-%s"[^>]*\bmax="' % k, tpl)
        check(not m, "in-%s has no hardcoded max attribute (bounds come from model)" % k)

    # Every sampled record must lie inside the advertised ranges, otherwise
    # loading it clamps the value and the "real record" is not real.
    R = meta["input_ranges"]
    outside = []
    for i, s in enumerate(model["samples"]):
        for k in ["TEMP", "DEWP", "PRES"]:
            if not (R[k][0] <= s[k] <= R[k][1]):
                outside.append("sample-%02d %s=%s not in %s" % (i, k, s[k], R[k]))
    check(not outside, "all 60 sample records fit inside input_ranges",
          outside[:3] if outside else "no clamping possible")

    check(set(meta["defaults"]) == {"TEMP", "DEWP", "PRES", "month", "hour"},
          "model defaults cover exactly the five slider inputs",
          str(sorted(meta["defaults"])))

    check(len(model["meta"]["features"]) == 10, "10 features",
          str(len(model["meta"]["features"])))
    check("Iws" not in model["meta"]["features"],
          "cumulative wind-speed counter is not a feature")

    # The provenance note shown on the page must actually mention the decision.
    check("Iws" in meta["wind_feature_note"],
          "wind-feature note in model.json explains the Iws decision")


def main():
    for p in (PAGE, TPL, CSS, MODEL):
        if not os.path.exists(p):
            raise SystemExit("missing required file: %s" % p)

    with open(PAGE) as f:
        out = f.read()
    with open(TPL) as f:
        tpl = f.read()
    with open(CSS) as f:
        css = f.read()
    with open(MODEL) as f:
        model = json.load(f)

    print("verifying %s (%.0f KB)" % (os.path.relpath(PAGE, ROOT), os.path.getsize(PAGE) / 1024))

    check_offline(out)
    check_css(tpl, css)
    check_parity(model, out)
    check_render(model, out)
    check_invariants(tpl, model)

    print()
    if failures:
        print("FAILED %d check(s):" % len(failures))
        for f in failures:
            print("  - %s" % f)
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())