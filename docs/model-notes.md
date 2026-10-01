# Model notes: serialisation, verification, and the vendored stylesheet

Technical companion to the [main README](../README.md). Covers how the model reaches the
browser without an ML library, how that hand-off is proved correct, and the two
maintenance traps this design creates.

## Contents

1. [Tree serialisation format](#1-tree-serialisation-format)
2. [The verification harness](#2-the-verification-harness)
3. [The vendored stylesheet](#3-the-vendored-stylesheet)
4. [Adding a feature safely](#4-adding-a-feature-safely)

---

## 1. Tree serialisation format

scikit-learn fitted trees are objects with NumPy arrays inside. To run them in a browser
with no ML library, each tree is flattened to a single flat array at **stride 5**:

| Offset | Meaning |
|---|---|
| `+0` | split feature index; **`-1` marks a leaf** |
| `+1` | threshold |
| `+2` | left child, as an **offset** (multiple of 5) |
| `+3` | right child, as an offset |
| `+4` | leaf value |

Three details are easy to get wrong, and none of them throw:

- **Child pointers are offsets, not indices.** `trees[o]` is reached with `o = 5 * child`.
  Storing indices instead produces a page that loads cleanly and predicts nonsense.
- **The comparison is `<=`.** `feature <= threshold` goes left, matching scikit-learn.
- **Child offsets are usually `0`.** Pointing at offset 0 means "back to the root", which
  is a perfectly valid reference to the tree's own start. Reading `0` as "node zero" is
  correct here only by coincidence of layout — which is exactly why the harness compares
  against a second implementation instead of trusting inspection.

Ensemble prediction:

```text
regression:  init_constant + learning_rate * sum(all tree leaf values)
classification (per class k): log(class_prior[k]) + learning_rate * sum(leaf values of k's trees)
```

The classifier emits **raw additive scores in log-odds space**. The page only ever needs
an `argmax`, so no softmax is applied — and none is needed, because softmax is monotonic
and cannot change an argmax.

Two rounding steps reduce `model.json` from ~1.4 MB to 640 KB: regressor leaves to 3
decimal places, classifier leaves to 5. The self-check quantifies the cost: maximum
deviation from a live `sklearn.predict` is **0.00073 µg/m³**, and class agreement is
**1.0000** over 2,000 rows. In other words, the rounding is free at the precision the page
displays.

### Why feature order is load-bearing

The flat arrays reference features **by index**, with no name attached. `FEATURES` in
`scripts/train_model.py` and `buildFeatures()` in `src/index.template.html` must stay in
the same order forever. Reordering one without the other silently scrambles the model:
the page keeps working, the numbers stay plausible, and every prediction is wrong.

`scripts/verify_page.py` compares feature *vectors* element by element, not just final
predictions, so a reordering fails loudly instead of degrading quietly.

---

## 2. The verification harness

`python3 scripts/verify_page.py` runs 26 checks in five groups and exits non-zero on the
first failing group. It is designed so that each check catches a failure that produces
**no error message at all**.

| Group | What it proves |
|---|---|
| **[1] self-contained** | No remote `<script>`/`<link>`/`<img>`, no `@import`, no `fetch()`/XHR, no placeholder survived. Comments are stripped before scanning, so documentation that *names* the CDN does not trip it. |
| **[2] stylesheet coverage** | Every class the markup and the JS-built class lists rely on has a matching class selector in `src/app.css`. |
| **[3] JS ↔ Python parity** | The pure-model region is extracted and executed under Node.js, then diffed against an independent Python implementation over 66 inputs: defaults, all 60 samples, four corners of the input space, and one point deliberately outside it. |
| **[4] end-to-end render** | The **whole** script runs under a stub DOM. If `init()` throws, the build fails. Then the displayed numbers are checked against Python, the evaluation and provenance panels are checked for content, slider bounds are checked against the model, and the random-record button is fired. |
| **[5] structural invariants** | No hard-coded `min`/`max` on the three measurement sliders; all 60 samples lie inside `input_ranges`; exactly 10 features; `Iws` is absent; the provenance note actually mentions the `Iws` decision. |

Three properties worth naming:

**The two implementations are independent.** `verify_page.py` deliberately does not import
`train_model.py`. It re-implements the feature builder and the tree traversal in plain
Python. If both sides shared code, agreement would be guaranteed and would prove nothing.
The same applies on the JS side: the Python evaluator never sees the JavaScript.

**Group [4] exists because groups [1]–[3] all passed on a broken build.** A missing
`let windDir` made `init()` throw, leaving every readout blank — while the page rendered
with full styling and every static check stayed green. Parity tests cannot see it, because
the broken variable lives only in the DOM-dependent half of the script. The stub DOM makes
"the page's JavaScript actually runs" a testable claim.

**Extrapolation is tested deliberately.** One case sits outside the training range. Trees
extrapolate to a finite constant by construction, and the check is that both languages do
so identically — not that the number is meaningful. It is not: it is a constant-prediction
region, and the page says nothing about inputs outside the trained range because there is
nothing honest to say.

### Working on the harness

The pure-model region is delimited by explicit `PURE-MODEL CORE — BEGIN/END` comments in
`src/index.template.html`. Keep DOM-dependent code **outside** it; anything inside must be
side-effect free or `verify_page.py` cannot run it under Node. The stub in
`verify_page.py` is deliberately crude — auto-creating elements on demand — and only needs
to be faithful enough to let the real code paths execute.

---

## 3. The vendored stylesheet

The page uses Tailwind utility classes in its markup, but loads no Tailwind. `src/app.css`
contains a **snapshot** of what the Tailwind Play CDN generated, captured once by loading
the page with the CDN active, exercising every interactive state, and reading the generated
`<style>` element out of the DOM. It is roughly 14 KB inlined, which keeps the whole page
at about 686 KB with nothing to download.

**The trade-off, stated plainly:** the page is now fully offline and double-clickable, but
the stylesheet no longer regenerates itself. Adding a utility class that is not already in
`src/app.css` produces an unstyled element with no error and no console message.

This is not hypothetical — it is the exact failure mode the design invites. It is caught by
check [2], which parses class selectors out of the stylesheet and cross-checks them against
every class the source uses.

**If you prefer not to maintain a snapshot**, write plain CSS by hand for new elements
instead of adding utilities. For a single-file teaching artefact this is usually clearer
anyway: the hand-written block at the top of `src/app.css` is where readable, semantic
styles belong, and it needs no regeneration step at all.

To legitimately regenerate after adding classes: temporarily restore the CDN `<script>`
tag and the `tailwind.config` block in the template, load the page, click every control so
the MutationObserver sees the new classes, re-read the generated `<style>`, paste it over
block 2 of `src/app.css`, remove the CDN tag again, and run `verify_page.py`.

---

## 4. Adding a feature safely

The ordering is not arbitrary — each step assumes the previous one is already correct.

1. **Add the column in `engineer()`** in `scripts/train_model.py`, and append it to
   `FEATURES`. Append, do not insert: inserting reorders the index mapping and silently
   invalidates any model you exported earlier.
2. **Mirror it in `buildFeatures()`** in `src/index.template.html`, in the same position,
   with a comment saying which Python line it corresponds to.
3. **Retrain** — `python3 scripts/train_model.py`. It prints the out-of-fold metrics and
   re-runs its own export self-check.
4. **Rebuild** — `python3 scripts/build_page.py`.
5. **Verify** — `python3 scripts/verify_page.py`. Check [3] compares feature vectors
   element by element, so a mismatch in count or order fails immediately.
6. **Check the new feature against the leakage lesson.** If it accumulates over time, or is
   measured at the same moment as the target, it is leakage no matter how good the score
   looks. Train it both ways and report both numbers, exactly as `Iws` is handled.

If the page's on-screen figures and the README ever disagree, the README is stale — the
page reads its numbers from `model/model.json` at runtime and cannot drift.