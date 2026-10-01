# Reproducing the demo with vibe coding

Companion guide to the [main README](../README.md). This is the teaching pack for the
lesson: why the page is designed the way it is, how it was actually built with an AI
coding tool, and the complete prompt to hand to one.

**Last verified: 2 October 2026** against OpenCode, Node.js 26, scikit-learn 1.5.1, and
the deployed page at <https://drhycheung.github.io/EnvML/>.

## Contents

1. [Design thinking: from a dirty CSV to an honest predictor](#1-design-thinking-from-a-dirty-csv-to-an-honest-predictor)
2. [How the page was actually built](#2-how-the-page-was-actually-built)
3. [The reproduction prompt](#3-the-reproduction-prompt) ← jump here if you just want to build it

---

## 1. Design thinking: from a dirty CSV to an honest predictor

The page is the output of one design-thinking loop applied to a real teaching problem:
a machine-learning demo that students will believe, and that will not teach them
something false.

| Stage | This project's arc |
|---|---|
| **1. Empathise** | The learner pain: ML demos usually show a number and a suspiciously high accuracy, with no baseline, no interval, and no admission of what the model cannot do. A student who reproduces such a demo learns that a bigger R² is simply better — which is how data leakage survives into production. |
| **2. Define** | Problem statement: *students need to feel what a model can and cannot infer from data, so the demo must make its own limits visible rather than hide them.* Design goal: the honest comparison is the interface, not a footnote. |
| **3. Ideate** | Options considered: (a) plain prediction readout; (b) readout + prediction interval; (c) baseline comparison; (d) leakage demonstration; (e) an explicit "what we could not use and why" panel. Chosen: (b)+(c)+(d)+(e) combined, with (d) as the centrepiece. The rejection of wind speed became the lesson rather than an omission. |
| **4. Prototype** | The single-file page. Every choice materialised: side-by-side baseline table, a marker on the AQI scale, an interval explained in words ("among 8,000 similar historical records…"), and a rose-coloured panel stating that a feature was deliberately discarded *because it scored better*. |
| **5. Test** | Cross-language parity tests, a DOM-stub end-to-end test, a browser pass with every non-document request aborted, and two real bugs caught by measurement rather than by looking — see §2. |

The measurable outcome was **trustworthiness**, not prettiness: a student who reads the
page should be able to state both what the model does well and where its 0.533 comes
from.

**Benchmark against the real thing**: the Beijing Municipal Ecological Environment
Monitoring Centre publishes authoritative hourly PM2.5. This project complements rather
than replaces it. The page says so, and frames its own numbers as "what machine learning
achieves on this dataset", not as a forecast.

> [!TIP]
> This project is deliberately **not** novel — pollution prediction from meteorology is a
> crowded field, and UCI's dataset has been used for a decade. That is by design: it is a
> teaching baseline. Students are encouraged to extend it, or build something adjacent —
> a different city, a different pollutant, a different audience — so that what they build
> brings genuinely unique value.

---

## 2. How the page was actually built

Built with OpenCode driven through Playwright, using a verify-first loop: write the
smallest thing, measure it, and only trust a claim once something independent confirms
it. The prompt in [Part 3](#3-the-reproduction-prompt) encodes the findings below so a
student gets a working result first-pass.

1. **Profile the data before modelling anything.** Checking `Iws` revealed negative
   one-hour differences and counter resets. That single check determined the model's
   feature set.
2. **Establish baselines first.** A dummy regressor and a linear model were computed
   before the real model, so every later number had something to be compared against.
3. **Measure the leak instead of asserting it.** The tempting claim is "we removed a leaky
   feature". Instead, a second gradient-boosting model was trained *with* `Iws` and
   cross-validated, so the page quotes a real measured 0.555 rather than a remembered
   constant.
4. **Serialise the model, then re-derive it.** Trees are exported to flat arrays, and the
   prediction is recomputed from those arrays by two independent implementations —
   scikit-learn's `predict` and a hand-written Python traversal. Disagreement means the
   serialisation is wrong.
5. **Prove the browser agrees with Python.** The pure-model region of the page is
   extracted and executed under Node.js, then diffed against an independent Python
   implementation across 66 inputs. Current agreement: features to 1.1e-16, regression
   predictions **exactly**.
6. **Prove the page is self-contained** by aborting every request except the top-level
   document and confirming the page still renders and predicts.

### Bugs that measurement caught and looking did not

All three produced a page that *looked finished*. None produced an error.

- **The dew-point slider was narrower than the data.** `min="-32" max="32"` was
  hard-coded while the training data reached −40 °C. Beijing winters routinely go below
  the slider's floor, so users could never enter a valid winter condition and the model
  could never be exercised at the edge of its own range. It survived review because all
  60 random samples happened to fall inside the slider — **the test data was kinder than
  the data**. Fixed by deleting every hard-coded bound and deriving them from
  `input_ranges` in the model file, with a structural test that fails if a literal
  `min`/`max` reappears on those inputs.

- **The page was completely dead, and all the static tests passed.** During a rewrite,
  `let windDir = 'cv'` was dropped. `init()` threw immediately, so every readout stayed
  blank — but the page was fully styled, and the parity tests were green, because
  `windDir` only exists in the DOM-dependent half of the script. The *browser* caught it.
  Fixed by adding a DOM-stub end-to-end check that executes the whole script under Node:
  if `init()` throws, the build now fails.

- **A single-file page still needed the network.** The first version pulled Tailwind from
  a CDN, which meant the page failed for any student offline or behind a restrictive
  network — while the README described it as self-contained. Fixed by vendoring the
  generated CSS inline. This introduced a subtler trap: the stylesheet became a *snapshot*,
  so a newly added utility class would render unstyled with no error. `verify_page.py`
  now cross-checks every class the markup uses against the stylesheet's class selectors.

> [!IMPORTANT]
> These three are the pedagogical heart of the lesson. In each case the page rendered
> beautifully and every number on it was either clamped, blank, or dependent on a network
> the reader might not have. An AI coding tool will hand you all three, describe the
> result as working, and be confident about it. The only defence is to state the expected
> result *before* running anything, then check it.

---

## 3. The reproduction prompt

Give the prompt below to Gemini, OpenCode, Claude, ChatGPT or any coding agent. It encodes
every pitfall above, so a working page should come out first-pass.

```text
Build a complete, standalone, single-file HTML page for a Beijing PM2.5 prediction demo,
deployable on GitHub Pages. Everything inline, native ES6 only, no frameworks.

HARD CONSTRAINT — the finished index.html must make ZERO network requests. No CDN, no
web fonts, no fetch(), no XHR. A student must be able to double-click the file from disk,
offline, and have it work. Verify this by loading the page with every request EXCEPT the
top-level document aborted, and confirming it still renders and predicts. Do not report
success until that check passes.

DATA: data/beijing_pm25.csv — UCI Beijing PM2.5 Data (Song et al., 2016). 43,824 hourly
rows for Beijing 2010-2014. Columns: No, year, month, day, hour, pm2.5, DEWP, TEMP,
PRES, cbwd (NW/NE/SE/cv), Iws, Is, Ir.

STEP 1 — profile the data BEFORE choosing features, and report what you find:
  - pm2.5 is missing in 4.72% of rows; drop them, leaving 41,757 rows.
  - Iws is a CUMULATIVE wind-speed counter, not an instantaneous reading. Measure the
    within-year distribution of one-hour differences. Expect ~18.9% negative (8,283 rows)
    and ~1,301 obvious resets (diff < -20); naive differencing yields ~-489 m/s.
  - CONSEQUENCE: the true instantaneous wind speed cannot be recovered. DO NOT use Iws as
    a feature, and DO NOT call it wind speed anywhere in the UI. Wind speed is the
    variable a reader will most expect to see; its absence must be stated explicitly in
    the page with the reason, not silently omitted.
  - Measure the leak rather than asserting it: train a second gradient-boosting regressor
    WITH Iws, cross-validate it, and report both R2 figures. Expect the clean model near
    0.533 and the leaky one near 0.555. Put BOTH numbers on the page and explain that the
    higher score comes from time accumulating in a counter, not from physics. A score
    bought with leakage is worse than a lower honest score.

STEP 2 — features. Exactly these 10, in this order (the tree arrays index them):
  TEMP, DEWP, TEMP-DEWP, PRES, PRES-1013.25, sin(2*pi*hour/24), cos(2*pi*hour/24),
  sin(2*pi*month/12), cos(2*pi*month/12), wind_dir (NW=0, NE=1, SE=2, cv=3)
  Do NOT include: the co-measured pollutants SO2/NO2/CO/O3 (same-hour, so predicting PM2.5
  from its own correlates is leakage), year, day, or No. Sin/cos rather than raw month and
  hour so that December and January are adjacent.

STEP 3 — models (scikit-learn, seed 42). Before fitting anything, compute the BASELINES:
  - regression: DummyRegressor (predict the mean). Expect R2 = 0 by construction.
  - classification: DummyClassifier(most_frequent). Expect ~40.7%.
  Then GradientBoostingRegressor(n_estimators=250, max_depth=4, learning_rate=0.06) and
  GradientBoostingClassifier(n_estimators=150, max_depth=4, learning_rate=0.08).
  ALL reported metrics must be OUT-OF-FOLD under 5-fold shuffled CV, never training fit.
  Also: train on year <= 2013 and test on 2014 to confirm the relationships are stable in
  time (expect R2 ~0.556).
  And report the classification accuracy you get by thresholding the regression output
  instead of classifying directly (expect ~66.9% vs ~69.7%) — the gap is the cost of
  boundary flips, and it is a lesson about task design, not about model quality.

STEP 4 — export the trees so no ML library is needed in the browser. Flatten each tree to
  stride 5: [feature, threshold, left, right, value]. A leaf has feature === -1. Child
  pointers are OFFSETS (multiples of 5), not indices. The browser rule is
  `feature <= threshold` goes LEFT — note `<=`, matching scikit-learn. Then IMMEDIATELY
  re-derive the predictions from the exported arrays with a hand-written traversal and
  diff against sklearn's own predict. Expect max abs diff < 0.001 for the regressor and
  1.0000 class agreement. A mismatch means the serialisation is wrong, and it will not
  throw — it will just predict quietly wrong numbers forever.

STEP 5 — the page. English throughout, including every code comment.
  - Slider bounds MUST be derived at runtime from the model's own input_ranges. Do NOT
    hard-code min/max on the range inputs: a slider narrower than the training data is a
    silent failure — the user cannot enter a real condition and a random real record gets
    clamped before the model sees it. Assert in your tests that every sample row lies
    inside the advertised ranges.
  - A prominent evaluation panel: the three-way regression table (baseline / linear /
    model), the classification table with the thresholding contrast, the full confusion
    matrix with row percentages, and feature importances with a plain-language reading.
  - An 80% prediction interval from empirical out-of-fold residual quantiles, bucketed by
    predicted level, explained in words: "among N similar historical records, the truth
    landed in this range 80% of the time". Do not present it as a probabilistic interval.
  - A "load a random real record" button cycling through 60 held-out rows, showing the
    measured value beside the prediction and the error. Use a seeded/deterministic cycle,
    not Math.random, so the demo and its tests are reproducible.
  - A provenance panel stating the dataset, the citation, the exact hyperparameters, and
    the external-validity limit: this model learned Beijing 2010-2014 only and is a
    demonstration of method, NOT a forecast, and certainly not a forecast for anywhere
    else.

STEP 6 — if you use a utility-class CSS framework (Tailwind, Bootstrap), do NOT leave it
on a CDN. Load the page with the CDN active, exercise EVERY interactive state so its
MutationObserver emits rules for classes that only appear after JS inserts them (drag
every slider, click every wind-direction button, click the random-record button), then
read the generated <style> element out of the DOM and vendor it into a separate
src/app.css with a header explaining that it is a SNAPSHOT and how to regenerate it.
Note the consequence in that file: adding a new utility class without regenerating yields
an unstyled element and NO error — so your test suite must cross-check every class used
in the markup against the class selectors present in the stylesheet.

STEP 7 — do not stop until all of the following are true, and report each as PASS/FAIL
with the actual measured number, not a summary:
  1. Extract the pure-model region of index.html (mark it with explicit
     BEGIN/END comment markers), run it under Node.js, and diff its features and
     predictions against an INDEPENDENT Python implementation across at least the
     default inputs, all 60 samples, and the corners of each input range. Two
     implementations that agree by construction prove nothing — write the Python side
     without importing your training code.
  2. Execute the ENTIRE page script under a minimal stub DOM and confirm init() does not
     throw. Static parity tests cannot catch an error in the DOM-dependent half: a build
     once shipped where a missing `let windDir` left every readout blank while the page
     looked styled and every static test passed.
  3. Load the page in a browser with all non-document requests aborted. Confirm zero
     blocked-needed requests and zero console errors.
  4. Confirm the page's displayed number equals the Python-computed number for the
     default inputs.
  5. Confirm no horizontal overflow at 390px and at 1280px.
  6. Confirm every class used by the markup is implemented in the stylesheet.
```

---

Back to the [main README](../README.md) ·
Model and verification notes: [model-notes.md](model-notes.md) ·
Dataset card: [dataset.md](dataset.md)