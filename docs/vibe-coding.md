# Reproducing the demo with vibe coding

Companion guide to the [main README](../README.md). This is the teaching pack for the
lesson: why the page is designed the way it is, how it was actually built with an AI
coding tool, and the complete prompt to hand to one.

**Last verified: 2 October 2026** against OpenCode, Node.js 26, scikit-learn 1.5.1, and
the deployed page at <https://drhycheung.github.io/EnvML/>.

## Contents

1. [Design thinking: from a dashboard that only looks back to a prediction you can act on](#1-design-thinking-from-a-dashboard-that-only-looks-back-to-a-prediction-you-can-act-on)
2. [How the page was built](#2-how-the-page-was-built)
3. [Further work for students](#3-further-work-for-students)
4. [The reproduction prompt](#4-the-reproduction-prompt) ← jump here if you just want to build it

---

## 1. Design thinking: from a dashboard that only looks back to a prediction you can act on

The page is the output of one design-thinking loop applied to a real teaching problem:
students already have a great deal of environmental data, and a great many dashboards for
displaying it. This is precisely why the data alone changes nothing.

| Stage | This project's arc |
|---|---|
| **1. Empathise** | The user pain: an environmental analyst or school administrator opens a monitoring dashboard and sees 41,757 hours of history rendered beautifully. Every question they actually have is forward-looking: *will tomorrow evening exceed 150? Should we issue a health advisory? Do we switch on the heaters? Should the outdoor sports lesson move indoors?* A chart cannot answer any of them, because a chart can only describe what has already happened. The data are abundant, but the decision remains unavailable. |
| **2. Define** | Problem statement: *we have the data but cannot make predictions, so the data is not very useful and does not lead to actions.* Design goal: the same data stream must produce a forward-looking estimate the user can act on — and must be honest enough that they trust it enough to act. |
| **3. Ideate** | Options considered: (a) a further monitoring dashboard — rejected, because that is what already exists and it cannot support a decision; (b) a single forecast number with no supporting information — rejected, because it cannot be acted upon; (c) a predicted value, an uncertainty range, and an actionable class; (d) a baseline comparison, so that the result can be assessed; (e) a panel stating which data could not be used, and why. Chosen: (c), (d) and (e), with the class threshold as the actionable output. |
| **4. Prototype** | The single-file page. Every choice was implemented: the prediction occupies the main position on the page, and the charts support it; a marker on the AQI scale showing where the estimate lands against the 50/150 thresholds that *are* the action triggers; an interval explained in words ("Among 5,220 historical records with similar test conditions, the measured PM2.5 fell in this range 80% of the time…"); side-by-side baselines; and a highlighted panel stating that a feature was excluded deliberately *because including it produced a higher score*. |
| **5. Test** | Parity tests between two independent implementations, an end-to-end test using a stub DOM, a browser test in which every request except the main document was blocked, and two faults that were detected by measurement but not by visual inspection — see section 2. |

The measurable outcome was **decision usefulness**: a user must be able to set tomorrow's
conditions, obtain an estimate, see whether it crosses an action threshold, and know how far
to trust it. Trustworthiness is a secondary requirement, and it is required for the same
reason — an estimate that a user does not trust will not be acted upon.

### Context: Monitor, Analyse, Control

This project covers the **analyse** and **control** stages of environmental informatics. The
**monitor** stage is covered by a separate project,
[EnvInfo](https://github.com/drhycheung/EnvInfo), which displays live air-quality
measurements. The design-thinking problem addressed here begins where that project stops:
once a monitoring system exists and the data are being displayed, the next question is
whether those data can support a decision about a future hour. A dashboard cannot answer
that question, and this project exists to answer it.

---

## 2. How the page was built

The page was built with OpenCode, driven through Playwright, using a method in which each
claim is measured before it is accepted: write the smallest useful version, measure it, and
accept a claim only after something independent has confirmed it. The prompt in
[Part 4](#4-the-reproduction-prompt) encodes the findings below, so that a working page
should be produced on the first attempt.

1. **Profile the data before modelling anything.** Checking `Iws` revealed negative
   one-hour differences and counter resets. That single check determined the model's
   feature set.
2. **Establish baselines first.** A dummy regressor and a linear model were computed
   before the real model, so every later number had something to be compared against.
3. **Measure the leak instead of asserting it.** The tempting claim is "we removed a leaky
   feature". Instead, a second gradient-boosting model was trained *with* `Iws` and
   cross-validated, so the page reports a measured value of 0.555 that is recalculated each
   time, rather than a value recorded once.
4. **Serialise the model, then re-derive it.** Trees are exported to flat arrays, and the
   prediction is recomputed from those arrays by two independent implementations:
   scikit-learn's `predict`, and a separately written Python traversal. If the two
   disagree, the serialisation is incorrect.
5. **Confirm that the browser and Python agree.** The pure-model region of the page is
   extracted and executed under Node.js, then diffed against an independent Python
   implementation across 66 inputs. Current agreement: features to 1.1e-16, regression
   predictions **exactly**.
6. **Confirm that the page is self-contained** by aborting every request except the top-level
   document and confirming the page still renders and predicts.

### Bugs that measurement caught and looking did not

All three produced a page that *looked finished*. None produced an error.

- **The dew-point slider was narrower than the data.** `min="-32" max="32"` was
  hard-coded while the training data reached −40 °C. Beijing winters routinely go below
  the slider's floor, so users could never enter a valid winter condition and the model
  could never be exercised at the edge of its own range. It was not detected by inspection because all 60 random samples happened to fall inside the slider: **the test cases were less demanding than the data**. Fixed by deleting every hard-coded bound and deriving them from
  `input_ranges` in the model file, with a structural test that fails if a literal
  `min`/`max` reappears on those inputs.

- **The page was completely dead, and all the static tests passed.** During a rewrite,
  `let windDir = 'cv'` was dropped. `init()` failed at the first statement, so every readout remained
  blank. The page was fully styled and the parity tests passed, because `windDir` is used
  only in the part of the script that depends on the page. The error was detected by the
  *browser*.
  Fixed by adding a DOM-stub end-to-end check that executes the whole script under Node:
  if `init()` throws, the build now fails.

- **A single-file page still required a network connection.** The first version loaded
  Tailwind from a CDN, so the page failed for any student who was offline or behind a
  restrictive network, while the README described it as self-contained. Fixed by storing the
  generated CSS inside the file. This introduced a second problem: the stylesheet became a
  *fixed copy*, so a newly added utility class would produce an element with no styling and
  no error message. `verify_page.py` now cross-checks every class the markup uses against the
  class selectors present in the stylesheet.

> [!IMPORTANT]
> These three faults form the central lesson of this project. In each case the page appeared
> complete, and every number on it was either limited to a narrower range than the data,
> absent, or dependent on a network that the reader might not have. An AI coding tool will
> produce all three, will describe the result as working, and will express confidence in it.
> The only reliable method is to state the expected result *before* running anything, and
> then to check it.

---

## 3. Further work for students

This project is deliberately **not** a research contribution. Predicting PM2.5 from
meteorological data is a well-established area, and the UCI dataset has been used for this
purpose for over a decade. That is intentional: this is a teaching baseline, not a
state-of-the-art result.

Students are encouraged to extend it, or to build something adjacent — a different city, a
different pollutant, or a different audience — so that their work addresses a question that
is genuinely not yet answered. Several extensions are suggested by the limitations listed in
the [main README](../README.md#7-known-limitations); the most direct of them are to add a
correct wind-speed measurement, to add an emissions inventory, and to test whether the
model remains valid in a different city or a later period.

---

## 4. The reproduction prompt

Give the prompt below to Gemini, OpenCode, Claude, ChatGPT or any coding agent. It encodes
every pitfall above, so a working page should come out first-pass.

```text
Build a complete, standalone, single-file HTML page for a Beijing PM2.5 prediction demo,
deployable on GitHub Pages. Everything inline, native ES6 only, no frameworks.

PURPOSE — read this before designing anything. Monitoring data is abundant and dashboards
to display it are easy, but a chart can only describe what ALREADY happened, so it never
leads to an action. Someone asking "will tomorrow evening exceed 150 µg/m³ — do we issue a
health advisory, move the sports lesson indoors, switch on the heaters?" cannot be helped
by any amount of history. This page closes that gap: the same data stream must produce a
forward-looking estimate the user can ACT on. Therefore the PREDICTION is the centre of the
page, in the prime visual position, with the charts supporting it and never competing with
it. A student must be able to set tomorrow's conditions, read a number, see whether it
crosses an action threshold, and know how far to trust it.

HARD CONSTRAINT — the finished index.html must make ZERO network requests. No CDN, no
web fonts, no fetch(), no XHR. A student must be able to double-click the file from disk,
offline, and have it work. Verify this by loading the page with every request EXCEPT the
top-level document aborted, and confirming it still renders and predicts. Do not report
success until that check passes.

STEP 0 — GET THE DATA FIRST. Assume the student does NOT have it. Never invent or fabricate a
dataset; download the real one and verify it before using it.
  mkdir -p data && curl -L -o /tmp/pm25.zip \
    "https://archive.ics.uci.edu/static/public/381/beijing+pm2+5+data.zip"
  unzip -o /tmp/pm25.zip -d data        # yields PRSA_data_2010.1.1-2014.12.31.csv
  mv data/PRSA_data_2010.1.1-2014.12.31.csv data/beijing_pm25.csv
  VERIFY, and stop if these do not match:
    wc -l data/beijing_pm25.csv   -> 43,825  (43,824 data rows + 1 header)
    head -1 data/beijing_pm25.csv -> No,year,month,day,hour,pm2.5,DEWP,TEMP,PRES,cbwd,Iws,Is,Ir
  The file is about 2.0 MB compressed, 2.0 MB raw, and is byte-identical to the copy in the
  repository. If the row count is not 43,825 you have the wrong file — do not proceed.
  Point to note: the legacy UCI path .../ml/machine-learning-databases/00381/BeijingPM2.5.data
  now returns 404 because the archive has been reorganised. Use the static/public URL above.
  Point to note: the column is named `pm2.5`. The full stop causes problems with attribute
  access in some libraries, so rename it to `pm25` when loading the file and record the
  change in a comment.
  Also download, or hand to the student, the dataset card:
    https://archive.ics.uci.edu/dataset/381/beijing+pm2+5+data
  Dataset: Beijing PM2.5 Data (Song et al., 2016), UCI Machine Learning Repository.
  43,824 hourly rows, 12 columns, 2010-01-01 to 2014-12-31.

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
  raise an error; it will return incorrect numbers without any indication that anything is
  wrong.

STEP 5 — the page. English throughout, including every code comment.
  - LEAD WITH THE ACTIONABLE OUTPUT. Show the predicted concentration and the air-quality
    class at the top, in the largest type on the page, with a one-sentence plain-English
    reading of what that class implies ("Advisory-level: sensitive groups should limit
    outdoor activity"). Place a marker on the AQI scale against the 50 and 150 µg/m³
    thresholds, because those thresholds ARE the decision the user is trying to make — make
    it visible whether the estimate is near one, over one, or between them. Charts come
    after this, never before it.
  - Support "will it be bad tomorrow?" directly: allow the user to set any month, hour,
    temperature, dew point, pressure and wind direction, and have the prediction update
    live. A user who can only replay historical rows has been given a dashboard, not a
    predictor.
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