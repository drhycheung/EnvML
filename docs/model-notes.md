# Model notes: the algorithms, and how the model reaches the browser

Technical companion to the [main README](../README.md).

Part 1 explains the algorithms from scratch, on the assumption you have never been taught
gradient boosting and are being asked to reason about whether this page's numbers mean
anything. Parts 2–4 cover the engineering: how scikit-learn's trees reach a browser with
no ML library, how that hand-off is proved correct, and the maintenance traps it creates.

## Contents

1. [The algorithms, in plain language](#1-the-algorithms-in-plain-language)
2. [Tree serialisation format](#2-tree-serialisation-format)
3. [The verification harness](#3-the-verification-harness)
4. [The vendored stylesheet](#4-the-vendored-stylesheet)
5. [Adding a feature safely](#5-adding-a-feature-safely)

---

## 1. The algorithms, in plain language

### 1.1 Why a baseline comes first

The single most useful habit in applied machine learning is to answer "how much better
than doing nothing?" before asking "is this model good?".

"Doing nothing" has to be spelled out, because *how* you spell it changes the answer. Here
are the two trivial predictors this page always shows:

| Baseline | What it does | Score on this data |
|---|---|---|
| Predict the mean | Ignore the inputs, always answer the average PM2.5 of the training set (98.6 µg/m³) | R² = 0.000 |
| Predict the majority class | Always answer "Moderate", the commonest air-quality band | Accuracy 40.66% |

**Why the mean-predictor scores exactly R² = 0, and why that is not a bug.** R² is defined
as

```text
R² = 1 − (sum of squared errors) / (sum of squared deviations from the mean)
```

The denominator is the total variance of the target. Always predicting the mean makes the
numerator *identical* to the denominator, so R² = 1 − 1 = 0. Any model that predicts
slightly worse than the mean scores negative. A negative R² therefore does not mean
"backwards" — it means the model is actively worse than a constant.

**Why 40.66% looks impressive and is not.** The three air-quality bands are not equally
common (Good 15,684 · Moderate 16,980 · Polluted 9,093), so a model that always answers
"Moderate" is right 40.66% of the time while knowing nothing at all. Any classifier you
build must be read against that number, not against 33% or against zero. This is why the
page prints both side by side.

### 1.2 Linear regression

For each training row, predict a weighted sum of the features and see how wrong you were:

```text
predicted PM2.5 = b₀ + b₁·TEMP + b₂·DEWP + … + b₁₀·wind_dir
```

The coefficients `b` are chosen to minimise the total squared error — "least squares",
hence **ordinary least squares (OLS)**. It is the first model worth trying for any numeric
target, and the reason it is on the page is not that it is good (R² 0.376) but that it is
the natural next step after the mean-predictor, and it makes the improvement attributable.

What it captures is a set of straight-line effects: "each degree colder adds so much",
"each month of winter adds so much". Its ceiling is that **it cannot bend**. Pollution
responds non-linearly — a small dew-point depression matters enormously (it means the air
is near saturation and secondary aerosol can form) while a large one barely matters, and a
linear model must apply the same slope everywhere.

### 1.3 Decision trees

A decision tree asks one question at a time and splits the data in two:

```text
Is TEMP − DEWP  ≤ 10.5 ?
    yes → ask the next question
    no  → ask the next question
…until it reaches a leaf, which stores a number.
```

That is the whole algorithm. It has two properties that matter here:

- **It can bend.** Every leaf is a separate number, so the model can say "cold *and* calm
  *and* December → 250" and "cold *and* windy *and* December → 80" with no difficulty. This
  is exactly the interaction linear regression cannot express.
- **It is interpretable.** You can read the rules off directly, which is why the first
  split in the real exported model (below) is a question about humidity.

A single deep tree, though, memorises the training data. It also predicts a *constant*
inside each leaf, so it can only ever be a piecewise-constant function.

### 1.4 Gradient boosting: many small trees, added together

Gradient boosting fixes both problems by refusing to use one tree.

1. Start from a constant — the mean of the target, **98.6132** here.
2. Fit a **small** tree to the current errors, not to the data itself.
3. Add a fraction of that tree's prediction to the running total.
4. Repeat, 250 times.

The fraction is the **learning rate**, `0.06`. Each tree is deliberately allowed to be
imperfect — it only has to nudge the estimate in the right direction, and the next tree
picks up what it missed. That is the "boosting": each tree learns from the previous one's
failure. The "gradient" is the formal reason it works: each tree is fitted to the negative
gradient of the loss function, which is the direction that most reduces the error.

Trees here have **`max_depth=4`**, so each one can have at most 2⁴ = 16 leaves — and in
practice far fewer, since splits stop when a split is no longer worth making. A depth-4
tree is a genuinely weak learner. Two hundred and fifty of them, added gently, are strong.
That is the whole trick.

**A real prediction, traced through the exported model.** The page's default input is
−5 °C air, −6 °C dew point, 1035 hPa, December, 20:00, calm. Here is the first of the 250
trees, followed by the arithmetic of the whole ensemble:

```text
walk through tree #1:
    TEMP_minus_DEWP = +1.00 ≤ 10.50   -> LEFT
    month_cos       = +1.00 >  0.25   -> RIGHT
    TEMP_minus_DEWP = +1.00 ≤  4.50   -> LEFT
    month_sin       = -0.00 > -0.25   -> RIGHT
    leaf value = 126.616

ensemble:
    98.6132  +  0.06 × (sum of 250 leaf values = 2121.079)
    = 98.6132 + 127.265
    = 225.88 µg/m³      <- exactly what the page displays
```

Read the first split aloud: *"how far is this air from saturation?"* A dew-point depression
of 1 °C is nowhere near the 10.5 °C threshold, so the model goes left. Then *"is this
December?"* — and `month_cos = +1.00` because December is the month in which the cyclical
encoding peaks. The model reached for humidity and season before anything else, which is
also what the feature-importance table says, and it matches atmospheric chemistry: Beijing's
winter heating season is when PM2.5 accumulates.

**Cost.** 250 trees × about 30 nodes each is **7,630 nodes** — around 640 KB of numbers.
Small enough to ship inside a web page, which is the whole reason this technique was
chosen over anything heavier.

### 1.5 Classification, and why it is a separate model

The regression answers *how much*. To answer *which band*, the page trains a second model
of the same family on the banded label instead of the concentration.

Internally, scikit-learn's `GradientBoostingClassifier` fits **one binary tree per class per
round**: 150 rounds × 3 classes = **450 trees**, in 13,852 nodes. Each tree's leaf value is
added to its own class's running score, and the class with the highest score wins.

Each class starts from its own prior in log-odds form, which is where the intercept comes
from:

```text
log(15684/41757) = -0.979226     log(16980/41757) = -0.899831     log(9093/41757) = -1.524362
```

Those three numbers are stored verbatim as `classifier.init`, and are exactly
`log(class frequency)` — verified to six decimal places. Because the scores live in
log-odds space (an unbounded scale where 0 means "no leaning either way"), the page takes an
`argmax` and never applies a softmax: softmax is monotonic, so it cannot change which
argument is largest.

**Three numbers appear in the classification panel, and they answer different questions:**

| Metric | Question it answers | Value |
|---|---|---|
| Accuracy | How often is the band right overall? | 69.70% |
| Balanced accuracy | How often is it right *in each band*? | 67.22% |
| Majority-class baseline | How often would "always say Moderate" be right? | 40.66% |

Balanced accuracy is the average of the per-band recalls. It is reported because the bands
are unequal — Polluted is 9,093 rows against Good's 15,684 — so a model could improve
overall accuracy simply by under-predicting the rare class.

**Reading the confusion matrix.** Rows are truth, columns are prediction, diagonal cells are
correct:

```text
              predicted →   Good   Moderate  Polluted
truth Good                  11,681     3,656       347
truth Moderate               2,568    12,614     1,798
truth Polluted                 348     3,935     4,810
```

The errors are almost entirely **between neighbouring bands**, never at the extremes: only
347 Good rows are called Polluted, and 348 Polluted rows are called Good. That is not a
weakness so much as the nature of the task. The 50 µg/m³ and 150 µg/m³ boundaries are
round numbers chosen by policy, not physical discontinuities in the atmosphere. A model
being unsure whether 149 is "Moderate" or "Polluted" is being asked an ill-posed question,
and the honest response is to be unsure.

### 1.6 Cross-validation: where the numbers come from

Every figure on this page is computed from **out-of-fold** predictions under 5-fold
shuffled cross-validation. Concretely: split the 41,757 rows into 5 folds; for each fold,
train on the other four and predict the held-out fifth; assemble all five held-out
prediction sets; only then compute R², RMSE, accuracy or the confusion matrix.

The reason is that a model scored on its own training data is scored on data it has already
memorised, and the score is meaningless. Out-of-fold scoring means every number is measured
on rows the model had never seen, which is the only way to estimate how it will behave on
tomorrow's data.

Two further checks guard against fooling ourselves:

- **Temporal holdout.** Train on 2010–2013, test on 2014 only: R² = 0.556, close to the
  0.533 from random folds. A large gap would mean the CV figure was an artefact of the
  splitting.
- **A deliberately leaky model.** See §1.8.

### 1.7 Feature engineering, and what it costs

**Why month and hour are sin/cos pairs.** Feeding `month = 12` as a number teaches the
model that December is "twelve times" something — so December would be treated as similar
to, but far from, January. Encoding the cycle instead gives December and January
neighbouring values and a smooth year-round path:

```text
month_sin = sin(2π · month / 12)      month_cos = cos(2π · month / 12)
```

December → (0, 1). January → (−0.5, 0.87). Adjacent, as they should be.

**Why dew-point depression is worth more than either input alone.** `TEMP − DEWP` measures
how far the air is from saturation, which is the condition under which fog and secondary
aerosol form. It is a derived quantity the model could not cheaply construct itself from
two linear terms, and it is the single most important feature at 0.394 of total importance
— comfortably ahead of dew point itself (0.105). Deriving the right feature beat handing
the learner more columns.

**What ordinal wind direction costs.** `cbwd` is coded NW=0, NE=1, SE=2, cv=3. That
ordering is arbitrary: the model will happily learn that cv is "more" than SE. The correct
encoding is one-hot (four separate 0/1 features), which would give up nothing to the model.
It is kept as an ordinal code because it keeps the tree arrays at 10 features and the
interactions the trees actually found were between wind and season rather than between
wind directions. **Flagged rather than hidden**: it is a simplification, and on a
predictive-accuracy-driven project it would be the first thing to change.

### 1.8 The leak: a higher score that means less

The dataset's `Iws` column looks like wind speed. It is a cumulative counter. Feeding it in
raises cross-validated R² from **0.533** to **0.555** — and both figures are recomputed on
every training run, so this is a measurement rather than a story.

The counter climbs steadily with elapsed time and resets when it overflows. A model given
it can partly infer *when in the record* a row sits, and Beijing's pollution is strongly
seasonal and strongly trending. That is not wind physics; it is time leaking through an
accumulator. The full investigation is in the [dataset card](dataset.md).

This is the most transferable lesson on the page: **when a model scores better than you
expected, the first hypothesis should be that you leaked something, not that you found
something.**

### 1.9 The prediction interval is empirical, not probabilistic

The page shows an 80% range. It is not a textbook confidence interval: it is the measured
10th and 90th percentile of the actual residuals of similar rows, bucketed by predicted
level because the error grows with the prediction.

The honest way to say it: *"among N historical records with similar conditions, the truth
landed in this range 80% of the time."* It says nothing about the probability of any single
future hour, and the page words it accordingly.

---

## 2. Tree serialisation format

scikit-learn fitted trees are objects with NumPy arrays inside. To run them in a browser
with no ML library, each tree is flattened to a single flat array at **stride 5**:

| Offset | Meaning |
|---|---|
| `+0` | split feature index; **`-1` marks a leaf** |
| `+1` | threshold |
| `+2` | left child, as an **offset** (multiple of 5) |
| `+3` | right child, as an offset |
| `+4` | leaf value |

The two ensembles are stored slightly differently, and the difference matters when you read
the file:

| | `regressor.trees` | `classifier.trees` |
|---|---|---|
| Shape | 250 bare flat arrays | 450 objects of `{ "cls": k, "tree": [...] }` |
| `init` | one scalar, 98.6132 (the mean) | three log-priors, `[-0.979, -0.900, -1.524]` |
| Grouping | none — all trees summed | `cls` says which class's score each tree feeds |
| Total nodes | 7,630 | 13,852 |

Three details are easy to get wrong, and none of them throw:

- **Child pointers are offsets, not indices.** `trees[o]` is reached with `o = 5 * child`.
  Storing indices instead produces a page that loads cleanly and predicts nonsense.
- **The comparison is `<=`.** `feature <= threshold` goes left, matching scikit-learn.
- **Child offsets are usually `0`.** Pointing at offset 0 means "back to the root", which
  is a valid reference to the tree's own start. Reading `0` as "node zero" is correct here
  only by coincidence of layout — which is exactly why the harness compares against a
  second implementation instead of trusting inspection.

Ensemble prediction:

```text
regression:   init_constant + learning_rate × sum(all tree leaf values)
classification (per class k): log(class_prior[k]) + learning_rate × sum(k's trees' leaf values)
```

Two rounding steps reduce `model.json` from ~1.4 MB to 640 KB: regressor leaves to 3
decimal places, classifier leaves to 5. The self-check quantifies the cost: maximum
deviation from a live `sklearn.predict` is **0.00073 µg/m³**, and class agreement is
**1.0000** over 2,000 rows. In other words, the rounding is free at the precision the page
displays.

### Why feature order is load-bearing

The flat arrays reference features **by index**, with no name attached. `FEATURES` in
`scripts/train_model.py` and `buildFeatures()` in `src/index.template.html` must stay in
the same order forever. Reordering one without the other silently scrambles the model: the
page keeps working, the numbers stay plausible, and every prediction is wrong.

`scripts/verify_page.py` compares feature *vectors* element by element, not just final
predictions, so a reordering fails loudly instead of degrading quietly.

---

## 3. The verification harness

`python3 scripts/verify_page.py` runs 26 checks in five groups and exits non-zero on failure.
Each check targets a failure that produces **no error message at all**.

| Group | What it proves |
|---|---|
| **[1] self-contained** | No remote `<script>`/`<link>`/`<img>`, no `@import`, no `fetch()`/XHR, no placeholder survived. Comments are stripped before scanning, so documentation that *names* the CDN does not trip it. |
| **[2] stylesheet coverage** | Every class the markup and the JS-built class lists rely on has a matching class selector in `src/app.css`. |
| **[3] JS ↔ Python parity** | The pure-model region is extracted and executed under Node.js, then diffed against an independent Python implementation over 66 inputs: defaults, all 60 samples, four corners of the input space, and one point deliberately outside it. |
| **[4] end-to-end render** | The **whole** script runs under a stub DOM. If `init()` throws, the build fails. Then the displayed numbers are checked against Python, both panels are checked for content, slider bounds are checked against the model, and the random-record button is fired. |
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
extrapolate to a finite constant by construction, and the check is that both languages do so
identically — not that the number is meaningful. It is not, and the page says nothing about
inputs beyond the trained range because there is nothing honest to say.

### Working on the harness

The pure-model region is delimited by explicit `PURE-MODEL CORE — BEGIN/END` comments in
`src/index.template.html`. Keep DOM-dependent code **outside** it; anything inside must be
side-effect free or `verify_page.py` cannot run it under Node. The stub in
`verify_page.py` is deliberately crude — auto-creating elements on demand — and only needs
to be faithful enough to let the real code paths execute.

---

## 4. The vendored stylesheet

The page uses Tailwind utility classes in its markup but loads no Tailwind. `src/app.css`
contains a **snapshot** of what the Tailwind Play CDN generated, captured once by loading
the page with the CDN active, exercising every interactive state, and reading the generated
`<style>` element out of the DOM. It is roughly 14 KB inlined, keeping the whole page at
about 686 KB with nothing to download.

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

To legitimately regenerate after adding classes: temporarily restore the CDN `<script>` tag
and the `tailwind.config` block in the template, load the page, click every control so the
MutationObserver sees the new classes, re-read the generated `<style>`, paste it over block
2 of `src/app.css`, remove the CDN tag again, and run `verify_page.py`.

---

## 5. Adding a feature safely

The ordering is not arbitrary — each step assumes the previous one is already correct.

1. **Add the column in `engineer()`** in `scripts/train_model.py`, and append it to
   `FEATURES`. Append, do not insert: inserting reorders the index mapping and silently
   invalidates any model you exported earlier.
2. **Mirror it in `buildFeatures()`** in `src/index.template.html`, in the same position,
   with a comment saying which Python line it corresponds to.
3. **Retrain** — `python3 scripts/train_model.py`. It prints the out-of-fold metrics and
   re-runs its own export self-check.
4. **Rebuild** — `python3 scripts/build_page.py`.
5. **Verify** — `python3 scripts/verify_page.py`. Check [3] compares feature vectors element
   by element, so a mismatch in count or order fails immediately.
6. **Check the new feature against the leakage lesson.** If it accumulates over time, or is
   measured at the same moment as the target, it is leakage no matter how good the score
   looks. Train it both ways and report both numbers, exactly as `Iws` is handled.

If the page's on-screen figures and the README ever disagree, the README is stale — the
page reads its numbers from `model/model.json` at runtime and cannot drift.