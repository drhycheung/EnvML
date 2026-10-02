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

The most useful habit in applied machine learning is to answer the question "how much
better than doing nothing?" before asking "is this model good?"

"Doing nothing" must be defined precisely, because the definition changes the answer. The
page always displays two such reference predictors:

| Baseline | What it does | Score on this data |
|---|---|---|
| Predict the mean | Ignore the inputs, always answer the average PM2.5 of the training set (98.6 µg/m³) | R² = 0.000 |
| Predict the majority class | Always answer "Moderate", the commonest air-quality band | Accuracy 40.66% |

**Why the mean-predictor scores exactly R² = 0, and why this is the expected result.** R² is defined
as

```text
R² = 1 − (sum of squared errors) / (sum of squared deviations from the mean)
```

The denominator is the total variance of the target. Always predicting the mean makes the
numerator *identical* to the denominator, so R² = 1 − 1 = 0. Any model that predicts
slightly worse than the mean scores negative. A negative R² does not mean that the model has predicted in the wrong direction. It means
that the model performs worse than a constant.

**Why 40.66% appears high but indicates no useful ability.** The three air-quality bands do
not occur with equal frequency (Good 15,684 · Moderate 16,980 · Polluted 9,093), so a model
that always answers "Moderate" is correct 40.66% of the time while using no information at
all. Any classifier must therefore be assessed against that figure, and not against 33% or
against zero. The page prints both values together for this reason.

### 1.2 Linear regression

For each training row, predict a weighted sum of the features and see how wrong you were:

```text
predicted PM2.5 = b₀ + b₁·TEMP + b₂·DEWP + … + b₁₀·wind_dir
```

The coefficients `b` are chosen to minimise the total squared error — "least squares",
hence **ordinary least squares (OLS)**. It is the first model to try for any numeric target. It appears on the page not because it
is accurate (R² 0.376) but because it is the natural next step after the mean-predictor,
and it makes the size of the improvement attributable to the model family.

What it captures is a set of straight-line effects: "each degree colder adds this much",
"each month of winter adds this much". Its limitation is that **it cannot represent curved
relationships**. Pollution responds non-linearly: a small dew-point depression matters greatly,
because the air is then close to saturation and secondary aerosol can form, while a large
depression matters little. A linear model must apply the same slope in both cases.

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
  split in the exported model is a question about humidity.

A single deep tree, however, memorises the training data. It also predicts a *constant*
within each leaf, so it can only ever represent a piecewise-constant function.

### 1.4 Gradient boosting: many small trees, added together

Gradient boosting addresses both problems by using not one tree but many.

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
practice far fewer, since splits stop when a split is no longer worth making. A depth-4 tree is therefore a deliberately weak learner. Two hundred and fifty of them, added
in small increments, are strong. This is the central idea.

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

The first split asks: *"how far is this air from saturation?"* A dew-point depression
of 1 °C is nowhere near the 10.5 °C threshold, so the model goes left. Then *"is this
December?"* — and `month_cos = +1.00` because December is the month in which the cyclical
encoding peaks. The model therefore considers humidity and season before any other variable. This agrees with
the feature-importance table and with atmospheric chemistry: PM2.5 accumulates in Beijing
during the winter heating season.

**Size.** 250 trees × about 30 nodes each gives **7,630 nodes**, approximately 640 KB of
numbers. This is small enough to store inside a web page, which is the reason this technique
was chosen in preference to a heavier one.

### 1.5 Classification, and why it is a separate model

The regression answers *how much*. To answer *which band*, the page trains a second model
of the same family on the banded label instead of the concentration.

Internally, scikit-learn's `GradientBoostingClassifier` fits **one binary tree for each class
in each round**: 150 rounds × 3 classes = **450 trees**, in 13,852 nodes. Each tree's leaf value is
added to its own class's running score, and the class with the highest score wins.

Each class starts from its own prior in log-odds form, which is where the intercept comes
from:

```text
log(15684/41757) = -0.979226     log(16980/41757) = -0.899831     log(9093/41757) = -1.524362
```

Those three numbers are stored verbatim as `classifier.init`, and are exactly
`log(class frequency)`, verified to six decimal places. Because the scores are held in
log-odds space, an unbounded scale on which 0 means an equal probability for each class, the
page takes an `argmax` and does not apply a softmax: a softmax is monotonic and cannot change
which value is the largest.

**Three numbers appear in the classification panel, and they answer different questions:**

| Metric | Question it answers | Value |
|---|---|---|
| Accuracy | How often is the band right overall? | 69.70% |
| Balanced accuracy | How often is it right *in each band*? | 67.22% |
| Majority-class baseline | How often would "always say Moderate" be right? | 40.66% |

Balanced accuracy is the mean of the per-band recall values. It is reported because the bands
are unequal in size: Polluted has 9,093 records against Good's 15,684, so a model could
raise overall accuracy simply by predicting the rarest class too rarely.

**Reading the confusion matrix.** Rows are truth, columns are prediction, diagonal cells are
correct:

```text
              predicted →   Good   Moderate  Polluted
truth Good                  11,681     3,656       347
truth Moderate               2,568    12,613     1,799
truth Polluted                 348     3,935     4,810
```

Almost all of the errors occur **between neighbouring bands**, and almost none occur at the
extremes: only 347 Good records are classified as Polluted, and 348 Polluted records as Good.
This reflects the nature of the task rather than a fault in the model. The 50 µg/m³ and
150 µg/m³ boundaries are values chosen by policy and do not correspond to any physical
discontinuity in the atmosphere. A model that is uncertain whether 149 µg/m³ is "Moderate"
or "Polluted" has been given a question with no precise answer, and the appropriate response
is to report the uncertainty.

### 1.6 Cross-validation: where the numbers come from

Every figure on this page is computed from **out-of-fold** predictions under 5-fold
shuffled cross-validation. Concretely: split the 41,757 rows into 5 folds; for each fold,
train on the other four and predict the held-out fifth; assemble all five held-out
prediction sets; only then compute R², RMSE, accuracy or the confusion matrix.

A model evaluated on its own training data has already seen the data it is being tested on,
so the resulting score does not indicate performance on new data. Out-of-fold scoring
measures every figure on records the model has not seen, which is the only basis on which its
future performance can be estimated.

Two further checks reduce the risk of accepting a misleading figure:

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
It is retained as an ordinal code in order to keep the tree arrays at 10 features, because the
interactions the trees found were between wind and season rather than between wind
directions. This is a deliberate simplification and is stated here rather than concealed: on a
project whose objective were predictive accuracy, it would be the first change made.

### 1.8 The leak: a higher score that means less

The dataset's `Iws` column looks like wind speed. It is a cumulative counter. Feeding it in
raises cross-validated R² from **0.533** to **0.555** — and both figures are recomputed on
every training run, so this is a measurement rather than a story.

The counter climbs steadily with elapsed time and resets when it overflows. A model given
it can partly infer *when in the record* a row sits, and Beijing's pollution is strongly
seasonal and strongly trending. That is not wind physics; it is time leaking through an
accumulator. The full investigation is in the [dataset card](dataset.md).

This is the most broadly applicable lesson in the project: **when a model performs better
than expected, the first hypothesis to test is that information has leaked, rather than that
something new has been found.**

### 1.9 Reproducibility: two perfectly duplicated features

`PRES` and `PRES_minus_1013` are related by `PRES = PRES_minus_1013 + 1013.25`. Their
correlation is 1.0 to twelve decimal places, and subtracting one from the other and
recovering the constant is exact to floating-point precision. One of the two features is
therefore mathematically redundant, and the model cannot do anything with the pair that it
could not do with either feature alone.

That redundancy has a consequence which is easy to miss. At any node where both features
offer an equally good split, the tree has no reason to prefer one over the other, and
scikit-learn resolves the tie by considering candidate features in a **random order**. That
order is drawn from a generator seeded by `random_state`; when `random_state` is `None` it is
seeded from operating-system entropy. The result is that `PRES` and `PRES_minus_1013` swap
places between training runs.

The effect is small but real. Across repeated runs of an otherwise identical script, the
feature importance figures, the RMSE in the third decimal place, and individual cells of the
confusion matrix all moved. No published figure could be reproduced, and a student who
retrained would obtain slightly different numbers from those printed in the README.

Setting `random_state=42` on both estimators resolves it. Two consecutive runs of
`train_model.py` now produce byte-for-byte identical output, which is the property the
project claims and the property a teaching example requires.

**The general lesson:** a fixed seed is not sufficient on its own. A model is reproducible
only if it is also *deterministic*, and any two features that are exact linear
transformations of each other create ties that a random tie-break can resolve differently on
every run. The correct response to a genuinely redundant feature is to remove one of the two.
Both are retained here because they cost nothing at prediction time and because the resulting
importance figures (0.028 and 0.029) demonstrate the tie directly — the two split the
importance between them almost evenly, which is what a redundant pair should do. On a
predictive-accuracy project, one of them would be deleted.

### 1.10 The prediction interval is empirical, not probabilistic

The page shows an 80% range. It is not a textbook confidence interval: it is the measured
10th and 90th percentile of the actual residuals of similar rows, bucketed by predicted
level because the error grows with the prediction.

The honest way to say it: *"among N historical records with similar conditions, the truth
landed in this range 80% of the time."* It says nothing about the probability of any single
future hour, and the page words it accordingly.

---

## 2. Tree serialisation format

A fitted scikit-learn tree is an object containing NumPy arrays. In order to run it in a
browser without a machine-learning library, each tree is converted to a single flat array at
**stride 5**:

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

Three details are easily implemented incorrectly, and none of them produces an error message:

- **Child pointers are offsets, not indices.** `trees[o]` is reached with `o = 5 * child`.
  Storing indices instead produces a page that loads cleanly and predicts nonsense.
- **The comparison is `<=`.** `feature <= threshold` goes left, matching scikit-learn.
- **A child offset is frequently `0`.** An offset of 0 refers to the start of the same tree,
  so it is a valid reference to that tree's own first node. Treating `0` as "node zero" gives
  the correct result here only because of the memory layout. This is precisely why the
  verification harness compares against a second implementation rather than relying on
  inspection.

Ensemble prediction:

```text
regression:   init_constant + learning_rate × sum(all tree leaf values)
classification (per class k): log(class_prior[k]) + learning_rate × sum(k's trees' leaf values)
```

Two rounding operations reduce `model.json` from approximately 1.4 MB to 640 KB: regressor
leaf values are rounded to 3 decimal places and classifier leaf values to 5. The export
self-check quantifies the cost: the maximum deviation from a live `sklearn.predict` is
**0.00073 µg/m³**, and class agreement is **1.0000** over 2,000 records. At the precision the
page displays, this rounding therefore has no practical effect.

### Why feature order is load-bearing

The flat arrays reference features **by index**, with no name attached. `FEATURES` in
`scripts/train_model.py` and `buildFeatures()` in `src/index.template.html` must stay in
the same order forever. Reordering one without the other silently scrambles the model: the
page keeps working, the numbers stay plausible, and every prediction is wrong.

`scripts/verify_page.py` compares feature *vectors* element by element, not just final
predictions, so a reordering of the feature list is reported as an error rather than
producing incorrect results.

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
`train_model.py`. It re-implements the feature builder and the tree traversal separately in
Python. If both sides used the same code, agreement would be guaranteed and would therefore
demonstrate nothing. The same principle applies on the JavaScript side: the Python
evaluator never executes the page's JavaScript.

**Group [4] exists because groups [1] to [3] all passed on a broken build.** A missing
`let windDir` caused `init()` to throw, leaving every readout blank, while the page rendered
with complete styling and every static check passed. A parity test cannot detect this,
because the affected variable is used only in the part of the script that depends on the
page. The stub DOM makes "the page's JavaScript runs to completion" a claim that can be
tested.

**Extrapolation is tested deliberately.** One case lies outside the training range. By
construction a tree returns a finite constant for such an input, and the check confirms that
both implementations do so identically. The check does not imply that the resulting number is
meaningful, and it is not. The page therefore makes no claim about inputs outside the trained
range, because no honest claim can be made about them.

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

This is not a hypothetical risk: it is the specific failure that the design makes possible. It
is detected by check [2], which extracts the class selectors from the stylesheet and compares
them against every class used in the source.

**If maintaining a stored copy is inconvenient**, write plain CSS by hand for new elements
instead of adding utility classes. For a single-file teaching resource this is often clearer
in any case: the hand-written block at the top of `src/app.css` is the appropriate place for
readable, semantic styles, and it requires no regeneration step.

To legitimately regenerate after adding classes: temporarily restore the CDN `<script>` tag
and the `tailwind.config` block in the template, load the page, click every control so the
MutationObserver sees the new classes, re-read the generated `<style>`, paste it over block
2 of `src/app.css`, remove the CDN tag again, and run `verify_page.py`.

---

## 5. Adding a feature safely

The order of these steps is significant, because each step assumes that the previous one has
been completed correctly.

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

If the figures displayed on the page and the figures in this README ever disagree, the README
is out of date. The page reads its values from `model/model.json` at run time and cannot
become inconsistent with it.