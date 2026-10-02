# Beijing PM2.5 Prediction Demo (Environmental ML Teaching Demo)

**Live demo (GitHub Pages): https://drhycheung.github.io/EnvML/**

![Beijing PM2.5 Prediction Demo screenshot](docs/screenshot.png)

A single-file, front-end-only interactive demo that predicts the **PM2.5 concentration** and a
three-band air-quality class from meteorological inputs. The prediction is produced by
gradient-boosted trees that run entirely in the browser. It is intended for classroom
demonstration in **Environmental Informatics**, **Environmental ML** and **Smart City**
courses.

File: `index.html` — no build step, no backend, no API key, and **no network requests at
all**. Double-click it from disk and it works, offline. Place the same file in a GitHub
Pages repository and it is deployed.

---

## 1. Where this project fits: Monitor, Analyse, Control

Environmental informatics is usually taught as three activities: **monitor**, **analyse** and
**control**.

| Stage | What it means | Which project |
|---|---|---|
This project is one of two teaching examples. The companion project,
[EnvInfo](https://github.com/drhycheung/EnvInfo) is a browser-based air-quality dashboard for
Hong Kong that displays live readings from EPD stations (monitoring with on-the-fly
visualisation). EnvML applies the same three ideas to a different dataset: it uses historical
weather and PM2.5 measurements (monitor), builds a statistical model to relate weather to
PM2.5 (analyse), and produces hour-ahead predictions with policy thresholds to support
decision-making (control).

**A monitoring dashboard alone is not enough.** A dashboard shows what has already happened.
It cannot answer *"will tomorrow evening exceed 150 µg/m³?"*, so it cannot directly support a
decision about the future. Analysis and prediction are needed to move from description to
action.

The two projects are designed to be used together: EnvInfo supplies the monitoring stage, and
this project supplies the analysis and control stages.

---

## 2. From data to a decision

| Question | Answered by | Not answered by |
|---|---|---|
| "What has the pollution been?" | A dashboard, a chart or a time series | — |
| "Will it exceed 150 µg/m³ tomorrow evening?" | — | A dashboard |
| "Should an advisory be issued?" | **A prediction, shown against the threshold that triggers it** | — |
| "How much should the estimate be trusted?" | Baselines, an uncertainty range, and a stated limitation | — |

For this reason the prediction occupies the main position on the page and is displayed in the
largest type. The air-quality scale beneath it shows where the estimate falls in relation to
the 50 and 150 µg/m³ thresholds, because those thresholds are the decision that the user is
trying to make. The charts appear after the prediction and are intended as supporting
information.

The user must be able to enter the conditions expected for a future hour and obtain an
estimate for that hour. A tool that can only display historical records is a dashboard,
regardless of how the records are labelled, and it would not answer the question the project
exists to answer.

---

## 3. What the page does

| Feature | Implementation |
|---|---|
| **Estimates PM2.5 from five sliders and a wind-direction control** | The main output of the page. Gradient-boosted trees serialised to flat arrays; the prediction requires about 15 lines of vanilla JavaScript |
| **Reports an actionable class against a threshold** | A second ensemble of 450 trees gives Good / Moderate / Polluted, with a marker on the scale at 50 and 150 µg/m³ |
| **Accepts any future hour** | The sliders accept any month, hour and conditions, so "tomorrow evening" is a question the page can answer |
| Reports a three-band air-quality class | Shown alongside the concentration |
| No machine-learning library in the browser | scikit-learn's fitted trees are exported as `[feature, threshold, left, right, value]` at stride 5 |
| Honest evaluation panel | Out-of-fold cross-validation metrics, shown beside a predict-the-mean baseline and a linear-regression baseline |
| Uncertainty range | The 10th and 90th percentiles of the out-of-fold errors, grouped by predicted level |
| Leakage demonstration | The page states that including the wind-speed feature raises R² from 0.533 to 0.555, and explains why this is undesirable |
| Task-design comparison | Accuracy when the class is predicted directly, compared with accuracy when the class is derived from the predicted concentration |
| Feature importance | A table with bars, and a plain-language explanation of why dew-point depression is the most important feature |
| "Load a random real record" | 60 held-out records; shows the prediction beside the measured value and the error |
| Slider limits taken from the data | Every slider range is read from the model file, so it can never be narrower than the data used for training |
| Zero external dependencies | All CSS is stored inside the file; no CDN, no web fonts, no `fetch()` |

> **New to machine learning?** Section 1 of the [model notes](docs/model-notes.md#1-the-algorithms-in-plain-language)
> explains baselines, linear regression, decision trees, gradient boosting and
> cross-validation from first principles, and works through one real prediction step by step.

---

## 4. Data, sample size, and the one feature that had to go

Source: [UCI Beijing PM2.5 Data](https://archive.ics.uci.edu/dataset/381/beijing+pm2+5+data)
(Song et al., 2016).

### Sample size

The sample size is stated in full on the page, and the figures are as follows.

| Stage | Records | Explanation |
|---|---|---|
| Source dataset | **43,824** | Hourly monitoring records from Beijing, 1 January 2010 to 31 December 2014 |
| Removed | **2,067** (4.72%) | Records with no PM2.5 measurement |
| Used for modelling | **41,757** | The remaining complete records |

All performance figures are **out-of-fold**, calculated under 5-fold cross-validation. The
41,757 records are divided into 5 folds of approximately **8,351** records each. Each figure
is calculated on the 8,351 records of one fold, using a model fitted on the other
**33,406** records. No record is therefore used to score a model that was fitted on it.

This distinction matters when the figures are quoted. The page does not display a model
fitted on all 41,757 records; the 41,757 records are the total available, and any single
figure on the page comes from a model fitted on approximately 80% of them.

<details>
<summary><strong>Downloading the data</strong> (a copy is already committed at <code>data/beijing_pm25.csv</code>)</summary>

```bash
mkdir -p data && curl -L -o /tmp/pm25.zip \
  "https://archive.ics.uci.edu/static/public/381/beijing+pm2+5+data.zip"
unzip -o /tmp/pm25.zip -d data          # yields PRSA_data_2010.1.1-2014.12.31.csv
mv data/PRSA_data_2010.1.1-2014.12.31.csv data/beijing_pm25.csv
```

Verify the file before using it:

```bash
wc -l data/beijing_pm25.csv     # must be 43,825  (43,824 records + 1 header line)
head -1 data/beijing_pm25.csv   # No,year,month,day,hour,pm2.5,DEWP,TEMP,PRES,cbwd,Iws,Is,Ir
```

Two points to note. First, the older address
`.../ml/machine-learning-databases/00381/BeijingPM2.5.data` now returns **404**, because the
UCI archive has been reorganised; use the `static/public` address above. Second, the column
name in the file is `pm2.5`, and the full stop in that name causes problems with some
libraries; rename it to `pm25` when loading the file. The committed copy is byte-for-byte
identical to a fresh download.

</details>

### Features

Ten features are used: air temperature, dew point, dew-point depression, pressure, pressure
anomaly, cyclical sine and cosine of hour, cyclical sine and cosine of month, and wind
direction as an ordinal code.

### Wind speed is missing, and this is the most important limitation on the page

The UCI column `Iws` is a *cumulative counter*, not a measurement taken at that hour. It
climbs from about 0.45 to 585. Within a single year, 18.9% of its consecutive one-hour
differences are negative, and 1,301 of those are clear counter resets. Differencing the
column produces physically impossible values, including −489 m/s.

Using the raw counter as a feature raises cross-validated R² from 0.533 to **0.555**. This
apparent improvement comes from time accumulating inside a counter, and not from atmospheric
physics. A score obtained in this way is misleading, so the feature is excluded and the page
states this openly. Both figures are recalculated on every training run. The full
investigation is in the [dataset card](docs/dataset.md).

### Relationship to official sources

The Beijing Municipal Ecological Environment Monitoring Centre publishes authoritative hourly
PM2.5 monitoring, and national centres publish official forecasts. This project does not
replace either of them, and its figures should not be presented as an official forecast. The
page states the same limitation. The teaching value lies in showing that the step from raw
monitoring data to a usable prediction is short, inspectable, and reproducible in a small
number of lines of Python.

---

## 5. Results

All figures are out-of-fold, from 5-fold cross-validation with a fixed random seed.

**Regression — PM2.5 concentration (µg/m³)**

| Method | R² | RMSE | MAE |
|---|---|---|---|
| Always predict the training mean | −0.000 | 92.05 | 68.83 |
| Multiple linear regression | 0.376 | 72.71 | 52.84 |
| **Gradient-boosted trees (used by the page)** | **0.533** | 62.90 | 42.93 |

Allowing non-linear relationships raises R² by 0.157 compared with linear regression. The
remaining 0.467 of the variance is not a fault of the model. Weather conditions do not
determine air pollution; emissions and the transport of pollution from surrounding regions do.

**Classification — three-band air quality** (Good < 50 · Moderate 50–150 · Polluted ≥ 150 µg/m³)

| Method | Accuracy | Balanced accuracy |
|---|---|---|
| Always predict the most common class | 40.66% | 50.0% |
| **Gradient-boosted classifier (used by the page)** | **69.70%** | 67.22% |
| Bands derived from the predicted concentration | 66.91% | — |

The last row is included to illustrate a property of the task rather than to rank the models.
The regression has an RMSE of about 63 µg/m³, so many predictions fall within one band width
of a threshold. When the predicted concentration is close to 50 or 150 µg/m³, a small error
changes the resulting class. Predicting a concentration and then assigning a class is
therefore a different task from predicting the class directly, and the two tasks do not
produce the same accuracy.

Two further checks support these figures. Training on 2010–2013 and testing on 2014 alone
gives R² = 0.556, which is close to the cross-validated figure; the relationships are
therefore stable over time rather than an artefact of the data division. The full confusion
matrix is printed on the page.

---

## 6. How to run

- **Students, teachers, and anyone else**: double-click `index.html`. Nothing else is
  required — no server, no network, no installation. This is the intended method of use.
- **With a local server** (only needed if you wish to edit the file):
  `python3 -m http.server 8000`, then open `http://localhost:8000/index.html`. Functionally
  identical.
- **GitHub Pages, for your own deployment**: place `index.html` in *your* repository, then
  enable Pages through **Settings → Pages → Deploy from a branch**, selecting the branch and
  `/ (root)`. Your deployment will be at `https://<your-username>.github.io/<repo-name>/`.
  The link at the top of this README is the author's own deployment.

### Rebuilding from source

`index.html` is generated. To change it, edit the source files, not the output.

```bash
python3 scripts/train_model.py    # retrain -> model/model.json   (~60 s)
python3 scripts/build_page.py     # inline model + CSS -> index.html
python3 scripts/verify_page.py    # confirm the page agrees with the Python implementation
```

Training is reproducible: two runs of `train_model.py` produce byte-for-byte identical
output. This requires the estimators to be given a fixed random seed, for a reason explained
in section 1.9 of the [model notes](docs/model-notes.md#19-reproducibility-two-perfectly-duplicated-features).

`verify_page.py` runs 26 checks and exits with a non-zero status if any of them fails. The
[model notes](docs/model-notes.md) explain what each check covers and why it exists.

---

## 7. Known limitations

Several of the following are consequences of the constraints described in section 4, and
would require a different study design to resolve.

| Limitation | Consequence |
|---|---|
| Trained **only** on Beijing, 2010–2014 | Applying the model to Hong Kong or to any other location would reduce accuracy substantially. It is a demonstration of method, not a forecast |
| **No wind speed**, which is the most directly relevant available predictor | This places a ceiling on the accuracy that can be achieved; section 4 explains why |
| Weather inputs only; no emissions inventory | The unexplained variance is structural and cannot be removed by a better algorithm |
| One city and one period | No claim of generalisation to other cities, seasons or instruments |
| Class bands are fixed thresholds at 50 and 150 µg/m³ | Adjacent bands have no sharp physical boundary, which is why the confusion matrix contains many near-misses |
| The uncertainty range is empirical rather than probabilistic | It reports where the measured value usually fell for similar conditions, and not a calibrated 80% credible interval |
| Leaf values are rounded when exported | The export self-check absorbs this (maximum deviation 0.00073 µg/m³), but the browser is not bit-for-bit identical to a live scikit-learn model |
| `src/app.css` is a stored snapshot | Adding a new utility class without regenerating the file produces an element with no styling; `verify_page.py` fails the build in order to prevent this |
| Designed for desktop screens | Tested at 390 px and 1280 px with no horizontal overflow, but it is not a complete mobile design |

---

## 8. Documentation

| Document | What it covers |
|---|---|
| **[Vibe-coding guide](docs/vibe-coding.md)** | The design-thinking rationale (why data without prediction cannot lead to an action), how the page was built with an AI coding tool, three faults that measurement detected and visual inspection did not, further work for students, and the complete prompt needed to reproduce the page, including how to download the dataset |
| **[Model notes](docs/model-notes.md)** | **The algorithms explained from first principles** (baseline, linear regression, decision trees, gradient boosting, cross-validation and data leakage), followed by the tree serialisation format, the verification harness, the stored-stylesheet trade-off, and how to add a feature safely |
| **[Dataset card](docs/dataset.md)** | Provenance, every column and the reason it is or is not used, and the complete `Iws` data-quality investigation |

---

## 9. Licences and attribution

- **Code**: MIT — see [LICENSE](LICENSE), © 2026 drhycheung.
- **Data**: [Beijing PM2.5 Data](https://archive.ics.uci.edu/dataset/381/beijing+pm2+5+data)
  (Song et al., 2016), CC BY 4.0, courtesy of the UCI Machine Learning Repository. Copyright
  remains with UCI and the original authors.

The two are separately licensed. The MIT licence covers this repository's code only and does
not extend to the dataset.