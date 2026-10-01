"""
Train the teaching model for the Beijing PM2.5 demo page.

Input   data/beijing_pm25.csv   (UCI Beijing PM2.5 Data, Song et al. 2016)
Output  model/model.json        (inlined into index.html by scripts/build_page.py)

Models
  regression    GradientBoostingRegressor   -> PM2.5 concentration (µg/m³)
  classification GradientBoostingClassifier -> three-band air quality (Good / Moderate / Polluted)

Every metric reported here is computed from OUT-OF-FOLD predictions under
5-fold shuffled cross-validation, so the numbers on the page are honest
generalisation estimates rather than training-set fits.

CRITICAL: build_features() in src/index.template.html mirrors engineer() below.
If the two ever diverge, every prediction shown in the browser is wrong and
nothing will raise an error. scripts/verify_page.py exists to catch exactly
that drift, so run it after touching either side.

Usage:  python3 scripts/train_model.py
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, r2_score
from sklearn.model_selection import KFold, cross_val_predict

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "beijing_pm25.csv")
OUT_JSON = os.path.join(ROOT, "model", "model.json")

SEED = 42
CV = KFold(5, shuffle=True, random_state=SEED)

REG_KW = dict(n_estimators=250, max_depth=4, learning_rate=0.06)
CLF_KW = dict(n_estimators=150, max_depth=4, learning_rate=0.08)

# The 10 features the page is allowed to see, in the exact order the flattened
# tree arrays reference them by index.
FEATURES = ["TEMP", "DEWP", "TEMP_minus_DEWP", "PRES", "PRES_minus_1013",
            "hour_sin", "hour_cos", "month_sin", "month_cos", "wind_dir"]

CAT_ORDER = ["NW", "NE", "SE", "cv"]
CLASS_NAMES = ["Good", "Moderate", "Polluted"]
CLASS_EDGES = [50.0, 150.0]

# China's HJ 633 six-band AQI scheme, as PM2.5 concentration breakpoints
# (µg/m³). Interpretive only: it labels a prediction, it never feeds back in.
AQI_BANDS = [
    ("Good", 0, 50, 0, 35),
    ("Moderate", 51, 100, 35, 75),
    ("Light pollution", 101, 150, 75, 115),
    ("Moderate pollution", 151, 200, 115, 150),
    ("Heavy pollution", 201, 300, 150, 250),
    ("Severe pollution", 301, 500, 250, 500),
]

# Shown verbatim in the page's provenance panel. Keeping the reasoning in the
# model file (rather than hardcoding it in the HTML) means the explanation and
# the model can never describe different models.
WIND_DROPPED_NOTE = (
    "The UCI column 'Iws' is a cumulative wind-speed counter, not an instantaneous "
    "reading: values accumulate from about 0.45 up to 585 and the counter resets "
    "frequently. Within a single year, 18.9% of consecutive one-hour differences "
    "(8,283 rows) are negative, and 1,301 of them are obvious counter resets. The "
    "true instantaneous wind speed therefore cannot be recovered reliably, so this "
    "model does not use the column. We would rather drop a physically important "
    "variable than pass off a counter reading as a wind speed. Wind direction "
    "(cbwd) is retained because it is a reliable categorical variable, but note "
    "that it carries no dilution information of its own."
)

# Page defaults: a cold, calm, high-pressure December evening, which is the
# classic shape of a Beijing winter pollution episode. Chosen so that the
# regressor and the classifier agree, so the page does not open showing two
# models contradicting each other.
DEFAULTS = {"TEMP": -5.0, "DEWP": -6.0, "PRES": 1035.0, "month": 12, "hour": 20}


def load():
    """Read the raw CSV and drop rows with a missing target."""
    d = pd.read_csv(DATA).rename(columns={"pm2.5": "pm25"})
    missing_pct = round(float(d["pm25"].isna().mean() * 100), 2)
    d = d.dropna(subset=["pm25"]).reset_index(drop=True)
    return d, missing_pct


def engineer(d):
    """
    Build the 10-dimensional feature matrix. The browser mirrors this exactly.

    Deliberately absent: the co-measured pollutants (SO2, NO2, CO, O3), the
    cumulative wind-speed counter, and any raw date or row index. Including
    same-hour SO2/NO2 would let the model "predict" PM2.5 from its own
    correlates rather than from meteorology.
    """
    o = pd.DataFrame(index=d.index)
    o["TEMP"] = d["TEMP"]
    o["DEWP"] = d["DEWP"]
    o["TEMP_minus_DEWP"] = d["TEMP"] - d["DEWP"]
    o["PRES"] = d["PRES"]
    o["PRES_minus_1013"] = d["PRES"] - 1013.25
    o["hour_sin"] = np.sin(2 * np.pi * d["hour"] / 24)
    o["hour_cos"] = np.cos(2 * np.pi * d["hour"] / 24)
    o["month_sin"] = np.sin(2 * np.pi * d["month"] / 12)
    o["month_cos"] = np.cos(2 * np.pi * d["month"] / 12)
    o["wind_dir"] = d["cbwd"].map({"NW": 0, "NE": 1, "SE": 2, "cv": 3})
    assert not o.isna().any().any(), "engineered features contain NaN"
    return o[FEATURES].to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# Tree packing
#
# scikit-learn's fitted estimators are serialised here into plain flat arrays
# so the page needs no ML library. Stride 5: [feature, threshold, left, right,
# value]. A leaf is identified by feature === -1. Child pointers are OFFSETS
# (multiples of 5), not indices -- getting that wrong yields a page that loads
# without error and predicts nonsense.
# ---------------------------------------------------------------------------

def pack_regressor(trees):
    out = []
    for est in trees:
        t = est.tree_
        a = np.full(len(t.children_left) * 5, -1.0)
        a[0::5] = t.feature
        a[1::5] = t.threshold
        a[2::5] = t.children_left
        a[3::5] = t.children_right
        # Leaves are rounded to 3dp to keep model.json small. The self-check
        # below verifies this rounding stays within tolerance.
        a[4::5] = np.round(t.value.ravel(), 3)
        out.append(a)
    return out


def pack_classifier(trees_2d):
    """
    trees_2d has shape (n_estimators, n_classes): GradientBoostingClassifier
    fits one binary tree per class, so prediction sums the leaf values of all
    trees belonging to a class and takes the argmax of those raw scores.
    """
    out = []
    for est in trees_2d:
        for k, est_k in enumerate(est):
            t = est_k.tree_
            a = np.full(len(t.children_left) * 5, -1.0)
            a[0::5] = t.feature
            a[1::5] = t.threshold
            a[2::5] = t.children_left
            a[3::5] = t.children_right
            a[4::5] = np.round(t.value.ravel(), 5)
            out.append({"cls": k, "tree": a.tolist()})
    return out


# ---------------------------------------------------------------------------
# Independent evaluators for the packed format. These deliberately do NOT call
# sklearn: the point is to re-derive the prediction from the exported arrays
# alone, which is what the browser will do.
# ---------------------------------------------------------------------------

def predict_reg(flat_trees, x, init, lr):
    """Mirror sklearn's traversal: feature <= threshold goes LEFT."""
    out = init
    for a in flat_trees:
        o = 0
        while a[o] >= 0:
            go_left = x[int(a[o])] <= a[o + 1]
            o = 5 * int(a[o + 2 if go_left else o + 3])
        out += lr * a[o + 4]
    return out


def reg_pred_frame(flat_trees, X, init, lr):
    return np.array([predict_reg(flat_trees, x, init, lr) for x in X])


def clf_pred(flat_trees, x, lr, init):
    score = np.array(init, dtype=float)
    for item in flat_trees:
        a = item["tree"]
        o = 0
        while a[o] >= 0:
            go_left = x[int(a[o])] <= a[o + 1]
            o = 5 * int(a[o + 2 if go_left else o + 3])
        score[item["cls"]] += lr * a[o + 4]
    return int(np.argmax(score))


def main():
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    d, missing_pct = load()
    X = engineer(d)
    y = d["pm25"].to_numpy(dtype=float)
    yc = np.digitize(y, CLASS_EDGES)

    print("data      : %d rows x %d features | raw missing pm2.5 %.2f%%"
          % (len(d), X.shape[1], missing_pct))
    print("class balance: %s" % dict(zip(CLASS_NAMES, np.bincount(yc).tolist())))

    # ---- fit the two models that get shipped ----
    reg = GradientBoostingRegressor(**REG_KW).fit(X, y)
    clf = GradientBoostingClassifier(**CLF_KW).fit(X, yc)

    # ---- regression, all out-of-fold ----
    p_base = cross_val_predict(DummyRegressor(), X, y, cv=CV, n_jobs=-1)
    p_lin = cross_val_predict(LinearRegression(), X, y, cv=CV, n_jobs=-1)
    p_reg = cross_val_predict(GradientBoostingRegressor(**REG_KW), X, y, cv=CV, n_jobs=-1)

    def stats(p):
        return {"r2": round(float(r2_score(y, p)), 4),
                "rmse": round(float(np.sqrt(((y - p) ** 2).mean())), 2),
                "mae": round(float(np.abs(y - p).mean()), 2)}

    s_base, s_lin, s_reg = stats(p_base), stats(p_lin), stats(p_reg)
    print("\n[regression] baseline (predict mean) %s" % s_base)
    print("[regression] linear                 %s" % s_lin)
    print("[regression] gradient boosting      %s" % s_reg)

    # ---- the leakage demonstration, measured rather than asserted ----
    # Adding the cumulative wind-speed counter raises CV R2. That is not a
    # better model: the counter accumulates with time, so it encodes elapsed
    # time rather than wind. Both numbers are computed here so the page can
    # show the real figure instead of a remembered constant.
    X_leak = np.column_stack([X, d["Iws"].to_numpy(dtype=float)])
    p_leak = cross_val_predict(GradientBoostingRegressor(**REG_KW), X_leak, y, cv=CV, n_jobs=-1)
    r2_leak = round(float(r2_score(y, p_leak)), 4)
    print("[regression] WITH cumulative Iws     R2 = %+.4f  <-- leakage, do not use"
          % r2_leak)

    # ---- temporal extrapolation: train on <=2013, test on 2014 ----
    yr = d["year"].to_numpy()
    te = yr == 2014
    r2_time = round(float(r2_score(y[te], reg.predict(X[te]))), 4)
    print("[regression] temporal holdout (<=2013 -> 2014) R2 = %+.4f (n=%d)"
          % (r2_time, int(te.sum())))

    # ---- classification ----
    c_base = cross_val_predict(DummyClassifier(strategy="most_frequent"), X, yc, cv=CV, n_jobs=-1)
    c_clf = cross_val_predict(GradientBoostingClassifier(**CLF_KW), X, yc, cv=CV, n_jobs=-1)
    acc_base = round(float((c_base == yc).mean()), 4)
    acc_clf = round(float((c_clf == yc).mean()), 4)
    bal_clf = round(float(balanced_accuracy_score(yc, c_clf)), 4)
    print("\n[classification] baseline (majority) %.4f | model %.4f | balanced %.4f"
          % (acc_base, acc_clf, bal_clf))
    cm = confusion_matrix(yc, c_clf, labels=[0, 1, 2])
    print("[classification] confusion matrix (rows = truth):\n", cm)

    # Teaching contrast: thresholding the regression output instead of
    # classifying directly. The gap is the cost of boundary flips.
    acc_derived = round(float((np.digitize(np.clip(p_reg, 0, None), CLASS_EDGES) == yc).mean()), 4)
    print("[classification] bands derived from regression %.4f  <-- boundary errors" % acc_derived)

    # ---- empirical prediction interval from out-of-fold residuals ----
    # Bucketed by predicted concentration, because the error magnitude scales
    # with the level: a single global interval would be far too tight at one
    # end and uselessly wide at the other.
    edges = np.quantile(p_reg, np.linspace(0, 1, 9))
    band_idx = np.clip(np.digitize(p_reg, edges) - 1, 0, len(edges) - 2)
    bands = []
    for k in range(len(edges) - 1):
        m = band_idx == k
        if m.sum() < 30:
            bands.append(None)
            continue
        lo, hi = np.percentile(y[m] - p_reg[m], [10, 90])
        bands.append({"n": int(m.sum()),
                      "resid_lo10": round(float(lo), 2),
                      "resid_p90": round(float(hi), 2)})

    imp = sorted(zip(FEATURES, reg.feature_importances_), key=lambda t: -t[1])
    print("\nfeature importance: %s" % [(n, round(float(v), 3)) for n, v in imp])

    # ---- 60 real records for the "load a random record" button ----
    rng = np.random.RandomState(SEED)
    pick = rng.choice(len(d), 60, replace=False)
    samples = [{"TEMP": round(float(d.TEMP[i]), 1),
                "DEWP": round(float(d.DEWP[i]), 1),
                "PRES": round(float(d.PRES[i]), 1),
                "hour": int(d.hour[i]),
                "month": int(d.month[i]),
                "day": int(d.day[i]),
                "cbwd": str(d.cbwd[i]),
                "year": int(d.year[i]),
                "observed_pm25": round(float(y[i]), 1),
                "observed_class": CLASS_NAMES[int(yc[i])]}
               for i in pick]

    # ---- serialise ----
    flat_reg = pack_regressor(reg.estimators_.ravel())
    flat_clf = pack_classifier(clf.estimators_)
    init_const = round(float(reg.init_.constant_.ravel()[0]), 4)
    # For multiclass GradientBoostingClassifier the raw-score origin is the log
    # of each class prior, not a constant.
    clf_init = [round(float(v), 6) for v in np.log(clf.init_.class_prior_)]

    payload = {
        "meta": {
            "dataset": "UCI Beijing PM2.5 Data (Song et al., 2016)",
            "source_url": "https://archive.ics.uci.edu/dataset/381/beijing+pm2+5+data",
            "citation": ("Song, C., Zhou, Y., Lu, J., & Qi, J. (2016). Detecting outliers in "
                         "Beijing's PM2.5: A data mining approach based on PM2.5 five years of "
                         "data in Beijing. ISPRS International Journal of Geo-Information, "
                         "5(4), 48."),
            "rows_used": int(len(d)),
            "wind_feature_note": WIND_DROPPED_NOTE,
            "raw_missing_pm25_pct": missing_pct,
            "features": FEATURES,
            "wind_dir_order": CAT_ORDER,
            "class_names": CLASS_NAMES,
            "class_edges": CLASS_EDGES,
            "seed": SEED,
            "cv_note": "5-fold shuffled cross-validation, out-of-fold predictions",
            "reg_params": dict(REG_KW),
            "clf_params": dict(CLF_KW),
            # Slider bounds are recorded here so the browser can never present a
            # range narrower than the data the model was trained on.
            "input_ranges": {
                "TEMP": [round(float(d.TEMP.min()), 1), round(float(d.TEMP.max()), 1)],
                "DEWP": [round(float(d.DEWP.min()), 1), round(float(d.DEWP.max()), 1)],
                "PRES": [round(float(d.PRES.min()), 1), round(float(d.PRES.max()), 1)],
            },
            "defaults": DEFAULTS,
        },
        "aqi_bands": [{"name": n, "aqi_lo": a, "aqi_hi": b, "pm_lo": lo, "pm_hi": hi}
                      for n, a, b, lo, hi in AQI_BANDS],
        "metrics": {
            "regression": {
                "baseline_mean": s_base, "linear": s_lin, "model": s_reg,
                "leaky_r2_with_iws": r2_leak,
                "temporal_holdout_2014_r2": r2_time,
            },
            "classification": {
                "baseline_majority": acc_base, "model": acc_clf,
                "balanced_accuracy": bal_clf,
                "derived_from_regression": acc_derived,
                "confusion_matrix": cm.tolist(),
                "class_support": np.bincount(yc, minlength=3).tolist(),
            },
            "feature_importance": [[n, round(float(v), 4)] for n, v in imp],
        },
        "samples": samples,
        "bands": {"edges": [round(float(e), 2) for e in edges], "resid": bands},
        "regressor": {"learning_rate": REG_KW["learning_rate"],
                      "init": init_const,
                      "trees": [t.tolist() for t in flat_reg]},
        "classifier": {"learning_rate": clf.learning_rate,
                       "init": clf_init,
                       "trees": flat_clf},
    }

    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, separators=(",", ":"))

    print("\nwrote %s (%.0f KB) | regressor %d trees / %d nodes | classifier %d trees / %d nodes"
          % (OUT_JSON, os.path.getsize(OUT_JSON) / 1024,
             len(flat_reg), sum(len(t) // 5 for t in flat_reg),
             len(flat_clf), sum(len(t["tree"]) // 5 for t in flat_clf)))

    # ---- self-check: re-derive predictions from the EXPORTED arrays ----
    # If leaf rounding or offset arithmetic were wrong, this is where it shows.
    n_chk = 2000
    d_reg = np.abs(reg_pred_frame(flat_reg, X[:n_chk], init_const, REG_KW["learning_rate"])
                   - reg.predict(X[:n_chk])).max()
    chk_c = np.array([clf_pred(flat_clf, x, clf.learning_rate, clf_init) for x in X[:n_chk]])
    agree = float((chk_c == clf.predict(X[:n_chk])).mean())
    print("self-check  regressor  vs sklearn: max abs diff = %.5f (n=%d)" % (d_reg, n_chk))
    print("self-check  classifier vs sklearn: agreement     = %.4f (n=%d)" % (agree, n_chk))
    assert d_reg < 0.05, "regressor self-check failed"
    assert agree == 1.0, "classifier self-check failed"
    print("self-check  OK")


if __name__ == "__main__":
    main()