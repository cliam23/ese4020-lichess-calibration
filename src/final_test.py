"""Score every model once on February 2015 (the test set).

    python3 src/final_test.py            # refuses if already run
    python3 src/final_test.py --force    # rerun (logged as a rerun)

Needs results/selection.json from select_models.py, so the choice of model
is fixed before February is read.  Refits each selected model on all of
January, predicts February, and reports log-loss and Brier with day-level
bootstrap percentile intervals (B = 2000, seed 4020), paired differences
against Elo and against the course baseline, and a calibration error.
Writes results/final_test.json, results/calibration_final.csv,
figures/key_figure.{png,pdf}, and appends to results/test_set_log.txt.
"""
import datetime
import json
import os
import sys

import numpy as np
import pandas as pd

import model as M
from baseline import ROOT, SEED, boot_counts, load, pct
from extended_cv import fit_gbm, fit_logit
from model import TCS

B = 2000
EDGES = np.arange(-600, 601, 50)


def per_day(feb, p, days):
    pos = {d: i for i, d in enumerate(days)}
    di = feb["day"].map(pos).values
    s = feb["score"].values.astype(float)
    pc = np.clip(p, 1e-12, 1 - 1e-12)
    ll = -(s * np.log(pc) + (1 - s) * np.log1p(-pc))
    br = (s - p) ** 2
    D = len(days)
    return (np.bincount(di, ll, D), np.bincount(di, br, D),
            np.bincount(di, minlength=D).astype(float))


def calib(feb, preds, days, counts):
    """Bin means by (tc, gap bin) with day-bootstrap SE of the observed mean."""
    t = feb[(feb.d >= EDGES[0]) & (feb.d < EDGES[-1])].copy()
    t["bin"] = np.digitize(t.d, EDGES) - 1
    pos = {d: i for i, d in enumerate(days)}
    t["di"] = t.day.map(pos)
    rows, abs_err = [], {k: [] for k in preds}
    for tc in TCS:
        for b in range(len(EDGES) - 1):
            g = t[(t.tc == tc) & (t.bin == b)]
            if len(g) < 200:
                continue
            n_d = np.bincount(g.di, minlength=len(days)).astype(float)
            s_d = np.bincount(g.di, g.score, minlength=len(days))
            boot = (counts @ s_d) / np.maximum(counts @ n_d, 1)
            r = dict(tc=tc, lo=EDGES[b], hi=EDGES[b + 1], n=len(g),
                     d_mid=g.d.mean(), obs=g.score.mean(), se=boot.std(ddof=1))
            for k, p in preds.items():
                r["p_" + k] = p[g.index].mean()
                abs_err[k].append((len(g), abs(r["obs"] - r["p_" + k])))
            rows.append(r)
    ece = {k: float(sum(n * e for n, e in v) / sum(n for n, _ in v))
           for k, v in abs_err.items()}
    return pd.DataFrame(rows), ece


def main(force):
    log = os.path.join(ROOT, "results", "test_set_log.txt")
    if os.path.exists(log) and not force:
        sys.exit("February has already been scored (results/test_set_log.txt). "
                 "Use --force to rerun; the rerun is logged.")
    sel = json.load(open(os.path.join(ROOT, "results", "selection.json")))
    jan, feb = load("2015-01"), load("2015-02")
    feb = feb.reset_index(drop=True)

    models = {
        "Elo formula": lambda g: M.elo_prob(g.d.values),
        "Mean-score floor": (lambda m: lambda g: np.full(len(g), m))(jan.score.mean()),
        "Logit: d (pooled)": fit_logit(dict(level=1, df=0, lam=0.0), jan)[0],
        "Logit: d by time control (baseline)":
            fit_logit(dict(level=2, df=0, lam=0.0), jan)[0],
    }
    L = sel["logit"]
    final_pred, des, w = fit_logit(dict(level=int(L["level"]), df=int(L["df"] or 0),
                                        lam=float(L["lam"])), jan)
    models["Selected logit: " + L["label"]] = final_pred
    G = sel["gbm"]
    models["Gradient boosting: " + G["label"]] = fit_gbm(
        dict(max_iter=int(G["max_iter"]), max_leaf_nodes=int(G["max_leaf_nodes"]),
             learning_rate=float(G["learning_rate"])), jan)[0]
    preds = {k: f(feb) for k, f in models.items()}

    days = np.sort(feb.day.unique())
    counts = boot_counts(days, np.random.default_rng(SEED), B)
    sums = {k: per_day(feb, p, days) for k, p in preds.items()}
    Nd = next(iter(sums.values()))[2]
    Nb = counts @ Nd
    base_key = "Logit: d by time control (baseline)"
    res = {}
    for k, (Ld, Bd, _) in sums.items():
        r = {"logloss": float(Ld.sum() / Nd.sum()), "logloss_ci": pct(counts @ Ld / Nb),
             "brier": float(Bd.sum() / Nd.sum()), "brier_ci": pct(counts @ Bd / Nb)}
        for ref, tag in (("Elo formula", "elo"), (base_key, "baseline")):
            dL = Ld - sums[ref][0]
            r[f"minus_{tag}"] = float(dL.sum() / Nd.sum())
            r[f"minus_{tag}_ci"] = pct(counts @ dL / Nb)
        res[k] = r

    short = {"elo": "Elo formula", "base": base_key,
             "final": "Selected logit: " + L["label"],
             "gbm": "Gradient boosting: " + G["label"]}
    cal, ece = calib(feb, {k: preds[v] for k, v in short.items()}, days, counts)
    cal.to_csv(os.path.join(ROOT, "results", "calibration_final.csv"), index=False)

    out = {"february_games": len(feb), "days": len(days), "B": B, "seed": SEED,
           "selection": sel, "scores": res,
           "mean_abs_calibration_error": {short_k: ece[short_k] for short_k in ece},
           "final_coefficients": dict(zip(des.names, map(float, w))),
           "feature_means": dict(zip(des.names, map(float, des.mu))),
           "feature_sds": dict(zip(des.names, map(float, des.sd)))}
    with open(os.path.join(ROOT, "results", "final_test.json"), "w") as f:
        json.dump(out, f, indent=2)
    with open(log, "a") as f:
        f.write(f"{datetime.datetime.now().isoformat(timespec='seconds')} "
                f"February scored{' (forced rerun)' if force else ''}; "
                f"selection: {L['label']} | {G['label']}\n")
    for k, r in res.items():
        print(f"{k:60s} {r['logloss']:.5f} [{r['logloss_ci'][0]:.5f}, "
              f"{r['logloss_ci'][1]:.5f}]  vs base {r['minus_baseline']:+.5f} "
              f"[{r['minus_baseline_ci'][0]:+.5f}, {r['minus_baseline_ci'][1]:+.5f}]")
    print("calibration error:", ece)
    figure(cal, short)


def figure(cal, short):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plotstyle import COLORS, INK, MUTED, setup, style_axes
    setup()
    fig, ax = plt.subplots(2, 3, figsize=(10, 5.6), sharex=True, sharey="row",
                           facecolor="#fcfcfb",
                           gridspec_kw={"height_ratios": [1.3, 1]})
    series = [("elo", "Elo formula", COLORS[0]),
              ("base", "Logistic baseline", COLORS[1]),
              ("final", "Selected model", COLORS[2])]
    xs = np.linspace(-600, 600, 300)
    for j, tc in enumerate(TCS):
        t = cal[cal.tc == tc]
        a, r = ax[0, j], ax[1, j]
        a.errorbar(t.d_mid, t.obs, yerr=1.96 * t.se, fmt="o", ms=3.5,
                   color=INK, lw=1, label="Observed (±1.96 day-bootstrap SE)",
                   zorder=4)
        a.plot(xs, M.elo_prob(xs), color=COLORS[0], lw=1.6, label="Elo formula")
        a.plot(t.d_mid, t.p_final, color=COLORS[2], lw=1.6, label="Selected model")
        r.fill_between(t.d_mid, -1.96 * t.se, 1.96 * t.se, color="#e1e0d9",
                       lw=0, label="±1.96 SE of observed")
        for off, (k, lab, col) in zip((-8, 0, 8), series):
            r.plot(t.d_mid + off, t.obs - t["p_" + k], "o", ms=3.2, color=col,
                   label=lab)
        r.axhline(0, color=MUTED, lw=1)
        a.set_title(f"{tc.capitalize()}  (n = {int(t.n.sum()):,})", loc="left",
                    fontsize=10, color=INK)
        r.set_xlabel("Rating gap d = WhiteElo − BlackElo (points)")
        for x in (a, r):
            style_axes(x)
    ax[0, 0].set_ylabel("Mean score for White")
    ax[1, 0].set_ylabel("Observed − predicted")
    h0, l0 = ax[0, 0].get_legend_handles_labels()
    h1, l1 = ax[1, 0].get_legend_handles_labels()
    keep = {"Observed (±1.96 day-bootstrap SE)": None}
    handles, labels = [], []
    for h, l in list(zip(h0, l0)) + list(zip(h1, l1)):
        if l not in labels and l != "Elo formula" or (l == "Elo formula" and l not in labels):
            if l in labels:
                continue
            handles.append(h)
            labels.append(l)
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(ROOT, "figures", f"key_figure.{ext}"), dpi=200,
                    facecolor=fig.get_facecolor())


def redraw():
    """Re-render the key figure from results/ without touching February."""
    sel = json.load(open(os.path.join(ROOT, "results", "selection.json")))
    cal = pd.read_csv(os.path.join(ROOT, "results", "calibration_final.csv"))
    figure(cal, None)


if __name__ == "__main__":
    if "--redraw" in sys.argv:
        redraw()
    else:
        main("--force" in sys.argv)
