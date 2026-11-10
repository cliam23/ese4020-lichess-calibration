"""Freeze the model choice from January CV, before February is read.

    python3 src/select_models.py

Reads results/cv_extended_logit.csv and results/cv_extended_gbm.csv.
For each config: the fold-size-weighted CV log-loss (Section 4A), its naive
SE, and the paired difference from the course baseline (level 2: separate
b0, b1 per time control) with the SE of the paired fold differences.
Picks the lowest-CV logistic model and the lowest-CV boosting model and
writes them to results/selection.json.  final_test.py refuses to run
without this file, so February cannot influence the choice.
"""
import json
import os

import numpy as np
import pandas as pd

from baseline import ROOT

NAMES = {1: "Logit: d (pooled)", 2: "Logit: d by time control (baseline)",
         3: "Logit: + mean rating", 4: "Logit: + spline in d"}


def summarise(cv, ref):
    rows = []
    for (fam, cfg), g in cv.groupby(["family", "config"]):
        g = g.sort_values("fold")
        ll = (g.logloss * g.games).sum() / g.games.sum()
        br = (g.brier * g.games).sum() / g.games.sum()
        diff = g.logloss.values - ref
        r = g.iloc[0].to_dict()
        r.update(cv_logloss=ll, cv_brier=br,
                 naive_se=g.logloss.std(ddof=1) / np.sqrt(len(g)),
                 minus_baseline=float(np.average(diff, weights=g.games)),
                 minus_baseline_se=float(diff.std(ddof=1) / np.sqrt(len(g))))
        for k in ("fold", "games", "logloss", "brier"):
            r.pop(k)
        rows.append(r)
    return pd.DataFrame(rows)


def label(r):
    if r["family"] == "gbm":
        return (f"Gradient boosting ({int(r['max_iter'])} trees, "
                f"{int(r['max_leaf_nodes'])} leaves, rate {r['learning_rate']:g})")
    lab = NAMES[int(r["level"])]
    if int(r["level"]) == 4:
        lab += f" (df {int(r['df'])})"
    return lab + (f", λ={r['lam']:g}" if r["lam"] else "")


def main():
    lg = pd.read_csv(os.path.join(ROOT, "results", "cv_extended_logit.csv"))
    gb = pd.read_csv(os.path.join(ROOT, "results", "cv_extended_gbm.csv"))
    cv = pd.concat([lg, gb], ignore_index=True)
    ref = lg[(lg.level == 2)].sort_values("fold").logloss.values
    s = summarise(cv, ref)
    s["label"] = s.apply(label, axis=1)
    s = s.sort_values(["family", "cv_logloss"])
    s.to_csv(os.path.join(ROOT, "results", "cv_extended_summary.csv"),
             index=False)
    best_logit = s[s.family == "logit"].iloc[0]
    best_gbm = s[s.family == "gbm"].iloc[0]
    keep = ["family", "config", "level", "df", "lam", "max_iter",
            "max_leaf_nodes", "learning_rate", "cv_logloss", "naive_se",
            "minus_baseline", "minus_baseline_se", "label"]
    sel = {"logit": {k: best_logit[k] for k in keep if k in best_logit},
           "gbm": {k: best_gbm[k] for k in keep if k in best_gbm}}
    with open(os.path.join(ROOT, "results", "selection.json"), "w") as f:
        json.dump(sel, f, indent=2, default=lambda x: None if pd.isna(x)
                  else (x.item() if hasattr(x, "item") else x))
    pd.set_option("display.width", 200)
    print(s[["label", "cv_logloss", "naive_se", "minus_baseline",
             "minus_baseline_se"]].to_string(index=False))
    print("\nSelected:", best_logit["label"], "|", best_gbm["label"])


if __name__ == "__main__":
    main()
