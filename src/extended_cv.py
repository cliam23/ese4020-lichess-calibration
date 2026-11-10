"""5-fold day-blocked CV on January for the extended models (Section 4A).

    python3 src/extended_cv.py logit    # feature ladder levels 1-4, ridge grid
    python3 src/extended_cv.py gbm      # gradient boosting comparison

Same folds as baseline.py (contiguous blocks of January days).  Every
feature transform and every fitted parameter is learned on the training
folds only.  Writes results/cv_extended_<family>.csv with one row per
(config, fold): held-out log-loss, Brier and number of games.
February is not read here.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

import model as M
from baseline import K, ROOT, load
from features import Design

LOGIT_GRID = (
    [dict(level=1, df=0, lam=0.0), dict(level=2, df=0, lam=0.0)]
    + [dict(level=3, df=0, lam=l) for l in (0.0, 1e-5, 1e-4, 1e-3)]
    + [dict(level=4, df=k, lam=l) for k in (4, 6, 8, 10, 12)
       for l in (0.0, 1e-5, 1e-4, 1e-3)]
)
GBM_GRID = (
    [dict(max_iter=it, max_leaf_nodes=lv, learning_rate=0.1)
     for it in (50, 100, 150, 400) for lv in (7, 15, 63)]
    + [dict(max_iter=it, max_leaf_nodes=lv, learning_rate=0.1)
       for it in (25, 50) for lv in (127, 255)]
    + [dict(max_iter=it, max_leaf_nodes=lv, learning_rate=0.05)
       for it in (50, 100) for lv in (63, 127)]
)
SEED = 4020


def folds(jan):
    days = np.sort(jan["day"].unique())
    return np.array_split(days, K)


def fit_logit(cfg, tr):
    des = Design(cfg["level"], cfg["df"] or 5)
    X = des.fit_transform(tr)
    n = np.ones(len(tr))
    s = tr["score"].values.astype(float)
    pen = np.where(des.pen)[0].tolist()
    w = M.newton(X, n, s, cfg["lam"], pen)
    return lambda g: M.sigmoid(des.transform(g) @ w), des, w


def gbm_xy(g):
    """Draws become a win and a loss with weight 1/2 each."""
    X = np.column_stack([g["d"].values,
                         (g["white_elo"].values + g["black_elo"].values) / 2,
                         g["tc"].cat.codes.values])
    S = g["score"].values.astype(float)
    win, loss = S > 0, S < 1
    Xr = np.vstack([X[win], X[loss]])
    y = np.r_[np.ones(win.sum()), np.zeros(loss.sum())]
    w = np.r_[S[win], 1 - S[loss]]
    return X, Xr, y, w


def fit_gbm(cfg, tr):
    from sklearn.ensemble import HistGradientBoostingClassifier
    _, Xr, y, w = gbm_xy(tr)
    m = HistGradientBoostingClassifier(
        learning_rate=cfg["learning_rate"], max_iter=cfg["max_iter"],
        max_leaf_nodes=cfg["max_leaf_nodes"], min_samples_leaf=200,
        categorical_features=[2], monotonic_cst=[1, 0, 0],
        early_stopping=False, random_state=SEED)
    m.fit(Xr, y, sample_weight=w)
    return lambda g: m.predict_proba(gbm_xy(g)[0])[:, 1], m


def score(p, g):
    n = np.ones(len(g))
    s = g["score"].values.astype(float)
    return M.log_loss(p, n, s), M.brier(p, n, s, s ** 2)


def main(family):
    jan = load("2015-01")
    grid = LOGIT_GRID if family == "logit" else GBM_GRID
    fit = fit_logit if family == "logit" else fit_gbm
    rows = []
    for f, held in enumerate(folds(jan)):
        te = jan["day"].isin(held).values
        tr, ts = jan[~te], jan[te]
        for i, cfg in enumerate(grid):
            t0 = time.time()
            pred = fit(cfg, tr)[0]
            ll, br = score(pred(ts), ts)
            rows.append(dict(family=family, config=i, **cfg, fold=f,
                             games=len(ts), logloss=ll, brier=br))
            print(f"fold {f} cfg {i} {cfg} ll={ll:.5f} "
                  f"({time.time() - t0:.1f}s)", flush=True)
    out = os.path.join(ROOT, "results", f"cv_extended_{family}.csv")
    pd.DataFrame(rows).to_csv(out, index=False)


if __name__ == "__main__":
    main(sys.argv[1])
