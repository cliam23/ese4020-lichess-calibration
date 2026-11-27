"""5-fold cross-validation inside January 2015, in contiguous blocks of days.

Only January is ever opened here, so no choice can depend on a test month.
Writes results/cv_<part>.json; freeze.py then picks the settings and freezes them.

Usage: python -I src/tune.py linear|boost data/headers_2015-01.csv results/
"""

import itertools
import json
import sys
import time

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from lib import *  # noqa: E402,F401,F403

LAM_GRID = [0.0, 1e-5, 1e-4, 1e-3]
KNOT_GRID = [4, 6, 8, 10, 12]
BOOST_GRID = list(itertools.product([50, 100, 200], [31, 63, 127], [0.03, 0.05, 0.1, 0.2]))
N_BOOST = 20


def fold_losses(df, fold, fit_predict):
    """Mean validation log-loss of each fold, plus fold sizes."""
    out, sizes = [], []
    for k in range(len(FOLDS)):
        tr, va = df[fold != k], df[fold == k]
        p = fit_predict(tr, va)
        out.append(float(game_loss(va.S.values, p).mean()))
        sizes.append(int(len(va)))
    return out, sizes


def logit_fp(kind, lam=0.0, K=None):
    def fp(tr, va):
        knots = spline_knots(tr, K) if kind == "spline" else None
        m = fit_logit(design(kind, tr, knots), tr.S.values, lam=lam)
        return predict_logit(m, design(kind, va, knots))
    return fp


def main(part, path, out_dir):
    df, steps = load_month(path, "2015-01")
    fold = day_fold_ids(df.day.values)
    res = {"part": part, "n": len(df), "configs": {}}

    def run(name, fp, **meta):
        t = time.time()
        losses, sizes = fold_losses(df, fold, fp)
        res["configs"][name] = {"fold_loss": losses, "fold_size": sizes, **meta,
                                "cv": float(np.average(losses, weights=sizes))}
        print(f"{name:40s} cv={res['configs'][name]['cv']:.6f}  {time.time() - t:5.1f}s", flush=True)

    if part == "linear":
        run("floor", lambda tr, va: np.full(len(va), tr.S.mean()), kind="floor")
        run("elo", lambda tr, va: elo_p(va.d.values), kind="elo")
        run("pooled", logit_fp("pooled"), kind="pooled")
        run("bytc", logit_fp("bytc"), kind="bytc")
        for lam in LAM_GRID:
            run(f"rating|lam={lam:g}", logit_fp("rating", lam), kind="rating", lam=lam)
        for K, lam in itertools.product(KNOT_GRID, LAM_GRID):
            run(f"spline|K={K}|lam={lam:g}", logit_fp("spline", lam, K), kind="spline", K=K, lam=lam)
    else:
        rng = np.random.default_rng(SEED)
        pick = rng.choice(len(BOOST_GRID), size=N_BOOST, replace=False)
        for i in sorted(pick):
            trees, leaves, lr = BOOST_GRID[i]
            fp = lambda tr, va, a=(trees, leaves, lr): predict_hgb(fit_hgb(tr, *a), va)
            run(f"boost|trees={trees}|leaves={leaves}|lr={lr:g}", fp, kind="boost",
                trees=trees, leaves=leaves, lr=lr)
    json.dump(res, open(f"{out_dir}/cv_{part}.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
