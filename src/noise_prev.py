"""Second rating-noise test: smooth the gap with each player's past ratings.

    python3 src/noise_prev.py

January only.  For a game on day t, a player's earlier-day mean rating is
the average of his displayed ratings over all his games on days 1..t-1.
It uses only the past, so it cannot contain the game's own result, and it
averages out short-term rating fluctuation (it also lags real improvement).
On games where both players have at least k earlier-day games we fit the
two-parameter model with
  (a) the displayed gap d             (what Elo uses)
  (b) the smoothed gap d_prev
  (c) both, b1 d + b2 d_prev
If displayed ratings carry transient noise, (b) has the larger slope and (c)
puts weight on d_prev.  Day-bootstrap intervals, B = 500, seed 4020.
Writes results/noise_prev.csv.
"""
import os

import numpy as np
import pandas as pd

import model as M
from baseline import ROOT, SEED, boot_counts, load, pct

B = 500
MIN_PREV = [20, 100]


def fit_xy(X, s, w=None, init=None):
    n = np.ones(len(s)) if w is None else w
    return M.newton(X, n, s * n if w is not None else s, init=init)


def main():
    jan = load("2015-01")
    rows = []
    for k in MIN_PREV:
        g = jan[(jan.white_prev_n >= k) & (jan.black_prev_n >= k)].copy()
        g["d_prev"] = g.white_prev_mean - g.black_prev_mean
        s = g.score.values.astype(float)
        sc = 100.0
        designs = {
            "displayed gap d": np.column_stack([np.ones(len(g)), g.d / sc]),
            "earlier-day gap d_prev": np.column_stack([np.ones(len(g)), g.d_prev / sc]),
            "both": np.column_stack([np.ones(len(g)), g.d / sc, g.d_prev / sc]),
        }
        days = np.sort(g.day.unique())
        counts = boot_counts(days, np.random.default_rng(SEED), B)
        di = g.day.map({d: i for i, d in enumerate(days)}).values
        for name, X in designs.items():
            w0 = fit_xy(X, s)
            bb = np.array([fit_xy(X, s, m[di].astype(float), w0) for m in counts])
            r = dict(min_prev=k, games=len(g), share=len(g) / len(jan),
                     design=name, sd_d=float(g.d.std()),
                     sd_d_prev=float(g.d_prev.std()),
                     corr=float(np.corrcoef(g.d, g.d_prev)[0, 1]))
            for j, lab in enumerate(["b_d", "b_dprev"][: X.shape[1] - 1]):
                if name == "earlier-day gap d_prev":
                    lab = "b_dprev"
                ratio = w0[1 + j] / sc / M.ELO_SLOPE
                r[lab + "_over_elo"] = ratio
                r[lab + "_ci"] = pct(bb[:, 1 + j] / sc / M.ELO_SLOPE)
            r["sum_over_elo"] = w0[1:].sum() / sc / M.ELO_SLOPE
            r["sum_ci"] = pct(bb[:, 1:].sum(1) / sc / M.ELO_SLOPE)
            rows.append(r)
            print(r, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "results", "noise_prev.csv"),
                              index=False)


if __name__ == "__main__":
    main()
