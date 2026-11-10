"""Does the slope or White's advantage hinge on a filtering choice?

    python3 src/sensitivity.py

January only.  Refits the two-parameter model on subsets and reports
b1 / Elo slope and b0 / b1 with day-bootstrap percentile intervals
(B = 500, seed 4020).  Writes results/sensitivity.csv.
"""
import os

import numpy as np
import pandas as pd

import model as M
from baseline import ROOT, SEED, boot_counts, load, pct
from model import Logit

B = 500


def subsets(j):
    either1500 = (j.white_elo == 1500) | (j.black_elo == 1500)
    return [
        ("All kept games", j),
        ("Drop 'Rules infraction' endings", j[j.termination != "Rules infraction"]),
        ("Drop time-forfeit endings", j[j.termination != "Time forfeit"]),
        ("Drop games with a rating of exactly 1500 (new-account default)",
         j[~either1500]),
        ("Only |d| < 400", j[j.d.abs() < 400]),
        ("Both players ≥ 30 games this month", j[(j.n_white >= 30) & (j.n_black >= 30)]),
        ("Days 1-15 only", j[j.day <= 15]),
        ("Days 16-31 only", j[j.day >= 16]),
    ]


def main():
    rng = np.random.default_rng(SEED)
    jan = load("2015-01")
    days = np.sort(jan["day"].unique())
    counts = boot_counts(days, rng, B)
    rows = []
    for name, g in subsets(jan):
        c = M.cells(g, keys=("day", "d"))
        c["tc"] = "all"
        # resample the days this subset actually covers
        sd = np.sort(g.day.unique())
        cnt = counts if len(sd) == len(days) else boot_counts(
            sd, np.random.default_rng(SEED), B)
        sp = {d: i for i, d in enumerate(sd)}
        di = c["day"].map(sp).values
        pt = Logit("pooled").fit(c)
        b0, b1 = pt.coefs()["all"]
        bb = []
        for m in cnt:
            w = m[di].astype(float)
            if w.sum() == 0:
                continue
            bb.append(Logit("pooled").fit(c, weight=w, init=pt.w).coefs()["all"])
        bb = np.array(bb)
        rows.append(dict(subset=name, games=len(g), share=len(g) / len(jan),
                         slope_ratio=b1 / M.ELO_SLOPE,
                         slope_lo=pct(bb[:, 1] / M.ELO_SLOPE)[0],
                         slope_hi=pct(bb[:, 1] / M.ELO_SLOPE)[1],
                         adv_pts=b0 / b1, adv_lo=pct(bb[:, 0] / bb[:, 1])[0],
                         adv_hi=pct(bb[:, 0] / bb[:, 1])[1]))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "results", "sensitivity.csv"),
                              index=False)


if __name__ == "__main__":
    main()
