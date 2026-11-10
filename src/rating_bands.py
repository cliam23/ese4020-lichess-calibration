"""White's advantage and the Elo slope by rating level and time control.

    python3 src/rating_bands.py

January only.  For each band of mean rating rbar = (WhiteElo + BlackElo)/2
and each time control, fit the two-parameter model by maximum likelihood
and put day-bootstrap percentile intervals (B = 1000, seed 4020) on
  * b1 / (ln 10 / 400)   the slope relative to Elo
  * b0 / b1              White's advantage in rating points
  * sigma(b0) - 1/2      White's advantage in expected score at d = 0
Writes results/rating_bands.csv and figures/rating_bands.png.
"""
import os

import numpy as np
import pandas as pd

import model as M
from baseline import ROOT, SEED, boot_counts, load, pct
from model import TCS, Logit

B = 1000
EDGES = [0, 1300, 1500, 1700, 1900, 4000]
LABELS = ["<1300", "1300-1500", "1500-1700", "1700-1900", "≥1900"]


def fit_band(g, counts, days):
    c = M.cells(g, keys=("day", "d"))
    c["tc"] = "all"
    pos = {d: i for i, d in enumerate(days)}
    di = c["day"].map(pos).values
    pt = Logit("pooled").fit(c)
    b0, b1 = pt.coefs()["all"]
    bb = []
    for m in counts:
        f = Logit("pooled").fit(c, weight=m[di].astype(float), init=pt.w)
        bb.append(f.coefs()["all"])
    bb = np.array(bb)
    adv_s = M.sigmoid(bb[:, 0]) - 0.5
    return {"games": int(c["n"].sum()),
            "draw_rate": float((g["score"] == 0.5).mean()),
            "slope_ratio": b1 / M.ELO_SLOPE,
            "slope_ratio_ci": pct(bb[:, 1] / M.ELO_SLOPE),
            "adv_pts": b0 / b1, "adv_pts_ci": pct(bb[:, 0] / bb[:, 1]),
            "adv_score": M.sigmoid(b0) - 0.5, "adv_score_ci": pct(adv_s)}


def main():
    rng = np.random.default_rng(SEED)
    jan = load("2015-01")
    jan["rbar"] = (jan["white_elo"] + jan["black_elo"]) / 2
    jan["band"] = pd.cut(jan["rbar"], EDGES, labels=LABELS, right=False)
    days = np.sort(jan["day"].unique())
    counts = boot_counts(days, rng, B)
    rows = []
    for tc in list(TCS) + ["all"]:
        for band in LABELS:
            g = jan[(jan["band"] == band) & ((jan["tc"] == tc) | (tc == "all"))]
            r = fit_band(g, counts, days)
            r.update(tc=tc, band=band)
            rows.append(r)
            print(tc, band, r["games"], round(r["slope_ratio"], 3),
                  round(r["adv_pts"], 1), flush=True)
    df = pd.DataFrame(rows)
    for k in ("slope_ratio_ci", "adv_pts_ci", "adv_score_ci"):
        df[k + "_lo"] = df[k].str[0]
        df[k + "_hi"] = df[k].str[1]
    df.drop(columns=["slope_ratio_ci", "adv_pts_ci", "adv_score_ci"]).to_csv(
        os.path.join(ROOT, "results", "rating_bands.csv"), index=False)
    figure(df)


def figure(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plotstyle import COLORS, style_axes, setup
    setup()
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), facecolor="#fcfcfb")
    x = np.arange(len(LABELS))
    for i, tc in enumerate(TCS):
        t = df[df["tc"] == tc].set_index("band").loc[LABELS]
        off = (i - 1) * 0.12
        for a, k in ((ax[0], "slope_ratio"), (ax[1], "adv_score")):
            scale = 100 if k == "adv_score" else 1
            y = t[k].values * scale
            lo = t[k + "_ci_lo"].values * scale
            hi = t[k + "_ci_hi"].values * scale
            a.errorbar(x + off, y, yerr=[y - lo, hi - y], fmt="o-", ms=4,
                       lw=1.4, capsize=0, color=COLORS[i], label=tc.capitalize())
    ax[0].axhline(1, color="#898781", ls="--", lw=1)
    ax[0].text(len(LABELS) - 1.0, 1.0, "Elo", color="#52514e", va="bottom",
               fontsize=8)
    ax[0].set_ylabel("Fitted slope / Elo slope")
    ax[1].set_ylabel("White's expected score at d = 0, minus 50\n"
                     "(percentage points)")
    ax[1].axhline(0, color="#898781", lw=1)
    for a in ax:
        a.set_xticks(x)
        a.set_xticklabels(LABELS)
        a.set_xlabel("Mean rating of the two players")
        style_axes(a)
    ax[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "figures", "rating_bands.png"), dpi=200,
                facecolor=fig.get_facecolor())
    fig.savefig(os.path.join(ROOT, "figures", "rating_bands.pdf"),
                facecolor=fig.get_facecolor())


if __name__ == "__main__":
    main()
