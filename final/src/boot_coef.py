"""Bootstrap the baseline's coefficients: White's edge and the slope ratio.

For each group of games the course baseline logit E[S|d] = a + b d/400 is refitted
on every resample, from counts of score per gap (exact, so a replicate takes
milliseconds). Resampling schemes as in boot_test.py: game, day, player.

Groups for January: all games, time controls, bands of mean rating, players with
many games, the robustness subsets, and the history-based noise checks. Other
months get all games and time controls.

Usage: python -I src/boot_coef.py results/ 2015-01 data/headers_2015-01.csv day|game|player [B]
"""

import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from lib import *  # noqa: E402,F401,F403
from boot_test import weights  # noqa: E402

BANDS = [("m<1300", -np.inf, 1300), ("1300-1500", 1300, 1500), ("1500-1700", 1500, 1700),
         ("1700-1900", 1700, 1900), ("m>=1900", 1900, np.inf)]
CONTRASTS = [
    ("Blitz - Bullet", "Blitz", "Bullet", "edge_pp"),
    ("Classical - Bullet", "Classical", "Bullet", "edge_pp"),
    ("Classical - Blitz", "Classical", "Blitz", "edge_pp"),
    ("top band - bottom band", "m>=1900", "m<1300", "slope_ratio"),
    ("top band - bottom band (edge)", "m>=1900", "m<1300", "edge_pp"),
    ("least - most noisy (history)", "hist20_T1", "hist20_T3", "slope_ratio"),
]


def make_groups(df, full):
    G = {"All": np.arange(len(df))}
    for i, c in enumerate(CLASSES):
        G[c] = np.flatnonzero(df.tc.values == i)
    if not full:
        return G
    for name, lo, hi in BANDS:
        G[name] = np.flatnonzero((df.m.values >= lo) & (df.m.values < hi))
    for n in (10, 30, 100):
        G[f"games>={n}"] = np.flatnonzero((df.w_games.values >= n) & (df.b_games.values >= n))
    G["no_infraction"] = np.flatnonzero(~df.infr.values)
    G["no_rating_1500"] = np.flatnonzero((df.w.values != 1500) & (df.b.values != 1500))
    G["no_time_forfeit"] = np.flatnonzero(~df.tf.values)
    G["days_1_15"] = np.flatnonzero(df.day.values <= 15)
    G["days_16_31"] = np.flatnonzero(df.day.values >= 16)
    hist = (df.w_hn.values >= 20) & (df.b_hn.values >= 20)
    G["hist20"] = np.flatnonzero(hist)
    pdiff = np.fmax(df.w_hdiff.values, df.b_hdiff.values)
    ok = hist & ~np.isnan(pdiff)
    q1, q2 = np.quantile(pdiff[ok], [1 / 3, 2 / 3])
    G["hist20_T1"] = np.flatnonzero(ok & (pdiff <= q1))
    G["hist20_T2"] = np.flatnonzero(ok & (pdiff > q1) & (pdiff <= q2))
    G["hist20_T3"] = np.flatnonzero(ok & (pdiff > q2))
    G["_tercile_cuts"] = np.array([q1, q2])
    return G


def main(out_dir, month, path, scheme, B):
    df, _ = load_month(path, month)
    full = month == "2015-01"
    if full:
        add_history(df)
    G = make_groups(df, full)
    cuts = G.pop("_tercile_cuts", None)
    key = (df.d.values + DOFF).astype(np.int64)
    S = df.S.values
    sub = {n: (key[i], S[i]) for n, i in G.items()}

    def fit_all(g):
        out = {}
        for n, i in G.items():
            k, s = sub[n]
            gw = g[i]
            wins = np.bincount(k, weights=gw * s, minlength=NBIN)
            tot = np.bincount(k, weights=gw, minlength=NBIN)
            out[n] = edge_stats(*agg_fit_ab(wins, tot))
        return out

    point = fit_all(np.ones(len(df)))
    rng = np.random.default_rng([SEED, {"game": 0, "day": 1, "player": 2}[scheme], int(month[-2:]), 7])
    z = {"S": S, "day": df.day.values, "wid": df.wid.values, "bid": df.bid.values}
    stats = ["edge_pp", "edge_pts", "slope_ratio"]
    draws = {n: {s: np.empty(B) for s in stats} for n in G}
    for b in range(B):
        r = fit_all(weights(scheme, z, rng))
        for n in G:
            for s in stats:
                draws[n][s][b] = r[n][s]
    res = {"month": month, "scheme": scheme, "B": B, "n": {n: int(len(i)) for n, i in G.items()},
           "share": {n: float(len(i) / len(df)) for n, i in G.items()}, "groups": {}, "contrasts": {}}
    if cuts is not None:
        res["tercile_cuts_mean_abs_rating_change"] = [float(c) for c in cuts]
    for n in G:
        res["groups"][n] = {s: {"point": point[n][s],
                                "ci": [float(np.percentile(draws[n][s], 2.5)), float(np.percentile(draws[n][s], 97.5))],
                                "se": float(draws[n][s].std(ddof=1))} for s in stats}
    for name, a, b_, s in CONTRASTS:
        if a in G and b_ in G:
            dd = draws[a][s] - draws[b_][s]
            res["contrasts"][name + "|" + s] = {"point": point[a][s] - point[b_][s],
                                               "ci": [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))]}
    json.dump(res, open(f"{out_dir}/boot_coef_{month}_{scheme}.json", "w"), indent=1)
    print(month, scheme, {n: round(point[n]["slope_ratio"], 3) for n in list(G)[:4]}, flush=True)


if __name__ == "__main__":
    B = int(sys.argv[5]) if len(sys.argv) > 5 else 1000
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], B)
