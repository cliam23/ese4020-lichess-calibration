"""Rating-noise checks on January 2015.

smooth   check (ii): replace the displayed gap by the gap between each player's mean
         displayed rating on earlier days. Fitted on games where both players have
         at least 20 earlier-day games.
sim      check (iii): simulation with known truth. Outcomes follow the Elo formula
         exactly in TRUE strength; the displayed gap keeps January's empirical
         distribution; two noise models are compared (see simulate()).

Usage: python -I src/noise.py smooth|sim data/headers_2015-01.csv results/
"""

import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from lib import *  # noqa: E402,F401,F403
from boot_test import weights  # noqa: E402

SIGMAS = list(range(0, 101, 5)) + list(range(110, 201, 10))
N_SEEDS = 20


def smooth(path, out_dir, B=300):
    df, _ = load_month(path, "2015-01")
    add_history(df)
    ok = (df.w_hn.values >= 20) & (df.b_hn.values >= 20)
    s = df[ok].reset_index(drop=True)
    x_d = s.d.values / 400.0
    x_s = (s.w_hmean.values - s.b_hmean.values) / 400.0
    S = s.S.values

    def fits(w=None):
        a = fit_logit(x_d[:, None], S, w=w)["coef"][0] / LN10
        b = fit_logit(x_s[:, None], S, w=w)["coef"][0] / LN10
        c = fit_logit(np.column_stack([x_d, x_s]), S, w=w)["coef"] / LN10
        return {"displayed": a, "smoothed": b, "both_displayed": c[0], "both_smoothed": c[1],
                "both_sum": c[0] + c[1]}

    point = fits()
    res = {"n": int(len(s)), "share_of_games": float(len(s) / len(df)), "point": point, "ci": {}}
    z = {"S": S, "day": s.day.values, "wid": s.wid.values, "bid": s.bid.values}
    for si, scheme in enumerate(["day", "player"]):
        rng = np.random.default_rng([SEED, 11, si])
        d = {k: [] for k in point}
        for _ in range(B):
            r = fits(weights(scheme, z, rng))
            for k in d:
                d[k].append(r[k])
        res["ci"][scheme] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in d.items()}
        print("smooth", scheme, res["ci"][scheme], flush=True)
    json.dump(res, open(f"{out_dir}/noise_smooth.json", "w"), indent=1)
    print(json.dumps(point))


def simulate(d, sigma, kind, a_pts, rng):
    """One simulated January. White's true edge is a_pts rating points and
    S ~ Bernoulli(Elo(T + a_pts)) for the TRUE gap T, so the data follow the Elo
    formula exactly in true strength. Three ways to relate T to the displayed gap:

    classical  (headline) displayed rating = true strength + independent N(0, sigma^2)
        for each player, so the displayed gap is T + N(0, 2 sigma^2). True gaps are
        January's observed gaps rescaled about their mean, T = mu + sqrt(lam)(d - mu)
        with lam = 1 - 2 sigma^2 / Var(d), so the simulated displayed gap has
        January's mean and variance at every sigma (its shape is a smoothed version
        of January's).
    posterior  the displayed gap IS January's observed gap, exactly. T is drawn from
        its Gaussian posterior, T | d ~ N(mu + lam (d - mu), lam 2 sigma^2); this is
        exact if T is Gaussian.
    berkson    the displayed rating is a posterior mean: true strength = displayed +
        independent N(0, sigma^2) per player, T = d + N(0, 2 sigma^2), d as observed.
    """
    n = len(d)
    mu, v = d.mean(), d.var()
    g2 = 2.0 * sigma ** 2
    lam = 1.0 - g2 / v
    if kind != "berkson" and lam <= 0:
        return None
    if kind == "classical":
        T = mu + np.sqrt(lam) * (d - mu)
        dobs = T + rng.normal(0.0, np.sqrt(g2), n)
    elif kind == "posterior":
        T = mu + lam * (d - mu) + rng.normal(0.0, np.sqrt(lam * g2), n)
        dobs = d
    else:
        T = d + rng.normal(0.0, np.sqrt(g2), n)
        dobs = d
    S = (rng.random(n) < elo_p(T + a_pts)).astype(float)
    key = np.clip(np.rint(dobs).astype(np.int64) + DOFF, 0, NBIN - 1)
    wins = np.bincount(key, weights=S, minlength=NBIN)
    tot = np.bincount(key, minlength=NBIN).astype(float)
    out = edge_stats(*agg_fit_ab(wins, tot))
    out["gap_var"] = float(np.var(dobs))
    return out


def sim(path, out_dir):
    df, _ = load_month(path, "2015-01")
    d = df.d.values.astype(np.int64)
    wins = np.bincount(d + DOFF, weights=df.S.values, minlength=NBIN)
    tot = np.bincount(d + DOFF, minlength=NBIN).astype(float)
    obs = edge_stats(*agg_fit_ab(wins, tot))
    # True edge: the observed edge in score units, converted to rating points under Elo.
    p = 0.5 + obs["edge_pp"]
    a_pts = 400.0 * np.log(p / (1 - p)) / LN10
    res = {"observed": obs, "true_edge_pts": a_pts, "var_d": float(d.var()), "seeds": N_SEEDS,
           "sigmas": SIGMAS, "kinds": {}}
    for kind in ("classical", "posterior", "berkson"):
        rows = []
        for sg in SIGMAS:
            rng = np.random.default_rng([SEED, 3, int(sg), ["classical", "posterior", "berkson"].index(kind)])
            r = [simulate(d, sg, kind, a_pts, rng) for _ in range(N_SEEDS)]
            if r[0] is None:
                rows.append(None)
                continue
            rows.append({s: [x[s] for x in r] for s in ("slope_ratio", "edge_pp", "edge_pts", "gap_var")})
            print(kind, sg, round(np.mean(rows[-1]["slope_ratio"]), 4), flush=True)
        res["kinds"][kind] = rows
    json.dump(res, open(f"{out_dir}/noise_sim.json", "w"), indent=1)


if __name__ == "__main__":
    {"smooth": smooth, "sim": sim}[sys.argv[1]](sys.argv[2], sys.argv[3])
