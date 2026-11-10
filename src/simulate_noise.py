"""How much rating noise would it take to explain the low slope?

    python3 src/simulate_noise.py

A simulation with a known answer (topic guide: "something to check
against").  Nature: the true gap d* follows January's observed gap
distribution shrunk to variance tau^2, and White's expected score follows
the Elo curve exactly, sigma(b0* + b1* d*) with b1* = ln(10)/400 and
b0* = 0.0667 (our January estimate).  Each player's displayed rating carries
independent noise N(0, sigma^2), so the observed gap is d = d* + e with
Var(e) = 2 sigma^2, and tau^2 = Var(d) - 2 sigma^2 keeps Var(d) equal to the
data's.  Outcomes are 1, 1/2, 0 with the observed 3.7% draw rate and mean p.

For each sigma we fit the two-parameter model on N = 1.48 M simulated games
(the size of January), 20 seeds, and compare b1 / b1* with the
regression-dilution factor lambda = tau^2 / (tau^2 + 2 sigma^2).
Writes results/simulation_noise.csv and figures/simulation_noise.png.
"""
import json
import os

import numpy as np
import pandas as pd

import model as M
from baseline import ROOT, load

SIGMAS = [0, 25, 40, 50, 60, 75, 100]
SEEDS = range(20)
B0_TRUE = 0.0667
DRAW = 0.037


def simulate(d_obs_pool, sigma, rng):
    N = len(d_obs_pool)
    var_obs = d_obs_pool.var()
    tau2 = var_obs - 2 * sigma ** 2
    mu = d_obs_pool.mean()
    # true gaps: observed-shape draws, rescaled to variance tau^2
    z = rng.choice(d_obs_pool, size=N) - mu
    d_true = mu + z * np.sqrt(tau2 / var_obs)
    d = d_true + rng.normal(0, np.sqrt(2) * sigma, N) if sigma else d_true
    p = M.sigmoid(B0_TRUE + M.ELO_SLOPE * d_true)
    p = np.clip(p, DRAW / 2, 1 - DRAW / 2)
    u = rng.random(N)
    S = np.where(u < p - DRAW / 2, 1.0, np.where(u < p + DRAW / 2, 0.5, 0.0))
    return np.round(d).astype(int), S, tau2


def fit(d, S):
    g = pd.DataFrame({"d": d, "score": S, "day": 0, "tc": "all"})
    c = M.cells(g, keys=("day", "d"))
    c["tc"] = "all"
    b0, b1 = M.Logit("pooled").fit(c).coefs()["all"]
    return b0, b1


def main():
    jan = load("2015-01")
    pool = jan["d"].values.astype(float)
    rows = []
    for sigma in SIGMAS:
        for seed in SEEDS:
            rng = np.random.default_rng(1000 * sigma + seed)
            d, S, tau2 = simulate(pool, sigma, rng)
            b0, b1 = fit(d, S)
            rows.append(dict(sigma=sigma, seed=seed, b0=b0, b1=b1,
                             slope_ratio=b1 / M.ELO_SLOPE, adv_pts=b0 / b1,
                             dilution=tau2 / pool.var()))
        r = pd.DataFrame(rows)
        r = r[r.sigma == sigma]
        print(sigma, round(r.slope_ratio.mean(), 4), round(r.slope_ratio.std(), 4),
              round(r.dilution.iloc[0], 4), round(r.adv_pts.mean(), 1), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, "results", "simulation_noise.csv"), index=False)

    # per-player noise at which dilution alone would give the observed slopes
    var = pool.var()
    need = {f"{x:.2f}": float(np.sqrt((1 - x) * var / 2)) for x in (0.78, 0.82)}
    summ = (df.groupby("sigma")
              .agg(slope=("slope_ratio", "mean"), slope_sd=("slope_ratio", "std"),
                   dilution=("dilution", "first"), adv=("adv_pts", "mean"),
                   adv_sd=("adv_pts", "std")).reset_index())
    out = {"true_adv_pts": B0_TRUE / M.ELO_SLOPE, "sd_d_obs": float(pool.std()),
           "sigma_needed_per_player": need,
           "table": summ.to_dict("records")}
    with open(os.path.join(ROOT, "results", "simulation_noise.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    figure(df, summ, need)


def figure(df, summ, need):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plotstyle import COLORS, MUTED, setup, style_axes
    setup()
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.3), facecolor="#fcfcfb")
    sd = load_sd()
    s = np.linspace(0, 100, 200)
    a = ax[0]
    a.plot(s, 1 - 2 * s ** 2 / sd ** 2, color=MUTED, lw=1.2, ls="--",
           label="Linear dilution factor τ²/(τ² + 2σ²)")
    a.errorbar(summ.sigma, summ.slope, yerr=1.96 * summ.slope_sd, fmt="o-",
               ms=4.5, lw=1.4, color=COLORS[0],
               label="Logistic fit to simulated games")
    for y, lab in ((0.782, "Observed: all January games"),
                   (0.828, "Observed: both players ≥ 100 earlier-day games")):
        a.axhline(y, color="#52514e", lw=0.8, ls=":")
        a.text(99, y + 0.008, lab, color="#52514e", fontsize=7, ha="right")
    a.set_xlabel("Rating noise per player, σ (rating points)")
    a.set_ylabel("Fitted slope / true slope")
    a.set_ylim(0.4, 1.05)
    a.legend(frameon=False, fontsize=7.5, loc="lower left")
    b = ax[1]
    df = df.assign(r_pts=df.adv_pts / (B0_TRUE / M.ELO_SLOPE),
                   r_score=(M.sigmoid(df.b0) - 0.5) / (M.sigmoid(B0_TRUE) - 0.5))
    g = df.groupby("sigma").agg(pm=("r_pts", "mean"), ps=("r_pts", "std"),
                                sm=("r_score", "mean"), ss=("r_score", "std"))
    b.errorbar(g.index, g.pm, yerr=1.96 * g.ps, fmt="o-", ms=4.5, lw=1.4,
               color=COLORS[1], label="In rating points, b0/b1")
    b.errorbar(g.index, g.sm, yerr=1.96 * g.ss, fmt="s-", ms=4.5, lw=1.4,
               color=COLORS[2], label="In expected score, σ(b0) − ½")
    b.axhline(1, color=MUTED, lw=1, ls="--")
    b.set_xlabel("Rating noise per player, σ (rating points)")
    b.set_ylabel("Estimated / true White advantage")
    b.legend(frameon=False, fontsize=7.5, loc="upper left")
    for x in ax:
        style_axes(x)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(ROOT, "figures", f"simulation_noise.{ext}"),
                    dpi=200, facecolor=fig.get_facecolor())


def load_sd():
    with open(os.path.join(ROOT, "results", "simulation_noise.json")) as f:
        return json.load(f)["sd_d_obs"]


if __name__ == "__main__":
    main()
