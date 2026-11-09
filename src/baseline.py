"""Baseline checkpoint: Elo, the mean-score floor and logistic regression.

Run from the project folder after parse_pgn.py has written both months:
    python3 src/baseline.py

Fits and selects on January 2015 only.  February 2015 is read at the end
to score the three pre-specified baselines once (proposal feedback, step 3);
no choice in this script looks at February.
Writes results/results.json, results/summary.md and figures/baseline.png.
"""
import json
import os
import time

import numpy as np
import pandas as pd

import model as M
from model import TCS, Logit

SEED = 4020
B = 2000                     # day-bootstrap replicates
K = 5                        # CV folds, contiguous blocks of days
LAMBDAS = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
STRUCTURES = ["pooled", "by_tc"]
NOISE_MIN_GAMES = [1, 10, 30, 100, 300]   # both players with >= k games that month
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(month):
    df = pd.read_csv(os.path.join(ROOT, "data", "processed",
                                  f"games_{month}.csv.gz"))
    df["d"] = df["white_elo"] - df["black_elo"]
    df["tc"] = pd.Categorical(df["tc"], categories=TCS)
    return df


def boot_counts(days, rng, b=B):
    """b x D matrix: how often each day appears in a day-level resample."""
    D = len(days)
    idx = rng.integers(0, D, size=(b, D))
    return np.stack([np.bincount(r, minlength=D) for r in idx])


def pct(x):
    lo, hi = np.percentile(x, [2.5, 97.5])
    return [float(lo), float(hi)]


def cross_validate(cJ, days):
    folds = np.array_split(np.sort(days), K)
    rows = []
    for f, held in enumerate(folds):
        te = cJ["day"].isin(held).values
        tr, ts = cJ[~te], cJ[te]
        n, s = ts["n"].values, ts["s"].values
        rows.append({"fold": f, "model": "elo", "lam": None,
                     "logloss": M.log_loss(M.elo_prob(ts["d"].values), n, s),
                     "games": int(n.sum())})
        floor = tr["s"].sum() / tr["n"].sum()
        rows.append({"fold": f, "model": "floor", "lam": None,
                     "logloss": M.log_loss(np.full(len(ts), floor), n, s),
                     "games": int(n.sum())})
        for st in STRUCTURES:
            for lam in LAMBDAS:
                m = Logit(st, lam).fit(tr)
                rows.append({"fold": f, "model": st, "lam": lam,
                             "logloss": M.log_loss(m.predict(ts), n, s),
                             "games": int(n.sum())})
    cv = pd.DataFrame(rows)
    # Section 4A: the K-fold loss averages the fold losses weighted by fold
    # size, Loss_KF = sum_k (N_k / |D_KF|) Loss_k.  The SE is the naive one,
    # SD over folds / sqrt(K); fold scores are correlated, so read it as rough.
    cv["wl"] = cv["logloss"] * cv["games"]
    summ = (cv.groupby(["model", "lam"], dropna=False)
              .agg(wl=("wl", "sum"), games=("games", "sum"),
                   std=("logloss", "std")).reset_index())
    summ["mean"] = summ["wl"] / summ["games"]
    summ["se"] = summ["std"] / np.sqrt(K)
    summ = summ.drop(columns=["wl", "games"])
    cv = cv.drop(columns="wl")
    return cv, summ, [list(map(int, f)) for f in folds]


def day_bootstrap_fit(c, structure, counts, days):
    pos = {d: i for i, d in enumerate(days)}
    di = c["day"].map(pos).values
    w0 = Logit(structure).fit(c).w
    out = []
    for m in counts:
        fit = Logit(structure).fit(c, weight=m[di].astype(float), init=w0)
        out.append(fit.coefs())
    return out


def summarise_coefs(point, boots):
    res = {}
    for key, (b0, b1) in point.items():
        bb0 = np.array([b[key][0] for b in boots])
        bb1 = np.array([b[key][1] for b in boots])
        res[key] = {
            "b0": float(b0), "b0_ci": pct(bb0),
            "b1": float(b1), "b1_ci": pct(bb1),
            "b1_over_elo": float(b1 / M.ELO_SLOPE),
            "b1_over_elo_ci": pct(bb1 / M.ELO_SLOPE),
            "white_adv_pts": float(b0 / b1), "white_adv_pts_ci": pct(bb0 / bb1),
            "white_adv_pts_at_elo_slope": float(b0 / M.ELO_SLOPE),
        }
    return res


def score_test(cF, preds, rng):
    """Log-loss and Brier on February, with day-bootstrap 95% intervals."""
    days = np.sort(cF["day"].unique())
    pos = {d: i for i, d in enumerate(days)}
    di = cF["day"].map(pos).values
    n, s, s2 = (cF[k].values.astype(float) for k in ("n", "s", "s2"))
    Nd = np.bincount(di, weights=n, minlength=len(days))
    counts = boot_counts(days, rng)
    per_day, res = {}, {}
    for name, p in preds.items():
        pc = np.clip(p, 1e-12, 1 - 1e-12)
        ll = -(s * np.log(pc) + (n - s) * np.log1p(-pc))
        br = s2 - 2 * p * s + n * p ** 2
        per_day[name] = (np.bincount(di, weights=ll, minlength=len(days)),
                         np.bincount(di, weights=br, minlength=len(days)))
    Nb = counts @ Nd
    for name, (L, Br) in per_day.items():
        res[name] = {
            "logloss": float(L.sum() / Nd.sum()),
            "logloss_ci": pct(counts @ L / Nb),
            "brier": float(Br.sum() / Nd.sum()),
            "brier_ci": pct(counts @ Br / Nb),
        }
        if name != "elo":
            dL = L - per_day["elo"][0]
            res[name]["logloss_minus_elo"] = float(dL.sum() / Nd.sum())
            res[name]["logloss_minus_elo_ci"] = pct(counts @ dL / Nb)
    return res


def calibration_table(df, preds, edges):
    t = df[["tc", "d", "score"]].copy()
    for k, v in preds.items():
        t["p_" + k] = v
    t = t[(t["d"] >= edges[0]) & (t["d"] < edges[-1])]
    t["bin"] = pd.cut(t["d"], edges, right=False)
    agg = {"n": ("score", "size"), "obs": ("score", "mean"),
           "sd": ("score", "std"), "d_mid": ("d", "mean")}
    agg.update({"p_" + k: ("p_" + k, "mean") for k in preds})
    out = t.groupby(["tc", "bin"], observed=True).agg(**agg).reset_index()
    out["se"] = out["sd"] / np.sqrt(out["n"])
    return out[out["n"] >= 200]


def figure(tab, path, labels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid = "#0b0b0b", "#898781", "#e1e0d9"
    colors = {"elo": "#2a78d6", "fit": "#eb6834"}
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c3c2b7",
                         "axes.labelcolor": ink, "xtick.color": muted,
                         "ytick.color": muted})
    fig, ax = plt.subplots(2, 3, figsize=(10, 6.2), sharex="row",
                           sharey="row", facecolor="#fcfcfb",
                           gridspec_kw={"height_ratios": [1.25, 1]})
    for j, tc in enumerate(TCS):
        t = tab[tab["tc"] == tc]
        a, r = ax[0, j], ax[1, j]
        a.plot([0, 1], [0, 1], color=muted, lw=1, ls="--", zorder=1)
        for off, k in ((-6, "elo"), (6, "fit")):
            a.plot(t["p_" + k], t["obs"], "o", ms=4, color=colors[k],
                   mec="#fcfcfb", mew=0.8, label=labels[k], zorder=3)
            r.errorbar(t["d_mid"] + off, t["obs"] - t["p_" + k],
                       yerr=1.96 * t["se"], fmt="o", ms=3.5, lw=1,
                       color=colors[k], label=labels[k])
        r.axhline(0, color=muted, lw=1)
        a.set_title(f"{tc.capitalize()}  (n = {int(t['n'].sum()):,})",
                    color=ink, fontsize=10, loc="left")
        for x in (a, r):
            x.grid(color=grid, lw=0.6)
            x.set_axisbelow(True)
            x.set_facecolor("#fcfcfb")
            for side in ("top", "right"):
                x.spines[side].set_visible(False)
        r.set_xlabel("Rating gap d = WhiteElo − BlackElo (points)")
    ax[0, 0].set_ylabel("Observed mean score for White")
    ax[0, 0].set_xlabel("Predicted mean score (bin average)")
    ax[1, 0].set_ylabel("Observed − predicted (±1.96 SE)")
    ax[0, 0].legend(frameon=False, loc="upper left")
    fig.suptitle("February 2015 (held out): observed vs predicted White score "
                 "in 50-point bins of rating gap", x=0.01, ha="left",
                 color=ink, fontsize=11)
    fig.text(0.01, 0.005,
             "Elo's curve is too steep: it overstates the stronger side's score "
             "by up to about 0.07. The logistic fit (slope ≈ 0.78 × Elo, fitted "
             "on January) removes most of the miss;\nwhat remains lies beyond "
             "|d| = 400. Bins with at least 200 games; bars are ±1.96 SE of "
             "the bin's mean score.",
             fontsize=8, color="#52514e", ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)

    jan = load("2015-01")
    cJ = M.cells(jan)
    days = np.sort(jan["day"].unique())
    R = {"seed": SEED, "B": B, "K": K,
         "jan": {"games": len(jan), "days": len(days),
                 "mean_score": float(jan["score"].mean()),
                 "draw_rate": float((jan["score"] == 0.5).mean()),
                 "by_tc": jan["tc"].value_counts().reindex(TCS).to_dict(),
                 "d_sd": float(jan["d"].std()),
                 "abs_d_over_600": float((jan["d"].abs() >= 600).mean())}}

    # 1. Model selection by day-blocked 5-fold CV on January
    cv, summ, folds = cross_validate(cJ, days)
    cv.to_csv(os.path.join(ROOT, "results", "cv_folds.csv"), index=False)
    a = cv[(cv.model == "by_tc") & (cv.lam == 0)]["logloss"].values
    p = cv[(cv.model == "pooled") & (cv.lam == 0)]["logloss"].values
    e = cv[cv.model == "elo"]["logloss"].values
    paired = {"by_tc_minus_pooled": float((a - p).mean()),
              "by_tc_minus_pooled_se": float((a - p).std(ddof=1) / np.sqrt(K)),
              "pooled_minus_elo": float((p - e).mean()),
              "pooled_minus_elo_se": float((p - e).std(ddof=1) / np.sqrt(K))}
    # Ties within 1e-5 log-loss are noise at this N: break them toward the
    # simpler structure and the smaller penalty.
    grid = summ[summ["model"].isin(STRUCTURES)].copy()
    grid["key"] = grid["mean"].round(5)
    grid["simple"] = grid["model"].map({s: i for i, s in enumerate(STRUCTURES)})
    best = grid.sort_values(["key", "simple", "lam"]).iloc[0]
    R["cv"] = {"folds_days": folds,
               "table": summ.replace({np.nan: None}).to_dict("records"),
               "chosen": {"structure": best["model"], "lam": float(best["lam"]),
                          "logloss": float(best["mean"]),
                          "se": float(best["se"])},
               "paired": paired}
    print("CV done", round(time.time() - t0, 1), "s")

    # 2. Two-parameter and by-time-control fits on all January,
    #    with day-bootstrap percentile intervals
    counts = boot_counts(days, rng)
    pooled = Logit("pooled").fit(cJ)
    bytc = Logit("by_tc").fit(cJ)
    R["fit_pooled"] = summarise_coefs(
        pooled.coefs(), day_bootstrap_fit(cJ, "pooled", counts, days))
    R["fit_by_tc"] = summarise_coefs(
        bytc.coefs(), day_bootstrap_fit(cJ, "by_tc", counts, days))
    print("bootstrap done", round(time.time() - t0, 1), "s")

    # 3. Rating-noise check: refit on games between players seen often
    noise = []
    small = counts[:500]
    for k in NOISE_MIN_GAMES:
        sub = jan[(jan["n_white"] >= k) & (jan["n_black"] >= k)]
        if len(sub) < 10000:
            noise.append({"min_games": k, "games": len(sub),
                          "share": len(sub) / len(jan), "skipped": True})
            continue
        c = M.cells(sub)
        point = Logit("pooled").fit(c).coefs()
        boots = day_bootstrap_fit(c, "pooled", small, days)
        r = summarise_coefs(point, boots)["all"]
        r.update({"min_games": k, "games": len(sub),
                  "share": len(sub) / len(jan)})
        noise.append(r)
    R["noise_check"] = noise

    # 4. Score the pre-specified baselines once on February
    feb = load("2015-02")
    cF = M.cells(feb)
    chosen = Logit(best["model"], float(best["lam"])).fit(cJ)
    floor = jan["score"].mean()
    preds = {"elo": M.elo_prob(cF["d"].values),
             "floor": np.full(len(cF), floor),
             "logit_2param": pooled.predict(cF),
             "logit_cv_chosen": chosen.predict(cF)}
    R["feb"] = {"games": len(feb),
                "mean_score": float(feb["score"].mean()),
                "scores": score_test(cF, preds, rng)}

    # 5. Key figure
    edges = np.arange(-600, 601, 50)
    tab = calibration_table(
        feb, {"elo": M.elo_prob(feb["d"].values), "fit": chosen.predict(feb)},
        edges)
    tab.to_csv(os.path.join(ROOT, "results", "calibration_feb.csv"),
               index=False)
    figure(tab, os.path.join(ROOT, "figures", "baseline.png"),
           {"elo": "Elo formula",
            "fit": "Logistic fit (CV-chosen)"})

    with open(os.path.join(ROOT, "results", "results.json"), "w") as f:
        json.dump(R, f, indent=2, default=float)
    write_summary(R)
    print("all done", round(time.time() - t0, 1), "s")


def write_summary(R):
    L = ["# Baseline results (auto-generated by src/baseline.py)", ""]
    j = R["jan"]
    L.append(f"January: {j['games']:,} games over {j['days']} days; White mean "
             f"score {j['mean_score']:.4f}; draw rate {j['draw_rate']:.4f}.")
    L += ["", "## 5-fold day-blocked CV log-loss on January (fold-size-weighted mean ± naive SE over folds)"]
    for r in R["cv"]["table"]:
        lam = "" if r["lam"] is None else f" lam={r['lam']:g}"
        L.append(f"- {r['model']}{lam}: {r['mean']:.5f} ± {r['se']:.5f}")
    c = R["cv"]["chosen"]
    L.append(f"Chosen: {c['structure']}, lam={c['lam']:g}")
    for k, v in R["cv"]["paired"].items():
        if not k.endswith("_se"):
            L.append(f"Paired over folds, {k}: {v:+.6f} ± {R['cv']['paired'][k + '_se']:.6f}")
    L += ["", "## Fits on January (day bootstrap, 95% percentile CI)"]
    for name in ("fit_pooled", "fit_by_tc"):
        for k, v in R[name].items():
            L.append(
                f"- {name}/{k}: b1 = {v['b1']:.5f} [{v['b1_ci'][0]:.5f}, "
                f"{v['b1_ci'][1]:.5f}] = {v['b1_over_elo']:.3f}x Elo; "
                f"b0 = {v['b0']:.4f}; White adv = {v['white_adv_pts']:.1f} "
                f"[{v['white_adv_pts_ci'][0]:.1f}, {v['white_adv_pts_ci'][1]:.1f}] pts")
    L += ["", "## Rating-noise check (pooled fit, both players >= k games in January)"]
    for v in R["noise_check"]:
        if v.get("skipped"):
            L.append(f"- k={v['min_games']}: only {v['games']:,} games, skipped")
            continue
        L.append(f"- k={v['min_games']}: {v['games']:,} games ({v['share']:.1%}); "
                 f"b1 = {v['b1']:.5f} [{v['b1_ci'][0]:.5f}, {v['b1_ci'][1]:.5f}] "
                 f"= {v['b1_over_elo']:.3f}x Elo; White adv {v['white_adv_pts']:.1f} pts")
    L += ["", f"## February ({R['feb']['games']:,} games), scored once"]
    for k, v in R["feb"]["scores"].items():
        extra = ""
        if "logloss_minus_elo" in v:
            extra = (f"; minus Elo {v['logloss_minus_elo']:+.5f} "
                     f"[{v['logloss_minus_elo_ci'][0]:+.5f}, "
                     f"{v['logloss_minus_elo_ci'][1]:+.5f}]")
        L.append(f"- {k}: log-loss {v['logloss']:.5f} [{v['logloss_ci'][0]:.5f}, "
                 f"{v['logloss_ci'][1]:.5f}]; Brier {v['brier']:.5f} "
                 f"[{v['brier_ci'][0]:.5f}, {v['brier_ci'][1]:.5f}]{extra}")
    with open(os.path.join(ROOT, "results", "summary.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
