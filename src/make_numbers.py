"""Collect every number the paper reports into paper/numbers.tex.

    python3 src/make_numbers.py

Reads only files in results/ and data/processed/ written by the other
scripts, and writes LaTeX macros (\\newcommand), so no number in the paper
is typed by hand.  Also writes results/paper_numbers.json for checking.
"""
import json
import os

import numpy as np
import pandas as pd

import model as M
from baseline import ROOT

R = os.path.join(ROOT, "results")
D = os.path.join(ROOT, "data", "processed")
OUT = {}


WORDS = {"10": "Ten", "20": "Twenty", "30": "Thirty", "100": "Hundred"}


def put(name, value, fmt="{}"):
    import re
    name = re.sub(r"\d+", lambda m: WORDS[m.group()], name)
    OUT[name] = fmt.format(value) if not isinstance(value, str) else value


def c(x):  # thousands separator for LaTeX
    return f"{int(x):,}".replace(",", "{,}")


def ci(lo, hi, f="{:.3f}"):
    return f"[{f.format(lo)}, {f.format(hi)}]"


def main():
    # ---- data ------------------------------------------------------------
    for m, tag in (("2015-01", "Jan"), ("2015-02", "Feb")):
        f = json.load(open(os.path.join(D, f"filters_{m}.json")))
        put(f"games{tag}", c(f["games_in_file"]))
        put(f"kept{tag}", c(f["kept"]))
        put(f"players{tag}", c(f["players"]))
        for k, v in f["removed_in_order"].items():
            put(f"rm{tag}{k.title().replace('_', '')}", c(v))
        for t, v in f["kept_by_tc"].items():
            put(f"n{tag}{t.title()}", c(v))
    base = json.load(open(os.path.join(R, "results.json")))
    put("meanScoreJan", base["jan"]["mean_score"], "{:.3f}")
    put("drawPctJan", 100 * base["jan"]["draw_rate"], "{:.1f}")
    put("sdGapJan", base["jan"]["d_sd"], "{:.0f}")

    # ---- two-parameter fit and by time control (January, day bootstrap) ---
    fp = base["fit_pooled"]["all"]
    put("betaOne", fp["b1"], "{:.5f}")
    put("betaOneCI", ci(*fp["b1_ci"], f="{:.5f}"))
    put("slopeRatio", fp["b1_over_elo"], "{:.2f}")
    put("slopeRatioCI", ci(*fp["b1_over_elo_ci"]))
    put("betaZero", fp["b0"], "{:.4f}")
    put("advPts", fp["white_adv_pts"], "{:.1f}")
    put("advPtsCI", ci(*fp["white_adv_pts_ci"], f="{:.1f}"))
    put("advScorePct", 100 * (M.sigmoid(fp["b0"]) - 0.5), "{:.2f}")
    for t, v in base["fit_by_tc"].items():
        T = t.title()
        put(f"slope{T}", v["b1_over_elo"], "{:.2f}")
        put(f"slope{T}CI", ci(*v["b1_over_elo_ci"], f="{:.2f}"))
        put(f"adv{T}", v["white_adv_pts"], "{:.1f}")
        put(f"adv{T}CI", ci(*v["white_adv_pts_ci"], f="{:.1f}"))
        put(f"advScore{T}", 100 * (M.sigmoid(v["b0"]) - 0.5), "{:.2f}")

    # ---- cross-validation (January) --------------------------------------
    cv = pd.read_csv(os.path.join(R, "cv_extended_summary.csv"))
    sel = json.load(open(os.path.join(R, "selection.json")))
    best = {}
    for lvl, key in ((1, "Pooled"), (2, "Base"), (3, "Rating"), (4, "Spline")):
        b = cv[(cv.family == "logit") & (cv.level == lvl)].sort_values("cv_logloss").iloc[0]
        best[key] = b
    best["Gbm"] = cv[cv.family == "gbm"].sort_values("cv_logloss").iloc[0]
    for k, b in best.items():
        put(f"cv{k}", b.cv_logloss, "{:.5f}")
        put(f"cv{k}SE", b.naive_se, "{:.5f}")
        put(f"cv{k}Diff", 1e4 * b.minus_baseline, "{:+.2f}")
        put(f"cv{k}DiffSE", 1e4 * b.minus_baseline_se, "{:.2f}")
    put("splineDf", int(sel["logit"]["df"]))
    put("gbmTrees", int(sel["gbm"]["max_iter"]))
    put("gbmLeaves", int(sel["gbm"]["max_leaf_nodes"]))
    put("gbmRate", sel["gbm"]["learning_rate"], "{:g}")
    put("nLogitConfigs", int((cv.family == "logit").sum()))
    put("nGbmConfigs", int((cv.family == "gbm").sum()))

    # ---- February, scored once --------------------------------------------
    ft = json.load(open(os.path.join(R, "final_test.json")))
    keys = {"Elo": "Elo formula", "Floor": "Mean-score floor",
            "Pooled": "Logit: d (pooled)",
            "Base": "Logit: d by time control (baseline)"}
    for k in ft["scores"]:
        if k.startswith("Selected logit"):
            keys["Spline"] = k
        if k.startswith("Gradient boosting"):
            keys["Gbm"] = k
    for tag, k in keys.items():
        s = ft["scores"][k]
        put(f"te{tag}LL", s["logloss"], "{:.4f}")
        put(f"te{tag}LLCI", ci(*s["logloss_ci"], f="{:.4f}"))
        put(f"te{tag}Brier", s["brier"], "{:.4f}")
        put(f"te{tag}BrierCI", ci(*s["brier_ci"], f="{:.4f}"))
        put(f"te{tag}VsBase", 1e4 * s["minus_baseline"], "{:+.1f}")
        put(f"te{tag}VsBaseCI", ci(*(1e4 * np.array(s["minus_baseline_ci"])), f="{:+.1f}"))
        put(f"te{tag}VsElo", 1e4 * s["minus_elo"], "{:+.1f}")
        put(f"te{tag}VsEloCI", ci(*(1e4 * np.array(s["minus_elo_ci"])), f="{:+.1f}"))
    ece = ft["mean_abs_calibration_error"]
    for tag, k in (("Elo", "elo"), ("Base", "base"), ("Spline", "final"), ("Gbm", "gbm")):
        put(f"cal{tag}", ece[k], "{:.4f}")
    put("calRatio", ece["elo"] / ece["base"], "{:.0f}")

    # ---- rating noise --------------------------------------------------------
    nc = {r["min_games"]: r for r in base["noise_check"] if not r.get("skipped")}
    for k in (10, 30, 100):
        put(f"freqSlope{k}", nc[k]["b1_over_elo"], "{:.2f}")
        put(f"freqShare{k}", 100 * nc[k]["share"], "{:.0f}")
    npv = pd.read_csv(os.path.join(R, "noise_prev.csv"))
    for k in (20, 100):
        g = npv[npv.min_prev == k].set_index("design")
        put(f"prev{k}Share", 100 * g.share.iloc[0], "{:.0f}")
        put(f"prev{k}Disp", g.loc["displayed gap d", "b_d_over_elo"], "{:.2f}")
        put(f"prev{k}Smooth", g.loc["earlier-day gap d_prev", "b_dprev_over_elo"], "{:.2f}")
        both = g.loc["both"]
        put(f"prev{k}BothD", both.b_d_over_elo, "{:.2f}")
        put(f"prev{k}BothPrev", both.b_dprev_over_elo, "{:.2f}")
        put(f"prev{k}Sum", both.sum_over_elo, "{:.2f}")
        put(f"prev{k}Corr", g["corr"].iloc[0], "{:.2f}")
    sim = json.load(open(os.path.join(R, "simulation_noise.json")))
    tab = pd.DataFrame(sim["table"])
    for target, tag in ((0.78, "All"), (0.82, "Freq")):
        s = np.interp(-target, -tab.slope.values, tab.sigma.values)
        put(f"simSigma{tag}", s, "{:.0f}")
    row50 = tab[tab.sigma == 50].iloc[0]
    put("simSlopeFifty", row50.slope, "{:.2f}")
    put("simAdvFifty", row50.adv, "{:.1f}")
    put("simAdvTrue", sim["true_adv_pts"], "{:.1f}")
    put("simDilutionFifty", row50.dilution, "{:.2f}")
    simdf = pd.read_csv(os.path.join(R, "simulation_noise.csv"))
    s50 = simdf[simdf.sigma == 50]
    r_score = ((M.sigmoid(s50.b0) - 0.5) / (M.sigmoid(0.0667) - 0.5)).mean()
    put("simScoreBiasFifty", 100 * (r_score - 1), "{:.0f}")

    # ---- rating bands ---------------------------------------------------------
    rb = pd.read_csv(os.path.join(R, "rating_bands.csv"))
    a = rb[rb.tc == "all"].set_index("band")
    put("bandSlopeLow", a.slope_ratio.min(), "{:.2f}")
    put("bandSlopeHigh", a.loc["≥1900", "slope_ratio"], "{:.2f}")
    put("bandSlopeHighCI", ci(a.loc["≥1900", "slope_ratio_ci_lo"],
                              a.loc["≥1900", "slope_ratio_ci_hi"], f="{:.2f}"))
    put("bandAdvScoreMin", 100 * a.adv_score.min(), "{:.1f}")
    put("bandAdvScoreMax", 100 * a.adv_score.max(), "{:.1f}")
    put("bandDrawLow", 100 * a.draw_rate.iloc[0], "{:.1f}")
    put("bandDrawHigh", 100 * a.draw_rate.iloc[-1], "{:.1f}")

    # ---- sensitivity ------------------------------------------------------------
    se = pd.read_csv(os.path.join(R, "sensitivity.csv")).set_index("subset")
    def row(prefix, name):
        r = se.loc[name]
        put(f"sens{prefix}Slope", r.slope_ratio, "{:.2f}")
        put(f"sens{prefix}Adv", r.adv_pts, "{:.1f}")
        put(f"sens{prefix}AdvCI", ci(r.adv_lo, r.adv_hi, f="{:.1f}"))
        put(f"sens{prefix}Share", 100 * r.share, "{:.0f}")
    row("Forfeit", "Drop time-forfeit endings")
    row("Default", "Drop games with a rating of exactly 1500 (new-account default)")
    row("Gap", "Only |d| < 400")
    row("Infraction", "Drop 'Rules infraction' endings")
    row("FirstHalf", "Days 1-15 only")
    row("SecondHalf", "Days 16-31 only")


    # ---- derived statements used in prose ------------------------------------
    elo_gain = 1e4 * ft["scores"][keys["Elo"]]["minus_baseline"]
    put("eloGain", elo_gain, "{:.0f}")
    sp = -1e4 * ft["scores"][keys["Spline"]]["minus_baseline"]
    gb = -1e4 * ft["scores"][keys["Gbm"]]["minus_baseline"]
    put("splineGain", sp, "{:.1f}")
    put("gbmGain", gb, "{:.1f}")
    put("ratioGbm", elo_gain / gb, "{:.0f}")
    put("ratioSpline", elo_gain / sp, "{:.0f}")
    cal = pd.read_csv(os.path.join(R, "calibration_final.csv"))
    put("maxEloMiss", (cal.obs - cal.p_elo).abs().max(), "{:.2f}")
    put("maxFitMiss", (cal.obs - cal.p_final).abs().max(), "{:.3f}")
    advs = [v["white_adv_pts"] for v in base["fit_by_tc"].values()]
    put("advSpread", max(advs) - min(advs), "{:.1f}")
    fj = json.load(open(os.path.join(D, "filters_2015-01.json")))
    put("nInfraction", c(fj["kept_by_termination"].get("Rules infraction", 0)))
    put("nForfeitPct", 100 * fj["kept_by_termination"]["Time forfeit"] / fj["kept"], "{:.0f}")
    put("testGames", c(ft["february_games"]))
    put("slopeGapPct", 100 * (1 - fp["b1_over_elo"]), "{:.0f}")

    tables(cv, se)

    os.makedirs(os.path.join(ROOT, "paper"), exist_ok=True)
    with open(os.path.join(ROOT, "paper", "numbers.tex"), "w") as f:
        f.write("% Auto-generated by src/make_numbers.py. Do not edit.\n")
        for k, v in OUT.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    with open(os.path.join(R, "paper_numbers.json"), "w") as f:
        json.dump(OUT, f, indent=1)
    print(len(OUT), "numbers written")


def tables(cv, se):
    """Appendix tables, written straight from the results files."""
    os.makedirs(os.path.join(ROOT, "paper"), exist_ok=True)
    L = [r"\begin{table}[h]", r"\centering\small",
         r"\caption{January 5-fold day-blocked CV for every configuration: "
         r"fold-size-weighted log-loss, its naive SE, and the paired difference "
         r"from the baseline in units of $10^{-4}$ ($\pm$ SE of the paired fold "
         r"differences). Selected models in bold.}",
         r"\label{tab:cv}", r"\begin{tabular}{lccc}", r"\toprule",
         r"Configuration & CV log-loss & naive SE & $\Delta$ vs.\ baseline \\",
         r"\midrule"]
    first = {"logit": True, "gbm": True}
    for fam in ("logit", "gbm"):
        g = cv[cv.family == fam].sort_values("cv_logloss")
        for i, r in enumerate(g.itertuples()):
            lab = r.label.replace("Logit: ", "Logit, ").replace("λ", r"$\lambda$")
            lab = lab.replace("≥", r"$\geq$")
            row = (f"{lab} & {r.cv_logloss:.5f} & {r.naive_se:.5f} & "
                   f"{1e4 * r.minus_baseline:+.2f} $\\pm$ {1e4 * r.minus_baseline_se:.2f}")
            if i == 0:
                row = r"\textbf{" + lab + "}" + row[len(lab):]
            L.append(row + r" \\")
        if fam == "logit":
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(ROOT, "paper", "cv_table.tex"), "w").write("\n".join(L) + "\n")

    S = [r"\begin{table}[h]", r"\centering\small",
         r"\caption{Two-parameter fits on subsets of January games, with 95\% "
         r"day-bootstrap intervals ($B = 500$).}", r"\label{tab:sens}",
         r"\begin{tabular}{lccc}", r"\toprule",
         r"Subset & Share & Slope / Elo & White adv.\ (points) \\", r"\midrule"]
    for name, r in se.iterrows():
        nm = name.replace("≥", r"$\geq$").replace("|d|", r"$|d|$")
        S.append(f"{nm} & {100 * r.share:.1f}\\% & {r.slope_ratio:.3f} "
                 f"[{r.slope_lo:.3f}, {r.slope_hi:.3f}] & {r.adv_pts:.1f} "
                 f"[{r.adv_lo:.1f}, {r.adv_hi:.1f}] \\\\")
    S += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(ROOT, "paper", "sens_table.tex"), "w").write("\n".join(S) + "\n")


if __name__ == "__main__":
    main()
