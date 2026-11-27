"""Write every table and every number the paper quotes, from results/*.json.

Headline error bars are 95% percentile intervals from the player-level (pigeonhole)
bootstrap; game-level and day-block intervals are reported alongside in Table 4.

Usage: python -I src/make_tables.py results/ paper/
"""

import json
import math
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from lib import LN10, elo_p, sigmoid  # noqa: E402

R, P = sys.argv[1], sys.argv[2]
J = lambda f: json.load(open(f"{R}/{f}"))
fz, sc, cv = J("frozen_selection.json"), J("scores.json"), J("cv_all.json")
bt = {**J("boot_test_2015-02.json"), **J("boot_test_2015-03.json"), **J("boot_test_2015-03L.json")}
bc = {m: {s: J(f"boot_coef_{m}_{s}.json") for s in ("game", "day", "player")} for m in ("2015-01", "2015-02", "2015-03")}
sm, sim, ss, f1 = J("noise_smooth.json"), J("noise_sim.json"), J("sigma_star.json"), J("fig1_info.json")
for _k, _n in fz["chosen_names"].items():
    cv[_k] = cv[_n]
MON = {"2015-01": "Jan", "2015-02": "Feb", "2015-03": "Mar"}
BASE = fz["baseline"]
K = fz["spline"]["K"]
MODELS = [("floor", "Floor (White's mean score)", "1"), ("elo", "Elo formula", "0"),
          ("pooled", "Logistic on $d$, pooled", "2"), ("bytc", "Logistic on $d$ by time control", "6"),
          ("rating", "Rating logistic (adds mean rating)", "9"), ("spline", "Spline logistic", str(9 + K - 2)),
          ("boost", "Boosted trees", "--"), ("ordinal", "Ordinal logit (draws modelled)", "9")]


def mn(x, nd=1, plus=False):
    """Number with a real minus sign (math mode)."""
    s = f"{abs(x):.{nd}f}"
    if round(x, nd) < 0:
        return f"$-${s}"
    return f"+{s}" if plus else s


def hw(a):
    return (a[2] - a[1]) / 2.0


def pm(a, scale=1.0, nd=1, plus=False):
    return f"{mn(a[0] * scale, nd, plus)} $\\pm$ {hw(a) * scale:.{nd}f}"


def ci(lo, hi, scale=1.0, nd=1):
    return f"[{mn(lo * scale, nd)}, {mn(hi * scale, nd)}]"


macros = {}


def lamtex(x):
    return "0" if x == 0 else f"10^{{{int(round(math.log10(x)))}}}"


def put(name, val):
    macros[name] = str(val)


# ---------------------------------------------------------------- data summary
for m, key in (("2015-01", "Jan"), ("2015-02", "Feb"), ("2015-03", "Mar")):
    st = sc["train_steps"] if m == "2015-01" else sc["months"][m]["steps"]
    sm_ = None
    put(f"nRaw{key}", f"{st['raw']:,}")
    put(f"nGames{key}", f"{st['final']:,}")
months_summary = {"2015-01": None}
# January summary is rebuilt from boot_coef counts + scores of Feb/Mar; train summary stored in cv_linear
jan_n = bc["2015-01"]["player"]["n"]["All"]
assert f"{jan_n:,}" == macros["nGamesJan"]
for m, key in (("2015-02", "Feb"), ("2015-03", "Mar")):
    s = sc["months"][m]["summary"]
    put(f"nDays{key}", s["n_days"])
    put(f"nPlayers{key}", f"{s['n_players']:,}")
put("nDaysJan", 31)

# ---------------------------------------------------------------- model table
def tb(month, scheme="player"):
    return bt[month][scheme]


rows = []
for key, label, par in MODELS:
    c = cv[key]["cv"]
    f, m_ = tb("2015-02"), tb("2015-03")
    cells = [label + (r" (baseline)" if key == BASE else ""), par, f"{c:.5f}", f"{f['loss'][key][0]:.4f} $\\pm$ {hw(f['loss'][key]):.4f}"]
    for mm in (f, m_):
        for ref in ("d_elo", "d_base"):
            if (ref == "d_elo" and key == "elo") or (ref == "d_base" and key == BASE):
                cells.append("--")
            else:
                cells.append(pm(mm[ref][key], 1e4, 1, plus=False))
    rows.append(" & ".join(cells) + r" \\")
tab = [r"\begin{tabular}{llcccccc}", r"\toprule",
       r" & & & Feb & \multicolumn{2}{c}{Feb $\Delta$ log-loss ($10^{-4}$)} & \multicolumn{2}{c}{Mar $\Delta$ log-loss ($10^{-4}$)} \\",
       r"\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
       r"Model & Params & CV (Jan) & log-loss & vs.\ Elo & vs.\ baseline & vs.\ Elo & vs.\ baseline \\", r"\midrule"] + rows + [r"\bottomrule", r"\end{tabular}"]
open(f"{P}/table_models.tex", "w").write("\n".join(tab) + "\n")

# appendix: Brier, calibration error, CV paired differences, hyper-parameters
rows = []
base_f = np.array(cv[BASE]["fold_loss"])
for key, label, par in MODELS:
    d = np.array(cv[key]["fold_loss"]) - base_f
    f, m_ = tb("2015-02"), tb("2015-03")
    calib = lambda mo: "--" if sc["months"][mo]["metrics"][key]["calib"] is None else f"{sc['months'][mo]['metrics'][key]['calib']:.4f}"
    rows.append(" & ".join([label, par,
                            "--" if key == BASE else f"{mn(1e4 * d.mean(), 2)} $\\pm$ {1e4 * d.std(ddof=1) / np.sqrt(len(d)):.2f}",
                            f"{f['brier'][key][0]:.4f} $\\pm$ {hw(f['brier'][key]):.4f}", calib("2015-02"),
                            f"{m_['loss'][key][0]:.4f} $\\pm$ {hw(m_['loss'][key]):.4f}",
                            f"{m_['brier'][key][0]:.4f} $\\pm$ {hw(m_['brier'][key]):.4f}", calib("2015-03")]) + r" \\")
tab = [r"\begin{tabular}{llcccccc}", r"\toprule",
       r" & & CV paired diff. & \multicolumn{2}{c}{Feb} & \multicolumn{3}{c}{Mar} \\", r"\cmidrule(lr){4-5}\cmidrule(lr){6-8}",
       r"Model & Params & vs.\ baseline ($10^{-4}$) & Brier & Calib. & log-loss & Brier & Calib. \\", r"\midrule"] + rows + [r"\bottomrule", r"\end{tabular}"]
open(f"{P}/table_models_app.tex", "w").write("\n".join(tab) + "\n")

# model-table numbers
fe, ma = tb("2015-02"), tb("2015-03")
fe_d = tb("2015-02", "day")
for key, nm in (("elo", "Elo"), ("floor", "Floor"), (BASE, "Base"), ("pooled", "Pooled"), ("rating", "Rating"), ("spline", "Spline"), ("boost", "Boost"), ("ordinal", "Ordinal")):
    put(f"ll{nm}Feb", f"{fe['loss'][key][0]:.4f}")
    put(f"ll{nm}Mar", f"{ma['loss'][key][0]:.4f}")
for tag, mm, sch in (("Feb", "2015-02", "player"), ("Mar", "2015-03", "player"), ("MarLabels", "2015-03L", "player"),
                     ("FebDay", "2015-02", "day"), ("MarDay", "2015-03", "day")):
    for ref, rn in (("d_elo", "Elo"), ("d_base", "Base")):
        for key, nm in (("pooled", "Pooled"), (BASE, "Base"), ("rating", "Rating"), ("spline", "Spline"), ("boost", "Boost"), ("ordinal", "Ordinal"), ("floor", "Floor")):
            a = bt[mm][sch][ref][key]
            put(f"d{rn}{nm}{tag}", f"{mn(a[0] * 1e4, 1)}")
            put(f"d{rn}{nm}{tag}Lo", f"{mn(a[1] * 1e4, 1)}")
            put(f"d{rn}{nm}{tag}Hi", f"{mn(a[2] * 1e4, 1)}")
            put(f"d{rn}{nm}{tag}Abs", f"{abs(a[0]) * 1e4:.1f}")
            put(f"d{rn}{nm}{tag}Pm", f"{hw(a) * 1e4:.1f}")
put("calibEloFeb", f"{sc['months']['2015-02']['metrics']['elo']['calib']:.4f}")
put("calibBaseFeb", f"{sc['months']['2015-02']['metrics'][BASE]['calib']:.4f}")
put("floorGap", f"{(fe['loss']['floor'][0] - fe['loss']['elo'][0]) * 1e4:.0f}")
put("gainRatioSpline", f"{abs(fe['d_elo'][BASE][0] / fe['d_base']['spline'][0]):.0f}")
put("gainRatioBoost", f"{abs(fe['d_elo'][BASE][0] / fe['d_base']['boost'][0]):.0f}")
put("bestKnots", K)
put("bestLam", lamtex(fz["rating"]["lam"]))
put("splineLam", lamtex(fz["spline"]["lam"]))
put("boostTrees", fz["boost"]["trees"])
put("boostLeaves", fz["boost"]["leaves"])
put("boostLr", f"{fz['boost']['lr']:g}")
put("selectionHash", fz["sha256"][:12])
put("selectionTime", fz["frozen_utc"][:10])
d = np.array(cv["pooled"]["fold_loss"]) - base_f
put("cvPooledDiff", f"{1e4 * d.mean():.2f}")
put("cvPooledSE", f"{1e4 * d.std(ddof=1) / np.sqrt(5):.2f}")
cvn = [cv[BASE]["fold_loss"], cv["elo"]["fold_loss"]]
put("naiveSE", f"{np.std(cv[BASE]['fold_loss'], ddof=1) / np.sqrt(5):.4f}")
for key, nm in (("spline", "Spline"), ("boost", "Boost")):
    d = np.array(cv[key]["fold_loss"]) - base_f
    put(f"cv{nm}Diff", f"{abs(1e4 * d.mean()):.1f}")
    put(f"cv{nm}SE", f"{1e4 * d.std(ddof=1) / np.sqrt(5):.1f}")
# declared criterion 2: ordering of mean log-loss in March is the same as in February
order = lambda mm: [k for k, _ in sorted(((k, bt[mm]["player"]["loss"][k][0]) for k in ("elo", BASE, "spline", "boost")), key=lambda t: -t[1])]
put("orderSame", "the same" if order("2015-02") == order("2015-03") else "a different")


# where the flexible models gain (February)
zf = np.load(f"{R}/test_2015-02.npz")
dd = zf["d"]
big = np.abs(dd) > 400
Lb = zf["L_" + BASE].astype(float)
put("bigGapShare", f"{100 * big.mean():.0f}")
for key, nm in (("spline", "Spline"), ("boost", "Boost")):
    gn = Lb - zf["L_" + key].astype(float)
    put(f"gainBig{nm}", f"{100 * gn[big].sum() / gn.sum():.0f}")
    put(f"gainPct{nm}", f"{100 * abs(fe['d_base'][key][0] / fe['d_elo'][BASE][0]):.0f}")

# ---------------------------------------------------------------- advantage table
GROUPS = ["Bullet", "Blitz", "Classical", "All"]
gname = {"Bullet": "Bullet", "Blitz": "Blitz", "Classical": "Classical", "All": "All games"}
def g(month, scheme, grp, stat, which="ci"):
    return bc[month][scheme]["groups"][grp][stat][which]
def trip(month, scheme, grp, stat):
    x = bc[month][scheme]["groups"][grp][stat]
    return [x["point"], x["ci"][0], x["ci"][1]]
rows = []
for grp in GROUPS:
    rows.append(" & ".join([gname[grp], f"{bc['2015-01']['player']['n'][grp]:,}",
                            pm(trip("2015-01", "player", grp, "edge_pp"), 100, 2),
                            pm(trip("2015-01", "player", grp, "edge_pts"), 1, 1),
                            pm(trip("2015-01", "player", grp, "slope_ratio"), 1, 3),
                            pm(trip("2015-02", "player", grp, "slope_ratio"), 1, 3),
                            pm(trip("2015-03", "player", grp, "slope_ratio"), 1, 3)]) + r" \\")
tab = [r"\begin{tabular}{lrcccccc}"[:-1] + "}", r"\toprule",
       r" & Games & \multicolumn{2}{c}{White's edge (Jan)} & \multicolumn{3}{c}{Slope / Elo slope} \\", r"\cmidrule(lr){3-4}\cmidrule(lr){5-7}",
       r"Time control & (Jan) & score pts (pp) & rating pts & Jan & Feb & Mar \\", r"\midrule"] + rows + [r"\bottomrule", r"\end{tabular}"]
tab[0] = r"\begin{tabular}{lrcccrr}".replace("crr", "ccc").replace("lrcccccc", "lrccccc")
tab[0] = r"\begin{tabular}{lrccccc}"
open(f"{P}/table_advantage.tex", "w").write("\n".join(tab) + "\n")

for m, key in MON.items():
    for grp in GROUPS:
        for stat, nm, sc_, nd in (("slope_ratio", "Slope", 1, 3), ("edge_pp", "EdgePP", 100, 2), ("edge_pts", "EdgePts", 1, 1)):
            for sch, tag in (("player", ""), ("day", "Day")):
                t = trip(m, sch, grp, stat)
                put(f"{nm}{grp}{key}{tag}", f"{t[0] * sc_:.{nd}f}")
                put(f"{nm}{grp}{key}{tag}Lo", f"{t[1] * sc_:.{nd}f}")
                put(f"{nm}{grp}{key}{tag}Hi", f"{t[2] * sc_:.{nd}f}")
for name, nm in (("Blitz - Bullet", "BlitzBullet"), ("Classical - Bullet", "ClassBullet"), ("Classical - Blitz", "ClassBlitz")):
    for sch, tag in (("player", ""), ("day", "Day")):
        c = bc["2015-01"][sch]["contrasts"][name + "|edge_pp"]
        put(f"con{nm}{tag}", mn(c["point"] * 100, 2, True))
        put(f"con{nm}{tag}Lo", mn(c["ci"][0] * 100, 2))
        put(f"con{nm}{tag}Hi", mn(c["ci"][1] * 100, 2))
put("slopeMonthDiff", f"{abs(trip('2015-03', 'player', 'All', 'slope_ratio')[0] - trip('2015-01', 'player', 'All', 'slope_ratio')[0]):.3f}")
put("slopeCritPass", "met" if abs(trip('2015-03', 'player', 'All', 'slope_ratio')[0] - trip('2015-01', 'player', 'All', 'slope_ratio')[0]) <= 0.03 else "failed")
put("critOnePass", "met" if bt["2015-03"]["player"]["d_elo"][BASE][2] < 0 else "failed")
put("critTwoPass", "met" if macros["orderSame"] == "the same" else "failed")

# worked example
a_, b_ = None, None
t_pts, t_sl = trip("2015-01", "player", "All", "edge_pts")[0], trip("2015-01", "player", "All", "slope_ratio")[0]
b_ = LN10 * t_sl
a_ = t_pts * b_ / 400
fit_p = lambda dd: 1 / (1 + math.exp(-(a_ + b_ * dd / 400)))
put("eloPlus", f"{elo_p(200):.2f}"); put("fitPlus", f"{fit_p(200):.2f}")
put("eloMinus", f"{elo_p(-200):.2f}"); put("fitMinus", f"{fit_p(-200):.2f}")
put("maxResid", f"{f1['max_abs_resid_outside_small_gaps']:.2f}")

# ---------------------------------------------------------------- bands and robustness (appendix)
BAND = [("m<1300", r"$<$1300", "BandA"), ("1300-1500", "1300--1500", "BandB"), ("1500-1700", "1500--1700", "BandC"),
        ("1700-1900", "1700--1900", "BandD"), ("m>=1900", r"$\ge$1900", "BandE")]
rows = []
for grp, lab, nm in BAND:
    rows.append(" & ".join([lab, f"{bc['2015-01']['player']['n'][grp]:,}", pm(trip("2015-01", "player", grp, "edge_pp"), 100, 2),
                            pm(trip("2015-01", "player", grp, "edge_pts"), 1, 1), pm(trip("2015-01", "player", grp, "slope_ratio"), 1, 3)]) + r" \\")
    for stat, sn, sc_, nd in (("slope_ratio", "Slope", 1, 3), ("edge_pp", "EdgePP", 100, 2)):
        t = trip("2015-01", "player", grp, stat)
        put(f"{sn}{nm}", f"{t[0] * sc_:.{nd}f}")
ROB = [("All", "All games (reference)"), ("no_infraction", "Drop rules-infraction games"), ("no_rating_1500", "Drop games with a rating of exactly 1500"),
       ("no_time_forfeit", "Drop time-forfeit games"), ("days_1_15", "January 1--15 only"), ("days_16_31", "January 16--31 only")]
rows.append(r"\midrule")
for grp, lab in ROB:
    rows.append(" & ".join([lab, f"{bc['2015-01']['player']['n'][grp]:,}", pm(trip("2015-01", "player", grp, "edge_pp"), 100, 2),
                            pm(trip("2015-01", "player", grp, "edge_pts"), 1, 1), pm(trip("2015-01", "player", grp, "slope_ratio"), 1, 3)]) + r" \\")
rn = {"no_infraction": "NoInf", "no_rating_1500": "NoFifteen", "no_time_forfeit": "NoForfeit", "days_1_15": "HalfA", "days_16_31": "HalfB"}
for grp, nm in rn.items():
    for stat, sn, sc_, nd in (("slope_ratio", "Slope", 1, 3), ("edge_pp", "EdgePP", 100, 2), ("edge_pts", "EdgePts", 1, 1)):
        t = trip("2015-01", "player", grp, stat)
        put(f"{sn}{nm}", f"{t[0] * sc_:.{nd}f}")
tab = [r"\begin{tabular}{lrccc}", r"\toprule", r"Subset of January & Games & White's edge (pp) & White's edge (rating pts) & Slope / Elo slope \\", r"\midrule"] + rows + [r"\bottomrule", r"\end{tabular}"]
open(f"{P}/table_bands.tex", "w").write("\n".join(tab) + "\n")
t_top, t_bot = bc["2015-01"]["player"]["contrasts"]["top band - bottom band|slope_ratio"], None
put("bandContrast", f"{t_top['point']:.3f}"); put("bandContrastLo", f"{t_top['ci'][0]:.3f}"); put("bandContrastHi", f"{t_top['ci'][1]:.3f}")
t_e = bc["2015-01"]["player"]["contrasts"]["top band - bottom band (edge)|edge_pp"]
put("bandEdgeContrast", mn(t_e["point"] * 100, 2, True)); put("bandEdgeLo", mn(t_e["ci"][0] * 100, 2)); put("bandEdgeHi", mn(t_e["ci"][1] * 100, 2))
edges = [bc["2015-01"]["player"]["groups"][g_]["edge_pp"]["point"] * 100 for g_, _, _ in BAND]
put("bandEdgeMin", f"{min(edges):.1f}"); put("bandEdgeMax", f"{max(edges):.1f}")

# ---------------------------------------------------------------- noise table
bj = lambda grp: trip("2015-01", "player", grp, "slope_ratio")
def nrow(label, share, t):
    return f"{label} & {share} & {mn(t[0], 3)} {ci(t[1], t[2], 1, 3)} \\\\"
rows = [nrow("All games", "100\\%", bj("All"))]
for grp, lab in (("games>=10", r"Both players $\ge$10 games in January"), ("games>=100", r"Both players $\ge$100 games")):
    rows.append(nrow(lab, f"{100 * bc['2015-01']['player']['share'][grp]:.0f}\\%", bj(grp)))
rows.append(r"\midrule")
hshare = f"{100 * sm['share_of_games']:.0f}\\%"
for k, lab in (("displayed", "Displayed gap alone"), ("smoothed", "Smoothed (earlier-day) gap alone"), ("both_sum", "Both together: sum of the two slopes")):
    t = [sm["point"][k]] + sm["ci"]["player"][k]
    rows.append(nrow(r"$\ge$20 earlier-day games each: " + lab if k == "displayed" else lab, hshare if k == "displayed" else "", t))
rows.append(r"\midrule")
for grp, lab in (("hist20_T1", "Most settled third (smallest past rating changes)"), ("hist20_T3", "Least settled third")):
    rows.append(nrow(lab, f"{100 * bc['2015-01']['player']['share'][grp]:.0f}\\%", bj(grp)))
rows.append(r"\midrule")
tab = [r"\begin{tabular}{lcc}", r"\toprule", r"Check (all on January) & Share of games & Slope / Elo slope \\", r"\midrule"] + rows[:-1] + [r"\bottomrule", r"\end{tabular}"]
open(f"{P}/table_noise.tex", "w").write("\n".join(tab) + "\n")
for k, nm in (("displayed", "Disp"), ("smoothed", "Smooth"), ("both_displayed", "BothD"), ("both_smoothed", "BothS"), ("both_sum", "BothSum")):
    put(f"slope{nm}", f"{sm['point'][k]:.2f}")
    put(f"slope{nm}Lo", f"{sm['ci']['player'][k][0]:.2f}"); put(f"slope{nm}Hi", f"{sm['ci']['player'][k][1]:.2f}")
put("histShare", f"{100 * sm['share_of_games']:.0f}")
for grp, nm in (("games>=10", "FreqTen"), ("games>=30", "FreqThirty"), ("games>=100", "FreqHundred"), ("hist20_T1", "TercA"), ("hist20_T2", "TercB"), ("hist20_T3", "TercC")):
    t = bj(grp)
    put(f"slope{nm}", f"{t[0]:.2f}"); put(f"slope{nm}Lo", f"{t[1]:.2f}"); put(f"slope{nm}Hi", f"{t[2]:.2f}")
    put(f"share{nm}", f"{100 * bc['2015-01']['player']['share'][grp]:.0f}")
c = bc["2015-01"]["player"]["contrasts"]["least - most noisy (history)|slope_ratio"]
put("tercContrast", f"{c['point']:.2f}"); put("tercContrastLo", f"{c['ci'][0]:.2f}"); put("tercContrastHi", f"{c['ci'][1]:.2f}")

# simulation
for kind, nm in (("classical", "Class"), ("posterior", "Post"), ("berkson", "Berk")):
    put(f"sigma{nm}", f"{ss[kind]['sigma_star']:.0f}")
    put(f"sigma{nm}Lo", f"{ss[kind]['sigma_lo']:.0f}"); put(f"sigma{nm}Hi", f"{ss[kind]['sigma_hi']:.0f}")
sg = sim["sigmas"]
def sim_at(kind, s, stat):
    return float(np.mean(sim["kinds"][kind][sg.index(s)][stat]))
put("simTrueEdgePts", f"{sim['true_edge_pts']:.1f}")
put("simTruePP", f"{100 * (elo_p(sim['true_edge_pts']) - 0.5):.2f}")
for kind, nm in (("classical", "Class"), ("berkson", "Berk")):
    for s_, w_ in ((50, "Fifty"), (60, "Sixty"), (110, "OneTen")):
        put(f"sim{nm}Slope{w_}", f"{sim_at(kind, s_, 'slope_ratio'):.2f}")
put("simClassEdgePtsFifty", f"{sim_at('classical', 50, 'edge_pts'):.1f}")
put("simClassEdgePPFifty", f"{100 * sim_at('classical', 50, 'edge_pp'):.2f}")
put("simClassEdgePPErr", f"{100 * (sim_at('classical', 50, 'edge_pp') / (elo_p(sim['true_edge_pts']) - 0.5) - 1):.0f}")
put("simBerkEdgePPHigh", f"{100 * sim_at('berkson', 150, 'edge_pp'):.2f}")
put("gapVar", f"{sim['var_d']:.0f}")
put("gapSD", f"{math.sqrt(sim['var_d']):.0f}")
rows = []
for s_ in (0, 30, 50, 56 if 56 in sg else 55, 60, 70, 110, 150):
    if s_ not in sg:
        continue
    cells = [str(s_)]
    for kind in ("classical", "posterior", "berkson"):
        r_ = sim["kinds"][kind][sg.index(s_)]
        cells += ["--", "--", "--"] if r_ is None else [f"{np.mean(r_['slope_ratio']):.3f}", f"{100 * np.mean(r_['edge_pp']):.2f}", f"{np.mean(r_['edge_pts']):.1f}"]
    rows.append(" & ".join(cells) + r" \\")
tab = [r"\begin{tabular}{lccccccccc}", r"\toprule",
       r" & \multicolumn{3}{c}{Classical (headline)} & \multicolumn{3}{c}{Classical, exact gap} & \multicolumn{3}{c}{Berkson} \\", r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
       r"$\sigma$ & slope & edge pp & edge pts & slope & edge pp & edge pts & slope & edge pp & edge pts \\", r"\midrule"] + rows + [r"\bottomrule", r"\end{tabular}"]
open(f"{P}/table_sim.tex", "w").write("\n".join(tab) + "\n")

# ---------------------------------------------------------------- bootstrap-scheme comparison
def hwt(mm, ref, key, sch):
    a = bt[mm][sch][ref][key]; return hw(a)
rowsb = []
items = [("Baseline $-$ Elo, Feb log-loss ($10^{-4}$)", lambda s: hwt("2015-02", "d_elo", BASE, s) * 1e4, 1),
         ("Baseline $-$ Elo, Mar log-loss ($10^{-4}$)", lambda s: hwt("2015-03", "d_elo", BASE, s) * 1e4, 1),
         ("Boosted $-$ baseline, Feb ($10^{-4}$)", lambda s: hwt("2015-02", "d_base", "boost", s) * 1e4, 2),
         ("Slope / Elo slope, Jan ($10^{-3}$)", lambda s: (bc["2015-01"][s]["groups"]["All"]["slope_ratio"]["ci"][1] - bc["2015-01"][s]["groups"]["All"]["slope_ratio"]["ci"][0]) / 2 * 1e3, 1),
         ("White's edge, Jan (pp)", lambda s: (bc["2015-01"][s]["groups"]["All"]["edge_pp"]["ci"][1] - bc["2015-01"][s]["groups"]["All"]["edge_pp"]["ci"][0]) / 2 * 100, 2)]
widths = {}
for lab, f_, nd in items:
    v = {s: f_(s) for s in ("game", "day", "player")}
    widths[lab] = v
    rowsb.append(f"{lab} & {v['game']:.{nd}f} & {v['day']:.{nd}f} & {v['player']:.{nd}f} & {v['player'] / v['day']:.1f} \\\\")
tab = [r"\begin{tabular}{lcccc}", r"\toprule", r"Quantity (half-width of 95\% interval) & Game & Day & Player & Player / day \\", r"\midrule"] + rowsb + [r"\bottomrule", r"\end{tabular}"]
open(f"{P}/table_boot.tex", "w").write("\n".join(tab) + "\n")
ratios = [v["player"] / v["day"] for v in widths.values()]
put("ratioMin", f"{min(ratios):.1f}"); put("ratioMax", f"{max(ratios):.1f}")
put("ratioDelta", f"{ratios[0]:.1f}"); put("ratioSlope", f"{ratios[3]:.1f}"); put("ratioEdge", f"{ratios[4]:.1f}")
gr = [v["day"] / v["game"] for v in widths.values()]
put("dayOverGameMin", f"{min(gr):.1f}"); put("dayOverGameMax", f"{max(gr):.1f}")

# ---------------------------------------------------------------- cleaning table (appendix)
def stp(m):
    return sc["train_steps"] if m == "2015-01" else sc["months"][m]["steps"]
srows = [("Games in the file", "raw"), ("Correspondence (not bullet, blitz or classical)", "not_bullet_blitz_classical"),
         ("Abandoned", "abandoned"), ("Missing rating", "missing_rating"), ("Dated outside the month", "outside_month"), ("Games kept", "final")]
rows = []
for lab, k in srows:
    rows.append(f"{lab} & " + " & ".join(f"{stp(m)[k]:,}" for m in MON) + r" \\")
    if k == "outside_month":
        rows.append(r"\midrule")
jan_sum = None
for ci_, cn in enumerate(["Bullet", "Blitz", "Classical"]):
    rows.append(f"{cn} games kept & " + " & ".join(
        f"{(sc['months'][m]['summary'] if m != '2015-01' else J('jan_summary.json'))['by_class'][cn]:,}" for m in MON) + r" \\")
tab = [r"\begin{tabular}{lrrr}", r"\toprule", r" & Jan 2015 & Feb 2015 & Mar 2015 \\", r"\midrule"] + rows + [r"\bottomrule", r"\end{tabular}"]
open(f"{P}/table_cleaning.tex", "w").write("\n".join(tab) + "\n")
js = J("jan_summary.json")
put("nTrainRaw", f"{sc['train_steps']['raw']:,}"); put("nTestRaw", f"{stp('2015-02')['raw']:,}")
put("nCorrJan", f"{sc['train_steps']['not_bullet_blitz_classical']:,}"); put("nAbandJan", f"{sc['train_steps']['abandoned']:,}"); put("nOutsideJan", f"{sc['train_steps']['outside_month']:,}")
put("nInfrJan", js["n_infraction"]); put("nTFJan", f"{js['n_time_forfeit']:,}")
put("drawRate", f"{100 * js['draw_rate']:.1f}"); put("whiteScore", f"{js['white_score']:.3f}")
put("medianGap", f"{js['abs_gap_median']:.0f}"); put("ratingMean", f"{js['rating_mean']:.0f}"); put("ratingSD", f"{js['m_sd']:.0f}")
put("nPlayersJan", f"{js['n_players']:,}")
put("nStandardMar", f"{J('mar_standard.json')['n_standard']:,}")
put("nMarLabels", f"{sc['months']['2015-03L']['summary']['n']:,}")

# ---------------------------------------------------------------- scalar macros for the summary / abstract
b2 = lambda s: (bc["2015-01"][s]["groups"]["All"]["slope_ratio"]["ci"])
with open(f"{P}/numbers.tex", "w") as f:
    for k, v in macros.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(len(macros), "macros")
