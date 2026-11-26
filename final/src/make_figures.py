"""Figure 1 (calibration by rating gap) and Figure 2 (noise simulation).

Usage: python -I src/make_figures.py results/ data/headers_2015-01.csv paper/
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from lib import *  # noqa: E402,F401,F403
from boot_test import weights  # noqa: E402

# categorical slots 1-3 (validated all-pairs for colour-blind safety)
STYLE = {"Bullet": ("#2a78d6", "o", -8), "Blitz": ("#eb6834", "s", 0), "Classical": ("#1baf7a", "^", 8)}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
B_FIG = 500


def style():
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7.5,
                         "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "font.family": "serif"})


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.grid(axis="y", color=GRID, lw=0.6)


def fig1(out_dir, jan_path, paper_dir):
    z = np.load(f"{out_dir}/test_2015-02.npz")
    d, S, tc = z["d"], z["S"], z["tc"]
    r = S - elo_p(d)
    inb = (d >= -600) & (d < 600)
    comp = (tc * 24 + np.floor((d + 600) / 50).astype(int))[inb]
    r_b = r[inb]
    zz = {k: z[k][inb] for k in ("S", "day", "wid", "bid")}
    n = np.bincount(comp, minlength=72)
    point = np.bincount(comp, weights=r_b, minlength=72) / np.maximum(n, 1)
    rng = np.random.default_rng([SEED, 21])
    bs = np.empty((B_FIG, 72))
    for b in range(B_FIG):
        g = weights("player", zz, rng)
        bs[b] = np.bincount(comp, weights=g * r_b, minlength=72) / np.maximum(np.bincount(comp, weights=g, minlength=72), 1e-9)
    lo, hi = np.percentile(bs, 2.5, axis=0), np.percentile(bs, 97.5, axis=0)

    tr, _ = load_month(jan_path, "2015-01")
    wins = np.bincount(tr.d.values + DOFF, weights=tr.S.values, minlength=NBIN)
    tot = np.bincount(tr.d.values + DOFF, minlength=NBIN).astype(float)
    a, b = agg_fit_ab(wins, tot)

    style()
    fig, ax = plt.subplots(figsize=(5.5, 2.45))
    x = np.linspace(-600, 600, 400)
    ax.axhline(0, color=MUTED, lw=1, ls="--", zorder=1)
    ax.text(590, 0.004, "Elo formula", ha="right", va="bottom", color=MUTED, fontsize=7.5)
    ax.plot(x, sigmoid(a + b * x / 400) - elo_p(x), color=INK, lw=1.2, zorder=2,
            label="Logistic baseline (fit on Jan)")
    mids = -575 + 50 * np.arange(24)
    for k, (c, (col, mk, dx)) in enumerate(STYLE.items()):
        i = np.arange(24) + 24 * k
        use = n[i] >= 200
        ax.errorbar(mids[use] + dx, point[i][use], yerr=[point[i][use] - lo[i][use], hi[i][use] - point[i][use]],
                    fmt=mk, ms=4, color=col, ecolor=col, elinewidth=1, capsize=0, mec="white", mew=0.6,
                    label=c, zorder=3)
    ax.set_xlim(-600, 600)
    ax.set_xlabel("Rating gap d = WhiteElo $-$ BlackElo (points)")
    ax.set_ylabel("Observed $-$ Elo expected score")
    clean(ax)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, handletextpad=0.4,
              columnspacing=1.2)
    fig.tight_layout(pad=0.3)
    fig.savefig(f"{paper_dir}/fig_calibration.pdf")
    fig.savefig(f"{paper_dir}/fig_calibration.png", dpi=200)
    json.dump({"max_abs_resid_outside_small_gaps": float(np.nanmax(np.abs(point[np.tile(np.abs(mids) > 100, 3) & (n >= 200)]))),
               "n_bins_plotted": int((n >= 200).sum())}, open(f"{out_dir}/fig1_info.json", "w"))


def crossing(x, m, target):
    j = int(np.argmax(m < target))
    return float(np.interp(target, [m[j], m[j - 1]], [x[j], x[j - 1]]))


def fig2(out_dir, paper_dir):
    sim = json.load(open(f"{out_dir}/noise_sim.json"))
    obs = json.load(open(f"{out_dir}/boot_coef_2015-01_player.json"))["groups"]["All"]["slope_ratio"]
    sg = np.array(sim["sigmas"], float)
    style()
    fig, ax = plt.subplots(figsize=(5.5, 2.35))
    ax.axhspan(obs["ci"][0], obs["ci"][1], color="#9b9a96", alpha=0.35, lw=0)
    ax.axhline(obs["point"], color=MUTED, lw=1, ls="--")
    ax.text(198, obs["point"] + 0.012, "observed 0.78", ha="right", va="bottom", color=MUTED, fontsize=7.5)
    ax.axvspan(60, 110, color="#f3e6c8", alpha=0.6, lw=0)
    ax.text(85, 0.415, "typical RD\n(2026 code)", ha="center", va="bottom", color=MUTED, fontsize=7)
    kinds = [("classical", "Classical error (headline)", "#2a78d6", "-", "o"),
             ("posterior", "Classical, exact gap", "#eb6834", "--", "s"),
             ("berkson", "Berkson error", "#1baf7a", ":", "^")]
    out = {}
    for kind, lab, col, ls, mk in kinds:
        rows = sim["kinds"][kind]
        ok = [i for i, r in enumerate(rows) if r]
        x = sg[ok]
        m = np.array([np.mean(rows[i]["slope_ratio"]) for i in ok])
        lo = np.array([np.min(rows[i]["slope_ratio"]) for i in ok])
        hi = np.array([np.max(rows[i]["slope_ratio"]) for i in ok])
        ax.fill_between(x, lo, hi, color=col, alpha=0.3, lw=0)
        ax.plot(x, m, color=col, ls=ls, lw=1.3, label=lab, zorder=3)
        ax.plot(x[::2], m[::2], mk, color=col, ms=3.5, mec="white", mew=0.5, zorder=4)
        out[kind] = {"sigma_star": crossing(x, m, obs["point"]),
                     "sigma_lo": crossing(x, m, obs["ci"][1]), "sigma_hi": crossing(x, m, obs["ci"][0])}
        ax.plot([out[kind]["sigma_star"]], [obs["point"]], "o", color="white", mec=col, mew=1.4, ms=6, zorder=5)
    ax.set_xlim(0, 200)
    ax.set_ylim(0.4, 1.03)
    ax.set_xlabel("Per-player rating noise $\\sigma$ (rating points)")
    ax.set_ylabel("Fitted slope / Elo slope")
    clean(ax)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=7.5,
              handletextpad=0.4, columnspacing=1.2)
    fig.tight_layout(pad=0.3)
    fig.savefig(f"{paper_dir}/fig_noise.pdf")
    fig.savefig(f"{paper_dir}/fig_noise.png", dpi=200)
    json.dump(out, open(f"{out_dir}/sigma_star.json", "w"), indent=1)
    print(json.dumps(out))


if __name__ == "__main__":
    out_dir, jan_path, paper_dir = sys.argv[1:4]
    fig2(out_dir, paper_dir) if "--fig2" in sys.argv else (fig1(out_dir, jan_path, paper_dir), fig2(out_dir, paper_dir))
