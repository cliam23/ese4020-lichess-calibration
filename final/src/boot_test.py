"""Bootstrap intervals for test-month log-loss, Brier score and paired differences.

Three resampling schemes, all on the same per-game losses:
  game    each game gets an independent Poisson(1) weight
  day     whole UTC days are resampled with replacement (the draft's scheme)
  player  pigeonhole bootstrap over players (Owen 2007): every player gets an
          independent Poisson(1) weight and a game's weight is the product of its
          two players' weights, so players who span days and opponents are
          resampled as units.

Usage: python -I src/boot_test.py results/ 2015-02 [2015-03] [--B 2000]
"""

import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from lib import SEED  # noqa: E402
from score import NAMES  # noqa: E402


def weights(scheme, z, rng):
    n = len(z["S"])
    if scheme == "game":
        return rng.poisson(1.0, n).astype(np.float64)
    if scheme == "day":
        day = z["day"]
        days = np.unique(day)
        c = np.bincount(rng.integers(0, len(days), len(days)), minlength=len(days)).astype(float)
        return c[np.searchsorted(days, day)]
    w = rng.poisson(1.0, int(max(z["wid"].max(), z["bid"].max())) + 1).astype(np.float64)
    return w[z["wid"]] * w[z["bid"]]


def ci(a):
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def main(out_dir, months, B):
    fz = json.load(open(f"{out_dir}/frozen_selection.json"))
    base = fz["baseline"]
    res = {}
    for month in months:
        z = np.load(f"{out_dir}/test_{month}.npz")
        M = np.column_stack([z["L_" + n] for n in NAMES] + [z["B_" + n] for n in NAMES]).astype(np.float64)
        pt = M.mean(0)
        k = len(NAMES)
        res[month] = {}
        for si, scheme in enumerate(["game", "day", "player"]):
            rng = np.random.default_rng([SEED, si, int(month[5:7]) + (100 if month.endswith('L') else 0)])
            st = np.empty((B, 2 * k))
            for b in range(B):
                g = weights(scheme, z, rng)
                st[b] = (g @ M) / g.sum()
            L, Br = st[:, :k], st[:, k:]
            idx = {n: i for i, n in enumerate(NAMES)}
            out = {"loss": {}, "brier": {}, "d_elo": {}, "d_base": {}}
            for n, i in idx.items():
                out["loss"][n] = [float(pt[i])] + ci(L[:, i])
                out["brier"][n] = [float(pt[k + i])] + ci(Br[:, i])
                out["d_elo"][n] = [float(pt[i] - pt[idx["elo"]])] + ci(L[:, i] - L[:, idx["elo"]])
                out["d_base"][n] = [float(pt[i] - pt[idx[base]])] + ci(L[:, i] - L[:, idx[base]])
            # the baseline's paired difference to the pooled fit and the SE of that difference
            res[month][scheme] = out
            print(month, scheme, "base-elo", out["d_elo"][base], flush=True)
    json.dump(res, open(f"{out_dir}/boot_test_{'_'.join(months)}.json", "w"), indent=1)


if __name__ == "__main__":
    args = [a for a in sys.argv[2:] if not a.startswith("--")]
    B = int(sys.argv[sys.argv.index("--B") + 1]) if "--B" in sys.argv else 2000
    args = [a for a in args if a != str(B)]
    main(sys.argv[1], args, B)
