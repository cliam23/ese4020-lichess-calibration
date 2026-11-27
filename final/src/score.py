"""Refit every model on all of January with the frozen settings and score test months.

Refuses to run unless results/frozen_selection.json verifies against its hash.
Writes results/test_<month>.npz (per-game outcomes, predictions, log-loss and
Brier score for each model) and results/scores.json (point metrics).

Usage: python -I src/score.py results/ data/headers_2015-01.csv 2015-02:data/headers_2015-02.csv [2015-03:...]
"""

import hashlib
import json
import sys
import time

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from lib import *  # noqa: E402,F401,F403

NAMES = ["floor", "elo", "pooled", "bytc", "rating", "spline", "boost", "ordinal"]


def verify(out_dir):
    fz = json.load(open(f"{out_dir}/frozen_selection.json"))
    body = json.dumps({k: fz[k] for k in ("baseline", "rating", "spline", "boost", "chosen_names")},
                      sort_keys=True)
    if hashlib.sha256(body.encode()).hexdigest() != fz["sha256"]:
        sys.exit("frozen_selection.json does not match its hash; refusing to score")
    return fz


def main(out_dir, jan_path, months):
    fz = verify(out_dir)
    tr, steps = load_month(jan_path, "2015-01")
    t0 = time.time()
    K, lam_s = fz["spline"]["K"], fz["spline"]["lam"]
    knots = spline_knots(tr, K)
    fits = {
        "pooled": fit_logit(design("pooled", tr), tr.S.values),
        "bytc": fit_logit(design("bytc", tr), tr.S.values),
        "rating": fit_logit(design("rating", tr), tr.S.values, lam=fz["rating"]["lam"]),
        "spline": fit_logit(design("spline", tr, knots), tr.S.values, lam=lam_s),
    }
    boost = fit_hgb(tr, fz["boost"]["trees"], fz["boost"]["leaves"], fz["boost"]["lr"])
    ordinal = fit_ordinal(tr)
    floor = float(tr.S.mean())
    print(f"fitted in {time.time() - t0:.0f}s", flush=True)

    def predict(df):
        return {
            "floor": np.full(len(df), floor),
            "elo": elo_p(df.d.values),
            "pooled": predict_logit(fits["pooled"], design("pooled", df)),
            "bytc": predict_logit(fits["bytc"], design("bytc", df)),
            "rating": predict_logit(fits["rating"], design("rating", df)),
            "spline": predict_logit(fits["spline"], design("spline", df, knots)),
            "boost": predict_hgb(boost, df),
            "ordinal": predict_ordinal(ordinal, df),
        }

    scores = {"frozen": fz, "train_steps": steps, "ordinal_params": {CLASSES[k]: v for k, v in ordinal.items()},
              "n_params": {"floor": 1, "elo": 0, "pooled": 2, "bytc": 6, "rating": 9,
                           "spline": 9 + K - 2, "boost": None, "ordinal": 9}, "months": {}}
    for spec in months:
        month, path = spec.split(":", 1)
        df, st = load_month(path, month)
        jobs = [(month, df)]
        if df["std"].any():  # sensitivity: March games labelled "Classical" only, "Standard" dropped
            jobs.append((month + "L", df[~df["std"]].reset_index(drop=True)))
        for key, d_ in jobs:
            P = predict(d_)
            S = d_.S.values
            arrs = {"S": S, "day": d_.day.values, "wid": d_.wid.values, "bid": d_.bid.values,
                    "tc": d_.tc.values, "d": d_.d.values, "m": d_.m.values}
            met = {}
            for n in NAMES:
                arrs["P_" + n] = P[n].astype(np.float32)
                arrs["L_" + n] = game_loss(S, P[n]).astype(np.float32)
                arrs["B_" + n] = game_brier(S, P[n]).astype(np.float32)
                met[n] = {"logloss": float(arrs["L_" + n].mean(dtype=np.float64)),
                          "brier": float(arrs["B_" + n].mean(dtype=np.float64)),
                          "calib": None if n == "floor" else calib_error(d_.d.values, S, P[n])}
            np.savez(f"{out_dir}/test_{key}.npz", **arrs)
            scores["months"][key] = {"steps": st, "summary": summarize(d_), "metrics": met}
            print(key, {n: round(met[n]["logloss"], 5) for n in NAMES}, flush=True)
    json.dump(scores, open(f"{out_dir}/scores.json", "w"), indent=1)
    with open(f"{out_dir}/score_log.txt", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} sha256={fz['sha256']} "
                f"months={[m.split(':')[0] for m in months]}\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
