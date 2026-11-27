"""Pick the settings with the lowest January CV log-loss and freeze them.

Reads results/cv_linear.json and cv_boost.json (January only), also runs the
5-fold CV for the ordinal logit (no hyperparameters), and writes
results/frozen_selection.json with a SHA-256 of its content. score.py refuses to
run unless the hash verifies, so the selection cannot change after a test month
has been read.

Usage: python -I src/freeze.py data/headers_2015-01.csv results/
"""

import hashlib
import json
import sys
import time

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from lib import *  # noqa: E402,F401,F403


def main(jan_path, out_dir):
    lin = json.load(open(f"{out_dir}/cv_linear.json"))["configs"]
    bst = json.load(open(f"{out_dir}/cv_boost.json"))["configs"]
    cfg = {**lin, **bst}

    df, _ = load_month(jan_path, "2015-01")
    fold = day_fold_ids(df.day.values)
    losses, sizes = [], []
    for k in range(len(FOLDS)):
        tr, va = df[fold != k], df[fold == k]
        losses.append(float(game_loss(va.S.values, predict_ordinal(fit_ordinal(tr), va)).mean()))
        sizes.append(int(len(va)))
    cfg["ordinal"] = {"fold_loss": losses, "fold_size": sizes, "kind": "ordinal",
                      "cv": float(np.average(losses, weights=sizes))}

    def best(prefix):
        names = [n for n in cfg if n.startswith(prefix)]
        return min(names, key=lambda n: cfg[n]["cv"])

    sel = {
        "baseline": min(["pooled", "bytc"], key=lambda n: cfg[n]["cv"]),
        "rating": {k: cfg[best("rating|")][k] for k in ("lam",)},
        "spline": {k: cfg[best("spline|")][k] for k in ("K", "lam")},
        "boost": {k: cfg[best("boost|")][k] for k in ("trees", "leaves", "lr")},
        "chosen_names": {"rating": best("rating|"), "spline": best("spline|"), "boost": best("boost|")},
    }
    body = json.dumps(sel, sort_keys=True)
    frozen = {**sel, "sha256": hashlib.sha256(body.encode()).hexdigest(),
              "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    json.dump(frozen, open(f"{out_dir}/frozen_selection.json", "w"), indent=1)
    json.dump(cfg, open(f"{out_dir}/cv_all.json", "w"), indent=1)
    print(json.dumps(frozen, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
