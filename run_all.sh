#!/usr/bin/env bash
# Reproduce every number, table and figure from scratch (about 30 minutes on 2 cores).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data/raw data/processed results figures
for m in 2015-01 2015-02; do
  f="data/raw/lichess_db_standard_rated_${m}.pgn.zst"
  [ -f "$f" ] || curl -L -o "$f" "https://database.lichess.org/standard/$(basename "$f")"
  python3 src/parse_pgn.py --month "$m" --input "$f" --out data/processed
done
cd src
python3 baseline.py                 # checkpoint baseline (Jan fit, CV, bootstrap)
python3 extended_cv.py logit        # model ladder, 5-fold day-blocked CV on January
python3 extended_cv.py gbm          # gradient-boosting comparison, same folds
python3 select_models.py            # freeze the choice before February is read
python3 rating_bands.py             # January: slope and White's edge by rating band
python3 sensitivity.py              # January: robustness subsets
python3 noise_prev.py               # January: smoothed-rating noise test
python3 simulate_noise.py           # simulation with known truth
python3 final_test.py --force       # February, scored once (reruns are logged)
python3 make_numbers.py             # every number in the paper -> paper/numbers.tex
cd ../paper && latexmk -pdf -interaction=nonstopmode main.tex
