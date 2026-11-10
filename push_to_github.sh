#!/usr/bin/env bash
# Push this folder to github.com/cliam23/ese4020-lichess-calibration as you.
# Run from Terminal:  bash push_to_github.sh
set -euo pipefail
cd "$(dirname "$0")"
rm -f run_parse.sh parse.log            # leftovers from the first parse attempt
[ -d .git ] || git init -q -b main
git add -A
git status --short
git commit -q -m "Baseline checkpoint: Elo vs logistic fit on Lichess January/February 2015

Data pipeline with filter counts, logistic regression by maximum likelihood
tuned by day-blocked 5-fold CV, day-level bootstrap intervals, rating-noise
check, February scoring of the pre-specified baselines, and the key figure." || true
git remote get-url origin >/dev/null 2>&1 || \
  git remote add origin https://github.com/cliam23/ese4020-lichess-calibration.git
git push -u origin main
echo "Done: https://github.com/cliam23/ese4020-lichess-calibration"
