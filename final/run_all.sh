#!/usr/bin/env bash
# Rebuild everything behind paper/draft.pdf from the raw Lichess files.
# Needs zstd, curl, pdflatex and Python 3 with numpy, pandas, scikit-learn, scipy, matplotlib.
# Order matters: January is tuned and the selection frozen BEFORE any test month is scored,
# and results/declaration.md (the second test month) must exist before March is downloaded.
set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-python3}
mkdir -p data/raw results logs
KEEP='^\[(Event|White|Black|UTCDate|Result|WhiteElo|BlackElo|WhiteRatingDiff|BlackRatingDiff|TimeControl|Termination) "'

for m in 01 02 03; do
  f=data/raw/lichess_db_standard_rated_2015-$m.pgn.zst
  if [ "$m" = 03 ] && [ ! -f results/declaration.md ]; then echo "write results/declaration.md before downloading March"; exit 1; fi
  [ -f "$f" ] || curl -sSfL -o "$f" "https://database.lichess.org/standard/$(basename "$f")"
  [ -f data/headers_2015-$m.csv ] || zstd -dc "$f" | grep -E "$KEEP" | "$PY" -I src/parse_headers.py data/headers_2015-$m.csv
done

# 1. January only: cross-validation, then freeze the selection (hash-checked by score.py)
"$PY" -I src/tune.py linear data/headers_2015-01.csv results
"$PY" -I src/tune.py boost  data/headers_2015-01.csv results
"$PY" -I src/freeze.py data/headers_2015-01.csv results

# 2. Refit on January with the frozen settings, score February and March (and March by labels only)
"$PY" -I src/score.py results data/headers_2015-01.csv 2015-02:data/headers_2015-02.csv 2015-03:data/headers_2015-03.csv

# 3. Bootstraps: test-month losses (game / day / player) and the baseline's coefficients
"$PY" -I src/boot_test.py results 2015-02 --B 2000
"$PY" -I src/boot_test.py results 2015-03 --B 2000
"$PY" -I src/boot_test.py results 2015-03L --B 2000
for m in 2015-01 2015-02 2015-03; do for s in game day player; do
  "$PY" -I src/boot_coef.py results $m data/headers_$m.csv $s 1000
done; done

# 4. Rating-noise checks and simulation
"$PY" -I src/noise.py smooth data/headers_2015-01.csv results
"$PY" -I src/noise.py sim    data/headers_2015-01.csv results

# 5. Figures, tables, numbers, PDF
"$PY" -I - <<'PYEOF'
import json, sys
sys.path.insert(0, "src")
from lib import load_month, summarize
df, _ = load_month("data/headers_2015-01.csv", "2015-01")
s = summarize(df); s["m_sd"] = float(df.m.std())
json.dump(s, open("results/jan_summary.json", "w"), indent=1)
d3, _ = load_month("data/headers_2015-03.csv", "2015-03")
json.dump({"n_standard": int(d3["std"].sum())}, open("results/mar_standard.json", "w"))
PYEOF
"$PY" -I src/make_figures.py results data/headers_2015-01.csv paper
"$PY" -I src/make_tables.py results paper
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex > build.log
pdflatex -interaction=nonstopmode -halt-on-error main.tex > build.log
cp main.pdf draft.pdf
echo "wrote paper/draft.pdf"
