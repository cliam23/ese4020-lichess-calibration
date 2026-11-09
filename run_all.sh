#!/usr/bin/env bash
# Download, parse and run the baseline from scratch.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data/raw data/processed
for m in 2015-01 2015-02; do
  f="data/raw/lichess_db_standard_rated_${m}.pgn.zst"
  [ -f "$f" ] || curl -L -o "$f" "https://database.lichess.org/standard/$(basename "$f")"
  python3 src/parse_pgn.py --month "$m" --input "$f" --out data/processed
done
python3 src/baseline.py
