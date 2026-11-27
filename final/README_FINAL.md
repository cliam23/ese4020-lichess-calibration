# Final-paper pipeline (December version)

This directory adds to the early-draft pipeline in `../src` without changing it. It is a rebuilt,
self-contained pipeline that reproduces the early draft's numbers (same cleaning counts, same Table 1
log-losses, same 0.78 slope, same 56-point simulation) and adds what the draft feedback asked for.

Added in this version:
- player-level (pigeonhole) bootstrap next to the day-block and game-level ones (`src/boot_test.py`, `src/boot_coef.py`)
- a second test month, March 2015, declared before it was downloaded (`results/declaration.md`, hash in `declaration.sha256`)
- selection frozen by hash before any test month is scored (`src/freeze.py`, `src/score.py`)
- simulation with three noise constructions (`src/noise.py sim`) and the simulation figure
- ordinal logit for draws, rating-change proxy for how settled a player is, intervals for White's edge by time class
- usernames are mapped to integers in memory only, per month (`src/parse_headers.py`); no username is written

Differences from the early-draft results in `../results`: the tuning grids differ, so the frozen choices differ
(here: spline K = 10, boosting 100 trees / 63 leaves; early draft: df 12, 100 trees / 127 leaves). The
CV differences between these choices are about 1e-6 and February log-losses agree to four decimals.

Rebuild (downloads about 880 MB, then runs tuning, scoring, bootstraps and the PDF):

    ./run_all.sh

Order matters: January is tuned and the selection frozen before any test month is scored, and
`results/declaration.md` must exist before March is downloaded. Per-game arrays (`results/test_*.npz`) are
rebuilt by `src/score.py` and are not committed.
