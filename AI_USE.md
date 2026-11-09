# AI use log

The topic guide asks each group to keep a short log of which AI tools were used and for
what. The paper's acknowledgements summarize it.

- **9 November 2026 (baseline checkpoint).** Liam used Claude (Anthropic) to draft the data
  pipeline (`src/parse_pgn.py`), the logistic-regression and bootstrap code (`src/model.py`,
  `src/baseline.py`), the synthetic test (`tests/`), the key figure, this README, and a first
  draft of `submission/baseline.txt`. The assistant also checked the method against the course
  notes (Sections 3C and 4A) and the topic guide's leakage list.
- **What we verified ourselves (to do before the early draft).** Each member reads every line of
  the code they own; one teammate reruns `./run_all.sh` from scratch with seed 4020 and confirms
  the numbers in `results/summary.md`; every equation and figure can be explained by every member.

Draft statement for the paper: "We used Claude to draft the data-processing, model-fitting and
plotting code and to edit the prose. We read every line, reran every experiment ourselves with
fixed seeds, and verified every reference. The analysis and conclusions are our own."
