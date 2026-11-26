# AI use log

The topic guide and the final-paper assignment ask each group to keep a short log of which AI
tools were used and for what. The acknowledgements in `paper/main.tex` summarize it.

- **9 November 2026 (baseline checkpoint).** Liam used Claude (Anthropic) to draft the data
  pipeline, the logistic-regression and bootstrap code, the synthetic test, the key figure, the
  README and a first draft of `submission/baseline.txt`, and to check the method against the
  course notes (Sections 3C and 4A) and the topic guide's leakage list.
- **9 November 2026 (extended analysis and paper draft).** Claude drafted the extended model
  ladder and gradient-boosting comparison (`extended_cv.py`, `features.py`, `select_models.py`),
  the January-only analyses (`rating_bands.py`, `sensitivity.py`, `noise_prev.py`,
  `simulate_noise.py`), the one-time test script (`final_test.py`), the number-to-LaTeX
  generator (`make_numbers.py`) and the paper draft. It also searched for related work; each
  reference in `paper/refs.bib` was checked against a publisher, journal index or the original
  page before use, and one candidate reference that could not be confirmed was left out.
- **Checks done.** Headline numbers recomputed independently with scikit-learn on game-level rows
  (slope ratio, White's advantage, February log-loss of Elo, pooled fit and floor) match the
  paper. The February test set was scored once (`results/test_set_log.txt`).
- **Still to do by the group.** Each member reads the code for their part; one teammate reruns
  `./run_all.sh` from scratch and compares `results/summary.md` and `paper/numbers.tex`; each
  member opens every reference; every member can explain every equation and figure.

- **26 November 2026 (final-paper version).** AI was used for guidance; project direction, design, verification and
  decisions were made by the team. Claude reimplemented the pipeline in `final/` (from the early-draft PDF
  and then compared with this repository), added the player-level bootstrap, the March 2015 test month (declared
  in `final/results/declaration.md` before it was downloaded), the extended simulations, the ordinal-logit and
  rating-change checks, and revised the paper text. It verified new references against Crossref, arXiv and the Lichess
  source code. A headline number (February log-loss of the baseline) was recomputed independently with scikit-learn
  and matches to ten digits; a tampered selection file is refused by `score.py`.
- **Still to do by the group.** Each member reads the new code for their part; one teammate runs `final/run_all.sh`
  from scratch and compares `final/paper/numbers.tex`; each member opens every new reference
  (Bradley and Terry 1952; Berkson 1950; Owen 2007; Cameron, Gelbach and Miller 2011; Gneiting and Raftery 2007;
  Guo et al. 2017; Hvattum and Arntzen 2010).
