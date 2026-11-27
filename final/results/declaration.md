# Second test month: declared before March 2015 was downloaded or read

Written (UTC): 2026-11-26T18:36:12Z

Second test month: March 2015 (lichess_db_standard_rated_2015-03.pgn.zst, database.lichess.org).

Procedure. Every modelling choice (model family, penalty, number of knots, boosting settings) is selected on January 2015 only and frozen in results/frozen_selection.json before any test month is scored. The models are refit on all of January and scored once on March. Nothing is retuned after March is read.

What would count as "the conclusions are stable":
1. The pooled course baseline beats the Elo formula on March log-loss, and the 95% interval of the paired difference (player-level bootstrap) excludes zero.
2. The ordering Elo > baseline >= spline > boosted trees on log-loss is the same as in February.
3. The slope ratio fitted on March alone lies within 0.03 of the January value.

If any of the three fails, we report that it failed.
