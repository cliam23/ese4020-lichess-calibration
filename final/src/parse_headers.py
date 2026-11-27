"""Turn Lichess PGN header lines into one CSV row per game.

Reads header lines on stdin (pre-filtered with grep so the move text never
reaches Python) and writes a CSV to the path given as the only argument.

Usernames are replaced by integers the moment they are read, in order of first
appearance within the file, and the name-to-integer table lives only in this
process's memory: it is never written, so no username reaches the output. The
integers let later code count each player's games and cluster the bootstrap
by player. They do not link players across months.

Usage:
    zstd -dc lichess_db_standard_rated_2015-01.pgn.zst \
      | grep -E '^\\[(Event|White|Black|UTCDate|Result|WhiteElo|BlackElo|WhiteRatingDiff|BlackRatingDiff|TimeControl|Termination) "' \
      | python -I src/parse_headers.py data/headers_2015-01.csv
"""

import csv
import re
import sys

FIELDS = ["Event", "UTCDate", "Result", "WhiteElo", "BlackElo", "WhiteRatingDiff",
          "BlackRatingDiff", "TimeControl", "Termination"]
LINE = re.compile(r'^\[(\w+) "(.*)"\]\s*$')


def main(out_path):
    ids = {}  # name -> integer, in memory only
    n = 0
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(FIELDS + ["white_id", "black_id"])

        def emit(g):
            w.writerow([g.get(k, "") for k in FIELDS] + [g.get("white_id", ""), g.get("black_id", "")])

        game = None
        for line in sys.stdin:
            m = LINE.match(line)
            if not m:
                continue
            key, value = m.groups()
            if key == "Event":
                if game is not None:
                    emit(game)
                    n += 1
                game = {}
            if game is None:
                continue
            if key in ("White", "Black"):
                game[key.lower() + "_id"] = ids.setdefault(value, len(ids))
            elif key in FIELDS:
                game[key] = value
        if game is not None:
            emit(game)
            n += 1
    print(f"{out_path}: {n} games, {len(ids)} players (names not stored)", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1])
