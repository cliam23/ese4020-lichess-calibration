"""Stream a Lichess PGN dump and keep only the header fields we need.

Usage:
    python3 src/parse_pgn.py --month 2015-01 \
        --input data/raw/lichess_db_standard_rated_2015-01.pgn.zst
or, with the zstd command:
    zstd -dc data/raw/...pgn.zst | python3 src/parse_pgn.py --month 2015-01

Reads PGN text one game at a time and never holds the move text.
Writes:
    games_<month>.csv.gz   one row per kept game
    filters_<month>.json   how many games each filter removed, in order

Columns kept: day, tc, white_elo, black_elo, score, termination,
n_white, n_black.  score is White's score (1, 0.5, 0).  n_white and
n_black are how many kept games each player has in the month; they let us
refit on players seen often (the rating-noise check) without storing any
username.  Usernames are mapped to integers in memory and dropped.
"""
import argparse
import csv
import io
import gzip
import json
import os
import sys
from collections import Counter, OrderedDict

KEEP = ("Event", "White", "Black", "Result", "UTCDate", "WhiteElo",
        "BlackElo", "TimeControl", "Termination")
SCORE = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}


def tc_from_clock(tc):
    """2015 Lichess speed classes from estimated duration base + 40*inc."""
    try:
        base, inc = (int(x) for x in tc.split("+"))
    except ValueError:
        return None
    est = base + 40 * inc
    if est < 180:
        return "bullet"
    if est < 480:
        return "blitz"
    return "classical"


def tc_class(event, tc):
    e = event.lower()
    for name in ("bullet", "blitz", "classical"):
        if name in e:
            return name, False
    return tc_from_clock(tc), True


def open_pgn(path):
    """Text stream over a .pgn.zst, a .pgn, or stdin."""
    if path is None:
        return sys.stdin
    if path.endswith(".zst"):
        import zstandard
        raw = open(path, "rb")
        dec = zstandard.ZstdDecompressor(max_window_size=2 ** 31)
        return io.TextIOWrapper(dec.stream_reader(raw), encoding="utf-8")
    return open(path, encoding="utf-8")


def games(stream):
    """Yield one dict of header fields per game."""
    hdr = {}
    for line in stream:
        if line.startswith("["):
            if line.startswith("[Event ") and hdr:
                yield hdr
                hdr = {}
            sp = line.find(" ")
            key = line[1:sp]
            if key in KEEP:
                hdr[key] = line[sp + 2:line.rfind('"')]
    if hdr:
        yield hdr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", required=True, help="e.g. 2015-01")
    ap.add_argument("--input", help=".pgn.zst or .pgn file (default: stdin)")
    ap.add_argument("--out", default="data/processed")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    removed = OrderedDict((k, 0) for k in (
        "correspondence", "abandoned", "missing_rating",
        "unfinished_result", "wrong_month_or_bad_date"))
    total = 0
    tc_fallback = 0
    term = Counter()
    rows, ids = [], {}

    for g in games(open_pgn(args.input)):
        total += 1
        ev, tc = g.get("Event", ""), g.get("TimeControl", "-")
        if "correspondence" in ev.lower() or tc == "-":
            removed["correspondence"] += 1
            continue
        if g.get("Termination", "") == "Abandoned":
            removed["abandoned"] += 1
            continue
        we, be = g.get("WhiteElo", "?"), g.get("BlackElo", "?")
        if not (we.isdigit() and be.isdigit()):
            removed["missing_rating"] += 1
            continue
        if g.get("Result") not in SCORE:
            removed["unfinished_result"] += 1
            continue
        date = g.get("UTCDate", "")
        if not date.startswith(args.month.replace("-", ".")):
            removed["wrong_month_or_bad_date"] += 1
            continue
        cls, fb = tc_class(ev, tc)
        tc_fallback += fb
        term[g.get("Termination", "")] += 1
        w = ids.setdefault(g.get("White", ""), len(ids))
        b = ids.setdefault(g.get("Black", ""), len(ids))
        rows.append((int(date[8:10]), cls, int(we), int(be),
                     SCORE[g["Result"]], g.get("Termination", ""), w, b))

    counts = Counter()
    for r in rows:
        counts[r[6]] += 1
        counts[r[7]] += 1
    del ids  # usernames never leave memory

    path = os.path.join(args.out, f"games_{args.month}.csv.gz")
    with gzip.open(path, "wt", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["day", "tc", "white_elo", "black_elo", "score",
                     "termination", "n_white", "n_black"])
        for r in rows:
            wr.writerow(r[:6] + (counts[r[6]], counts[r[7]]))

    summary = {
        "month": args.month,
        "games_in_file": total,
        "removed_in_order": removed,
        "kept": len(rows),
        "players": len(counts),
        "tc_from_clock_fallback": tc_fallback,
        "kept_by_termination": dict(term),
        "kept_by_tc": dict(Counter(r[1] for r in rows)),
    }
    with open(os.path.join(args.out, f"filters_{args.month}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
