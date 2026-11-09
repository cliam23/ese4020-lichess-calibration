"""Write a fake Lichess-style PGN with known b0, b1 to check the pipeline.
python3 tests/make_synthetic_pgn.py 2015-01 60000 > fake.pgn"""
import sys, numpy as np
month, N = sys.argv[1], int(sys.argv[2])
rng = np.random.default_rng(1)
B0, B1, DRAW = 0.10, 0.8 * np.log(10) / 400, 0.06
players = [f"p{i}" for i in range(3000)]
rate = rng.normal(1600, 250, 3000).round()
tcs = [("Rated Bullet game", "60+0"), ("Rated Blitz game", "300+0"),
       ("Rated Classical game", "600+5"), ("Rated Correspondence game", "-")]
out = sys.stdout
for g in range(N):
    w, b = rng.choice(3000, 2, replace=False)
    ev, tc = tcs[rng.choice(4, p=[.3, .45, .23, .02])]
    we, be = int(rate[w] + rng.normal(0, 40)), int(rate[b] + rng.normal(0, 40))
    p = 1 / (1 + np.exp(-(B0 + B1 * (we - be))))
    u = rng.random()
    res = "1/2-1/2" if u < DRAW else ("1-0" if rng.random() < (p - DRAW/2)/(1-DRAW) else "0-1")
    term = "Abandoned" if rng.random() < .005 else ("Time forfeit" if rng.random() < .3 else "Normal")
    if rng.random() < .002: we = "?"
    day = rng.integers(1, 29 if month.endswith("02") else 32)
    out.write(f'[Event "{ev}"]\n[Site "https://lichess.org/x"]\n[White "{players[w]}"]\n[Black "{players[b]}"]\n'
              f'[Result "{res}"]\n[UTCDate "{month.replace("-", ".")}.{day:02d}"]\n[UTCTime "00:00:00"]\n'
              f'[WhiteElo "{we}"]\n[BlackElo "{be}"]\n[ECO "C00"]\n[TimeControl "{tc}"]\n[Termination "{term}"]\n\n'
              f'1. e4 e6 2. d4 d5 {res}\n\n')
