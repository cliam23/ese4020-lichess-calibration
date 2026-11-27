"""Shared code for the final-paper pipeline: loading, features, models, metrics.

Every model predicts White's expected score S (1 win, 1/2 draw, 0 loss) from
pre-game information. Logistic models are fitted by Newton's method on the
fractional-response log-likelihood, which is exactly the weighted 0/1 MLE
(a draw is a half-weight win and a half-weight loss).
"""

import numpy as np
import pandas as pd

SEED = 4020
CLASSES = ["Bullet", "Blitz", "Classical"]
RESULT = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}
LN10 = float(np.log(10.0))
FOLDS = [(1, 7), (8, 13), (14, 19), (20, 25), (26, 31)]  # contiguous blocks of days
DOFF = 3000  # d + DOFF indexes an aggregation bin
NBIN = 2 * DOFF + 1
EPS = 1e-12


# ------------------------------------------------------------------ data

def load_month(path, month):
    """Clean one month of headers. Returns (DataFrame, dict of step counts)."""
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    tcname = raw.Event.str.extract(r"^Rated (\w+)")[0]
    # Lichess renamed part of the old "Classical" class "Standard" in March 2015, so the
    # class is taken from the estimated duration (initial seconds + 40 x increment), which
    # reproduces the January and February labels exactly (checked below):
    # bullet < 180 s, blitz 180-479 s, classical >= 480 s.
    tcs = raw.TimeControl.str.extract(r"^(\d+)\+(\d+)$").astype(float)
    est = (tcs[0] + 40.0 * tcs[1]).to_numpy()
    cls = np.where(est < 180, 0, np.where(est < 480, 1, 2))
    label_cls = tcname.map({"Bullet": 0, "Blitz": 1, "Classical": 2, "Standard": 2}).to_numpy()
    keep = (tcname.isin(["Bullet", "Blitz", "Classical", "Standard"]) & ~np.isnan(est)).to_numpy().copy()
    steps = {"raw": int(len(raw)), "not_bullet_blitz_classical": int((~keep).sum())}
    steps["label_vs_duration_mismatch"] = int((keep & (cls != label_cls)).sum())
    ab = (raw.Termination == "Abandoned").values
    steps["abandoned"] = int((keep & ab).sum())
    keep &= ~ab
    rat = (raw.WhiteElo.str.fullmatch(r"\d+") & raw.BlackElo.str.fullmatch(r"\d+")).values
    steps["missing_rating"] = int((keep & ~rat).sum())
    keep &= rat
    res = raw.Result.isin(list(RESULT)).values
    steps["bad_result"] = int((keep & ~res).sum())
    keep &= res
    inmonth = raw.UTCDate.str.startswith(month.replace("-", ".")).values
    steps["outside_month"] = int((keep & ~inmonth).sum())
    keep &= inmonth
    steps["final"] = int(keep.sum())
    r = raw[keep]
    df = pd.DataFrame({
        "day": r.UTCDate.str[8:10].astype(int).values,
        "tc": cls[keep].astype(np.int8),
        "std": (tcname[keep] == "Standard").to_numpy(),
        "w": r.WhiteElo.astype(int).values,
        "b": r.BlackElo.astype(int).values,
        "S": r.Result.map(RESULT).values.astype(float),
        "wid": r.white_id.astype(int).values,
        "bid": r.black_id.astype(int).values,
        "wdiff": pd.to_numeric(r.WhiteRatingDiff, errors="coerce").values,
        "bdiff": pd.to_numeric(r.BlackRatingDiff, errors="coerce").values,
        "tf": (r.Termination == "Time forfeit").values,
        "infr": (r.Termination == "Rules infraction").values,
    })
    df["d"] = df.w - df.b
    df["m"] = (df.w + df.b) / 2.0
    return df.reset_index(drop=True), steps


def summarize(df):
    return {
        "n": int(len(df)), "n_days": int(df.day.nunique()),
        "n_players": int(len(np.unique(np.r_[df.wid.values, df.bid.values]))),
        "white_score": float(df.S.mean()), "draw_rate": float((df.S == 0.5).mean()),
        "abs_gap_median": float(df.d.abs().median()),
        "rating_mean": float(df.m.mean()),
        "rating_sd": float(pd.concat([df.w, df.b]).std()),
        "gap_sd": float(df.d.std()), "gap_mean": float(df.d.mean()),
        "n_infraction": int(df.infr.sum()), "n_time_forfeit": int(df.tf.sum()),
        "by_class": {c: int((df.tc == i).sum()) for i, c in enumerate(CLASSES)},
    }


def day_fold_ids(day):
    f = np.full(len(day), -1)
    for k, (lo, hi) in enumerate(FOLDS):
        f[(day >= lo) & (day <= hi)] = k
    assert (f >= 0).all()
    return f


# ------------------------------------------------------- scores and metrics

def elo_p(d):
    return 1.0 / (1.0 + 10.0 ** (-np.asarray(d, dtype=float) / 400.0))


def game_loss(S, p):
    p = np.clip(p, EPS, 1 - EPS)
    return -(S * np.log(p) + (1 - S) * np.log1p(-p))


def game_brier(S, p):
    return (S - p) ** 2


def calib_error(d, S, p, lo=-600, hi=600, width=50, min_n=200):
    """Game-weighted mean |observed - predicted| over 50-point bins of the gap."""
    b = np.floor((d - lo) / width).astype(int)
    ok = (d >= lo) & (d < hi)
    nb = (hi - lo) // width
    n = np.bincount(b[ok], minlength=nb)
    so = np.bincount(b[ok], weights=S[ok], minlength=nb)
    sp = np.bincount(b[ok], weights=p[ok], minlength=nb)
    use = n >= min_n
    return float(np.sum(np.abs(so[use] - sp[use])) / n[use].sum())


# ----------------------------------------------------------- logistic fits

def sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def fit_logit(X, S, lam=0.0, w=None, maxit=60):
    """Penalised fractional logit by Newton's method.

    Minimises  sum_i w_i [log(1+e^eta_i) - S_i eta_i] + W*lam*||beta||^2  where W = sum w
    over standardised columns of X (the intercept is not penalised). Returns a
    dict that predict_logit understands.
    """
    X = np.asarray(X, dtype=float)
    n, k = X.shape
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Z = np.empty((n, k + 1))
    Z[:, 0] = 1.0
    Z[:, 1:] = (X - mu) / sd
    wv = np.ones(n) if w is None else np.asarray(w, dtype=float)
    W = wv.sum()
    pen = np.r_[0.0, np.full(k, 2.0 * W * lam)]
    beta = np.zeros(k + 1)
    sbar = np.average(S, weights=wv)
    beta[0] = np.log(sbar / (1 - sbar))

    def obj(b):
        eta = Z @ b
        return float(np.sum(wv * (np.logaddexp(0.0, eta) - S * eta)) + 0.5 * np.sum(pen * b * b))

    f = obj(beta)
    for _ in range(maxit):
        eta = Z @ beta
        p = sigmoid(eta)
        g = Z.T @ (wv * (p - S)) + pen * beta
        h = wv * p * (1 - p)
        H = (Z * h[:, None]).T @ Z + np.diag(pen) + 1e-10 * np.eye(k + 1)
        step = np.linalg.solve(H, g)
        t = 1.0
        while t > 1e-6:
            fn = obj(beta - t * step)
            if fn <= f + 1e-12 * abs(f):
                break
            t *= 0.5
        beta = beta - t * step
        done = abs(f - fn) < 1e-11 * W
        f = fn
        if done or np.max(np.abs(t * step)) < 1e-10:
            break
    coef = beta[1:] / sd
    return {"beta": beta, "mu": mu, "sd": sd,
            "intercept": float(beta[0] - np.sum(mu * coef)), "coef": coef}


def predict_logit(m, X):
    Z = (np.asarray(X, dtype=float) - m["mu"]) / m["sd"]
    return sigmoid(m["beta"][0] + Z @ m["beta"][1:])


def ns_nonlinear(x, knots):
    """Nonlinear part of the natural cubic spline basis (Hastie et al., eq. 5.4-5.5)."""
    K = len(knots)
    xiK = knots[-1]

    def dk(k):
        return (np.maximum(x - knots[k], 0.0) ** 3 - np.maximum(x - xiK, 0.0) ** 3) / (xiK - knots[k])

    dKm1 = dk(K - 2)
    return np.column_stack([dk(k) - dKm1 for k in range(K - 2)])


def design(kind, df, knots=None):
    x = df.d.values / 400.0
    if kind == "pooled":
        return x[:, None]
    B = (df.tc.values == 1).astype(float)
    C = (df.tc.values == 2).astype(float)
    base = np.column_stack([B, C, x, x * B, x * C])
    if kind == "bytc":
        return base
    mm = (df.m.values - 1600.0) / 400.0
    rating = np.column_stack([base, mm, mm ** 2, x * mm])
    if kind == "rating":
        return rating
    if kind == "spline":
        return np.column_stack([rating, ns_nonlinear(x, np.asarray(knots))])
    raise ValueError(kind)


def spline_knots(df, K):
    return np.quantile(df.d.values / 400.0, np.linspace(0, 1, K))


def n_params(kind, K=None):
    return {"floor": 1, "elo": 0, "pooled": 2, "bytc": 6, "rating": 9, "ordinal": 9}.get(
        kind, 9 + (K - 2) if kind == "spline" else None)


# --------------------------------------------------------------- boosting

def hgb_xy(df):
    return np.column_stack([df.d.values, df.m.values, df.tc.values]).astype(np.float64)


def fit_hgb(df, trees, leaves, lr, seed=4020):
    from sklearn.ensemble import HistGradientBoostingClassifier
    X, S = hgb_xy(df), df.S.values
    pos, neg = S > 0, S < 1
    Xe = np.vstack([X[pos], X[neg]])
    ye = np.r_[np.ones(pos.sum()), np.zeros(neg.sum())]
    we = np.r_[S[pos], 1.0 - S[neg]]
    clf = HistGradientBoostingClassifier(
        loss="log_loss", max_iter=trees, learning_rate=lr, max_leaf_nodes=leaves,
        min_samples_leaf=200, early_stopping=False, random_state=seed,
        categorical_features=[False, False, True], monotonic_cst=[1, 0, 0])
    return clf.fit(Xe, ye, sample_weight=we)


def predict_hgb(clf, df):
    return clf.predict_proba(hgb_xy(df))[:, 1]


# ----------------------------------------------- aggregated 2-parameter fit

def agg_fit_ab(wins, tot, maxit=30):
    """Fractional logit of S on x = d/400 from counts per integer gap.

    wins[k] is the weighted score mass at gap k - DOFF and tot[k] the weight.
    Returns (a, b) with logit E[S] = a + b d/400.
    """
    ok = tot > 0
    x = (np.arange(len(tot))[ok] - DOFF) / 400.0
    wv, nv = wins[ok], tot[ok]
    a, b = 0.0, LN10
    for _ in range(maxit):
        p = sigmoid(a + b * x)
        r = wv - nv * p
        v = nv * p * (1 - p)
        g0, g1 = r.sum(), (r * x).sum()
        h00, h01, h11 = v.sum(), (v * x).sum(), (v * x * x).sum()
        det = h00 * h11 - h01 * h01
        da, db = (h11 * g0 - h01 * g1) / det, (-h01 * g0 + h00 * g1) / det
        a, b = a + da, b + db
        if abs(da) + abs(db) < 1e-12:
            break
    return a, b


def edge_stats(a, b):
    """White's edge in score points, in rating points, and the slope ratio."""
    return {"edge_pp": float(sigmoid(a) - 0.5), "edge_pts": float(400.0 * a / b),
            "slope_ratio": float(b / LN10), "a": float(a), "b": float(b)}


# ------------------------------------------------- ordinal logit for draws

def fit_ordinal(df):
    """Cumulative logit for loss < draw < win, separately for each time control.

    P(loss) = s(t1 - b x), P(loss or draw) = s(t2 - b x), x = d / 400. Returns
    {tc: (t1, t2, b)}. Fitted by BFGS on counts per integer gap.
    """
    from scipy.optimize import minimize
    out = {}
    for k in range(3):
        g = df[df.tc == k]
        key = (g.d.values + DOFF).astype(int)
        n_l = np.bincount(key, weights=(g.S.values == 0), minlength=NBIN)
        n_d = np.bincount(key, weights=(g.S.values == 0.5), minlength=NBIN)
        n_w = np.bincount(key, weights=(g.S.values == 1), minlength=NBIN)
        ok = (n_l + n_d + n_w) > 0
        x = (np.arange(NBIN)[ok] - DOFF) / 400.0
        nl, nd, nw = n_l[ok], n_d[ok], n_w[ok]

        def nll(th):
            t1, t2, b = th[0], th[0] + np.exp(th[1]), th[2]
            c1, c2 = sigmoid(t1 - b * x), sigmoid(t2 - b * x)
            pl, pd_, pw = c1, c2 - c1, 1 - c2
            return -(nl @ np.log(np.maximum(pl, EPS)) + nd @ np.log(np.maximum(pd_, EPS))
                     + nw @ np.log(np.maximum(pw, EPS))) / (nl + nd + nw).sum()

        r = minimize(nll, np.array([-0.1, np.log(0.15), 1.8]), method="BFGS", options={"gtol": 1e-9})
        out[k] = (float(r.x[0]), float(r.x[0] + np.exp(r.x[1])), float(r.x[2]))
    return out


def predict_ordinal(par, df):
    """Expected score P(win) + P(draw)/2 under the fitted cumulative logit."""
    x = df.d.values / 400.0
    e = np.empty(len(df))
    for k, (t1, t2, b) in par.items():
        m = df.tc.values == k
        e[m] = 1.0 - 0.5 * sigmoid(t1 - b * x[m]) - 0.5 * sigmoid(t2 - b * x[m])
    return e


# --------------------------------------------------- earlier-day history

def add_history(df):
    """Add, for each side, what is known about the player before the game's day.

    Columns w_hn / b_hn: games on earlier days; w_hmean / b_hmean: mean displayed
    rating over those games; w_hdiff / b_hdiff: mean absolute rating change over
    those games; w_games / b_games: the player's games in the month. Nothing for
    day t uses a game from day t or later, so no history can contain the result.
    """
    n = len(df)
    long = pd.concat([
        pd.DataFrame({"pid": df.wid.values, "day": df.day.values, "r": df.w.values,
                      "ad": df.wdiff.abs().values}),
        pd.DataFrame({"pid": df.bid.values, "day": df.day.values, "r": df.b.values,
                      "ad": df.bdiff.abs().values}),
    ])
    long["has"] = long.ad.notna().astype(float)
    long["ad"] = long.ad.fillna(0.0)
    g = long.groupby(["pid", "day"], as_index=False).agg(
        s=("r", "sum"), n=("r", "size"), a=("ad", "sum"), h=("has", "sum"))
    g = g.sort_values(["pid", "day"]).reset_index(drop=True)
    for c in ["s", "n", "a", "h"]:
        g["c" + c] = g.groupby("pid")[c].cumsum() - g[c]
    g = g[["pid", "day", "cs", "cn", "ca", "ch"]]
    for side, idc in (("w", "wid"), ("b", "bid")):
        mg = df[[idc, "day"]].rename(columns={idc: "pid"}).merge(g, on=["pid", "day"], how="left")
        assert len(mg) == n
        cn, ch = mg.cn.values, mg.ch.values
        df[side + "_hn"] = cn
        with np.errstate(invalid="ignore", divide="ignore"):
            df[side + "_hmean"] = np.where(cn > 0, mg.cs.values / cn, np.nan)
            df[side + "_hdiff"] = np.where(ch > 0, mg.ca.values / ch, np.nan)
    cnt = np.bincount(np.r_[df.wid.values, df.bid.values])
    df["w_games"] = cnt[df.wid.values]
    df["b_games"] = cnt[df.bid.values]
    return df
