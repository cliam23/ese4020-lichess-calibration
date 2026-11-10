"""Design matrices for the extended models, built inside each fold.

Every quantity learned from data (spline knots, column means and SDs) is
computed from the training games passed to `fit_transform` and reused
unchanged by `transform`, so held-out games never shape their own features.
"""
import numpy as np

from model import TCS


def natural_spline_terms(x, knots):
    """Nonlinear columns of a natural cubic spline basis (ESL eq. 5.4-5.5).

    With K knots the full basis is {1, x, N_3, ..., N_K}; this returns the
    K - 2 columns N_3..N_K.  The fit is linear beyond the boundary knots.
    """
    K = len(knots)

    def dk(k):
        a = np.clip(x - knots[k], 0, None) ** 3
        b = np.clip(x - knots[-1], 0, None) ** 3
        return (a - b) / (knots[-1] - knots[k])

    last = dk(K - 2)
    return np.column_stack([dk(k) - last for k in range(K - 2)]) if K > 2 \
        else np.empty((len(x), 0))


class Design:
    """Feature ladder.

    level 1  pooled:      1, d
    level 2  by_tc:       tc intercepts, tc x d                      (baseline)
    level 3  rating:      level 2 + rbar, rbar^2, d x rbar
    level 4  spline:      level 3 + natural spline in d with `df` knots
    d and rbar are in units of 100 rating points before standardizing.
    """

    def __init__(self, level, df=5):
        self.level, self.df = level, df

    def _raw(self, g):
        d = g["d"].values / 100.0
        rbar = (g["white_elo"].values + g["black_elo"].values) / 200.0
        cols, names, pen = [], [], []
        if self.level == 1:
            cols += [np.ones_like(d), d]
            names += ["1", "d"]
            pen += [False, True]
        else:
            tc = g["tc"].astype(str).values
            for t in TCS:
                cols.append((tc == t).astype(float))
                names.append(f"1[{t}]")
                pen.append(False)
            for t in TCS:
                cols.append((tc == t) * d)
                names.append(f"d[{t}]")
                pen.append(True)
        if self.level >= 3:
            cols += [rbar, rbar ** 2, d * rbar]
            names += ["rbar", "rbar^2", "d*rbar"]
            pen += [True, True, True]
        if self.level >= 4:
            S = natural_spline_terms(d, self.knots)
            for j in range(S.shape[1]):
                cols.append(S[:, j])
                names.append(f"ns{j}")
                pen.append(True)
        return np.column_stack(cols), names, np.array(pen)

    def fit_transform(self, g):
        if self.level >= 4:
            d = g["d"].values / 100.0
            q = np.linspace(0.025, 0.975, self.df)
            self.knots = np.quantile(d, q)
        X, self.names, self.pen = self._raw(g)
        # standardise penalised columns with training statistics only
        self.mu = np.where(self.pen, X.mean(0), 0.0)
        sd = X.std(0)
        self.sd = np.where(self.pen & (sd > 0), sd, 1.0)
        return (X - self.mu) / self.sd

    def transform(self, g):
        X, _, _ = self._raw(g)
        return (X - self.mu) / self.sd
