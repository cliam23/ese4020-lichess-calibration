"""Logistic regression on White's expected score, fitted on grouped cells.

Every game with the same (day, time control, rating gap d) shares one
prediction, so we collapse games into cells holding n (games), s (sum of
White's score) and s2 (sum of squared score).  With S in {0, 1/2, 1} the
per-game log-likelihood S log p + (1 - S) log(1 - p) sums over a cell to
s log p + (n - s) log(1 - p): exactly the "draw = half a win plus half a
loss" weighting from the proposal feedback.  Fitting on cells is the same
maximum-likelihood fit as fitting on games, only faster, and it makes the
day bootstrap cheap (resampling a day just reweights its cells).
"""
import numpy as np

ELO_SLOPE = np.log(10) / 400  # 0.005756...
TCS = ("bullet", "blitz", "classical")


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def elo_prob(d):
    return sigmoid(ELO_SLOPE * d)


def cells(df, keys=("day", "tc", "d")):
    """Collapse a game table into cells keyed by `keys`."""
    return (df.assign(sq=df["score"] ** 2)
              .groupby(list(keys), sort=False, observed=True)
              .agg(n=("score", "size"), s=("score", "sum"), s2=("sq", "sum"))
              .reset_index())


def log_loss(p, n, s):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(s * np.log(p) + (n - s) * np.log1p(-p)).sum() / n.sum()


def brier(p, n, s, s2):
    return (s2 - 2 * p * s + n * p ** 2).sum() / n.sum()


def newton(X, n, s, lam=0.0, pen=None, iters=100, tol=1e-9, init=None):
    """Minimise  sum NLL + (N lam / 2) * ||beta[pen]||^2  by Newton's method.

    lam is a per-game penalty (like sklearn's 1/(C N)), so the grid means
    the same thing whatever the fold size.  The intercepts are never
    penalised.
    """
    k = X.shape[1]
    beta = np.zeros(k) if init is None else init.copy()
    P = np.zeros((k, k))
    if lam > 0 and pen is not None:
        P[pen, pen] = lam * n.sum()
    for _ in range(iters):
        p = sigmoid(X @ beta)
        grad = X.T @ (n * p - s) + P @ beta
        H = (X * (n * p * (1 - p))[:, None]).T @ X + P
        step = np.linalg.solve(H, grad)
        beta -= step
        if np.max(np.abs(step)) < tol:
            break
    return beta


class Logit:
    """Two structures from the proposal's first question.

    pooled : E[S|d] = sigmoid(b0 + b1 d)                 2 parameters
    by_tc  : separate b0 and b1 for bullet, blitz and     6 parameters
             classical
    d is standardised inside fit() with the training cells' own mean and
    SD, so nothing about held-out games leaks into the scaling.
    """

    def __init__(self, structure="pooled", lam=0.0):
        self.structure, self.lam = structure, lam

    def _design(self, c):
        z = (c["d"].values - self.mu) / self.sd
        if self.structure == "pooled":
            X = np.column_stack([np.ones_like(z), z])
            pen = [1]
        else:
            oh = np.column_stack([(c["tc"].values == t).astype(float)
                                  for t in TCS])
            X = np.hstack([oh, oh * z[:, None]])
            pen = [3, 4, 5]
        return X, pen

    def fit(self, c, weight=None, init=None):
        n = c["n"].values.astype(float)
        s = c["s"].values.astype(float)
        if weight is not None:
            n, s = n * weight, s * weight
        d = c["d"].values
        self.mu = np.average(d, weights=n)
        self.sd = np.sqrt(np.average((d - self.mu) ** 2, weights=n))
        X, pen = self._design(c)
        self.w = newton(X, n, s, self.lam, pen, init=init)
        return self

    def predict(self, c):
        X, _ = self._design(c)
        return sigmoid(X @ self.w)

    def coefs(self):
        """(b0, b1) on the raw rating-point scale, per time control."""
        if self.structure == "pooled":
            b1 = self.w[1] / self.sd
            b0 = self.w[0] - b1 * self.mu
            return {"all": (b0, b1)}
        out = {}
        for i, t in enumerate(TCS):
            b1 = self.w[3 + i] / self.sd
            out[t] = (self.w[i] - b1 * self.mu, b1)
        return out
