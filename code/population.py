"""Draw the bidders of one auction (independent private values) with a fixed seed per auction."""
import numpy as np

from auction import INCR, PROXY, SNIPER, Bidders

BASE = dict(n=6, shares=(0.4, 0.4, 0.2), dist="uniform", step=0.01, T=1440.0, react=10.0,
            late_frac=0.0, tau=(1 / 60, 20 / 60))


def draw(seed, n=6, shares=(0.4, 0.4, 0.2), dist="uniform", step=0.01, T=1440.0, react=10.0,
         late_frac=0.0, tau=(1 / 60, 20 / 60), kinds=None, focal=None):
    """Values are on [0, 1] (uniform) or lognormal scaled to median 0.5; `step` is the bid increment in value units.
    late_frac = share of non-sniper first visits drawn from the last hour instead of the whole auction."""
    rng = np.random.default_rng([seed, 7])
    if dist == "uniform":
        v = rng.random(n)
    elif dist == "lognormal":
        v = 0.5 * np.exp(0.5 * rng.standard_normal(n))
    elif dist == "beta_low":                                # many low values, few high ones
        v = rng.beta(1.0, 3.0, n)
    else:
        raise ValueError(dist)
    if kinds is None:
        kinds = rng.choice([PROXY, INCR, SNIPER], size=n, p=np.asarray(shares) / sum(shares))
    if focal is not None:                                   # bidder 0 is forced to use strategy `focal`
        kinds = np.array(kinds)
        kinds[0] = focal
    arrive = rng.uniform(0, T, n)
    late = rng.random(n) < late_frac
    arrive[late] = rng.uniform(T - 60, T, late.sum())
    tau_ = rng.uniform(tau[0], tau[1], n)
    return Bidders(v=v / step, kind=np.asarray(kinds), arrive=arrive, tau=tau_, react_mean=react, seed=int(seed))
