"""Correctness checks: revenue equivalence (analytic vs Monte Carlo) and the simulator's limiting cases.
Writes results/verification.csv. Seeds are fixed."""
import csv
from pathlib import Path

import numpy as np

import auction as A
import population as P
import test_engine

RES = Path(__file__).resolve().parents[1] / "results"


def myerson_uniform(n, r):
    """Expected revenue of an optimal-format auction with reserve r, n bidders, values U[0,1]."""
    return (n - 1) / (n + 1) + r**n - 2 * n * r ** (n + 1) / (n + 1)


def sealed_formats(n, draws=400_000, seed=11):
    """Second-price, first-price (equilibrium bid (n-1)/n v), English clock (= 2nd value), Dutch (= first-price)."""
    v = np.sort(np.random.default_rng([seed, n]).random((draws, n)), axis=1)
    spsb = v[:, -2]
    fpsb = (n - 1) / n * v[:, -1]
    return {"second-price sealed": spsb, "first-price sealed": fpsb, "English clock": spsb.copy(), "Dutch": fpsb.copy()}


def mc(sample):
    return float(sample.mean()), float(1.96 * sample.std(ddof=1) / np.sqrt(len(sample)))


def engine(n, rules, shares, R=20000, step=0.01, react=10.0, start_ticks=None):
    rev, eff = [], []
    for s in range(R):
        b = P.draw(100_000 + s, n=n, shares=shares, step=step, react=react)
        out = A.simulate(b, rules)
        rev.append(out["price"] * step)
        eff.append(out["eff"])
    return np.array(rev), np.array(eff)


def main():
    rows = []
    add = lambda check, n, theory, est, ci, note="": rows.append(
        {"check": check, "n": n, "theory": round(theory, 5), "estimate": round(est, 5), "ci95": round(ci, 5),
         "diff": round(est - theory, 5), "note": note})
    for n in (2, 3, 4, 6, 8, 10):
        th = (n - 1) / (n + 1)
        for name, x in sealed_formats(n).items():
            m, c = mc(x)
            add(f"revenue equivalence: {name}", n, th, m, c)
    for n in (2, 4, 6):
        for r in (0.25, 0.5):
            v = np.random.default_rng([12, n]).random((400_000, n))
            v.sort(axis=1)
            rev = np.where(v[:, -1] < r, 0.0, np.where(v[:, -2] < r, r, v[:, -2]))
            m, c = mc(rev)
            add("second-price sealed with reserve", n, myerson_uniform(n, r), m, c, f"reserve {r}")
    # simulator, only early proxies, no cap, no lost bids: a second-price auction on a 0.01 grid
    for n in (2, 4, 6, 8):
        rev, eff = engine(n, A.Rules("hard, no cap", q=0.0), (1, 0, 0))
        m, c = mc(rev)
        add("simulator: proxies only (hard close, no cap)", n, (n - 1) / (n + 1), m, c,
            f"grid 0.01; efficiency {eff.mean():.4f}")
    for n in (2, 4):
        rev, _ = engine(n, A.Rules("start price 0.5", q=0.0, start=50), (1, 0, 0), R=20000)
        m, c = mc(rev)
        add("simulator: proxies only, starting price 0.5 = reserve", n, myerson_uniform(n, 0.5), m, c, "grid 0.01")
    # incremental bidders who react almost instantly: an English auction
    for n in (2, 4, 6):
        rev, eff = engine(n, A.Rules("hard, no cap", q=0.0), (0, 1, 0), R=5000, react=0.01)
        m, c = mc(rev)
        add("simulator: incremental bidders, reaction 0.01 min (English)", n, (n - 1) / (n + 1), m, c,
            f"grid 0.01; efficiency {eff.mean():.4f}")
    # ending rule is irrelevant when everyone bids early by proxy
    for n in (4,):
        h, _ = engine(n, A.Rules("hard", q=0.0), (1, 0, 0), R=5000)
        s, _ = engine(n, A.Rules("soft", soft=10.0, q=0.0), (1, 0, 0), R=5000)
        add("simulator: proxies only, soft minus hard close (paired)", n, 0.0, float((s - h).mean()),
            float(np.abs(s - h).max()), "ci95 column = max absolute paired difference")
    bad = test_engine.test_ladder(trials=20000)
    add("closed-form ladder vs literal ArtSphere loop (mismatches in 20000 random states)", 0, 0, bad, 0)
    RES.mkdir(exist_ok=True)
    with open(RES / "verification.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
