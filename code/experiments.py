"""Agent-based experiments: ending rules x proxy ladder cap, sensitivity, cap stress test, payoffs of a focal bidder by strategy.
Writes results/*.csv. Common random numbers: auction s uses the same bidders under every rule."""
import csv
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import auction as A
import population as P

RES = Path(__file__).resolve().parents[1] / "results"
SEED0 = 2026_09_25
R_MAIN, R_SENS = 20000, 5000

ARTSPHERE = A.Rules("ArtSphere: hard close, ladder cap 50", cap=50)
HARD = A.Rules("hard close, no cap")
SOFT = {N: A.Rules(f"soft close {N:g} min", soft=float(N)) for N in (1, 2, 5, 10, 30)}
SOFT5_CAP = A.Rules("soft close 5 min, ladder cap 50", soft=5.0, cap=50)
MAIN_RULES = [ARTSPHERE, HARD] + list(SOFT.values()) + [SOFT5_CAP]


def one(args):
    s, pop, rules = args
    b = P.draw(SEED0 + s, **pop)
    step = pop.get("step", 0.01)
    rows = []
    for r in rules:
        o = A.simulate(b, r)
        kinds = b.kind
        surplus = np.zeros(3)
        if o["sold"]:
            surplus[kinds[o["winner"]]] = o["surplus"] * step
        rows.append([o["price"] * step, o["eff"], o["ext"], o["n_late"] / max(o["n_human"], 1), o["late_win"],
                     o["winner_kind"] == A.SNIPER, o["misalloc_sniper"], o["cap_hits"] > 0, o["sold"],
                     *surplus, *[(kinds == k).sum() for k in range(3)]])
    return rows


COLS = ["revenue", "efficiency", "extension", "late_share", "late_win", "sniper_win", "sniper_misalloc", "cap_hit",
        "sold", "surplus_proxy", "surplus_incr", "surplus_sniper", "n_proxy", "n_incr", "n_sniper"]


def run(pop, rules, R, pool):
    out = pool.map(one, [(s, pop, rules) for s in range(R)], chunksize=200)
    return np.array(out, dtype=float)                     # R x rules x cols


def ci(x):
    return float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(len(x)))


def summarise(arr, rules, ref=0, extra=None):
    rows = []
    for j, r in enumerate(rules):
        row = {"rule": r.name, **(extra or {})}
        for c, name in enumerate(COLS[:9]):
            m, h = ci(arr[:, j, c])
            row[name], row[name + "_ci"] = round(m, 5), round(h, 5)
        refs = [("ref", ref)] + [("hard", i) for i, x in enumerate(rules) if x.name == HARD.name and i != ref]
        for tag, rj in refs:
            for c, name in ((0, "revenue"), (1, "efficiency")):
                m, h = ci(arr[:, j, c] - arr[:, rj, c])
                row[f"d_{name}_vs_{tag}"], row[f"d_{name}_vs_{tag}_ci"] = round(m, 5), round(h, 5)
        row["ext_p95"] = round(float(np.percentile(arr[:, j, 2], 95)), 3)
        for k, t in enumerate(("proxy", "incr", "sniper")):          # mean surplus per bidder of each type
            tot, cnt = arr[:, j, 9 + k].sum(), arr[:, j, 12 + k].sum()
            row[f"payoff_{t}"] = round(tot / cnt, 5) if cnt else ""
        rows.append(row)
    return rows


def write(name, rows):
    RES.mkdir(exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(RES / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def sensitivity(pool):
    base = dict(P.BASE)
    rules = [ARTSPHERE, HARD, SOFT[5]]
    grid = [("q", [0.0, 0.1, 0.2, 0.4, 0.6]), ("sniper_share", [0.0, 0.1, 0.2, 0.4, 0.6]),
            ("react", [1.0, 5.0, 10.0, 30.0, 60.0]), ("n", [2, 3, 4, 6, 8, 12]),
            ("step", [0.001, 0.005, 0.01, 0.02, 0.05]), ("dist", ["uniform", "lognormal", "beta_low"]),
            ("late_frac", [0.0, 0.3, 0.6])]
    rows = []
    for par, vals in grid:
        for val in vals:
            pop, rl = dict(base), rules
            if par == "q":
                rl = [A.Rules(r.name, soft=r.soft, cap=r.cap, q=val) for r in rules]
            elif par == "sniper_share":
                pop["shares"] = ((1 - val) / 2, (1 - val) / 2, val)
            else:
                pop[par] = val
            arr = run(pop, rl, R_SENS, pool)
            rows += summarise(arr, rl, extra={"param": par, "value": val})
            print(par, val, [round(arr[:, j, 0].mean(), 4) for j in range(len(rl))], flush=True)
    write("sensitivity.csv", rows)


def cap_stress(pool):
    """Proxies only, arriving over the auction: the ArtSphere 50-step cap versus an uncapped ladder."""
    rows = []
    rules = [ARTSPHERE, HARD]
    for step in (0.001, 0.0025, 0.005, 0.01):
        for n in (2, 4, 8):
            pop = dict(P.BASE, shares=(1, 0, 0), step=step, n=n)
            arr = run(pop, rules, R_SENS, pool)
            rows += summarise(arr, rules, ref=1, extra={"step": step, "n": n})
    write("cap_stress.csv", rows)


def focal_payoffs(pool, R=R_MAIN):
    """Expected surplus of one focal bidder (bidder 0) who uses strategy k while the other bidders are drawn from a
    background mix. Same auctions (values, arrivals) for every k and every rule, so differences are paired."""
    backgrounds = {"baseline mix": (0.4, 0.4, 0.2), "incremental-heavy": (0.1, 0.8, 0.1),
                   "proxy-heavy": (0.8, 0.1, 0.1)}
    rules = [ARTSPHERE, HARD, SOFT[1], SOFT[5], SOFT[30], A.Rules("hard close, no cap, q = 0", q=0.0)]
    rows = []
    for bname, mix in backgrounds.items():
        res = {}
        for k in (A.PROXY, A.INCR, A.SNIPER):
            pop = dict(P.BASE, shares=mix, focal=k)
            res[k] = pool.map(focal_one, [(s, pop, rules) for s in range(R)], chunksize=200)
            res[k] = np.array(res[k])                     # R x rules x (surplus of bidder 0, won, price)
        for j, r in enumerate(rules):
            row = {"background": bname, "rule": r.name}
            for k, t in ((A.PROXY, "proxy"), (A.INCR, "incr"), (A.SNIPER, "sniper")):
                m, h = ci(res[k][:, j, 0])
                row[f"pay_{t}"], row[f"pay_{t}_ci"] = round(m, 5), round(h, 5)
                row[f"win_{t}"] = round(float(res[k][:, j, 1].mean()), 4)
                row[f"revenue_{t}"] = round(float(res[k][:, j, 2].mean()), 5)
            for a, b_ in (("sniper", A.SNIPER), ("incr", A.INCR)):
                m, h = ci(res[b_][:, j, 0] - res[A.PROXY][:, j, 0])
                row[f"d_{a}_minus_proxy"], row[f"d_{a}_minus_proxy_ci"] = round(m, 5), round(h, 5)
            rows.append(row)
            print(row, flush=True)
    write("focal_payoffs.csv", rows)


def focal_one(args):
    s, pop, rules = args
    b = P.draw(SEED0 + 5_000_000 + s, **pop)
    step = pop.get("step", 0.01)
    out = []
    for r in rules:
        o = A.simulate(b, r)
        won = o["sold"] and o["winner"] == 0
        out.append([o["surplus"] * step if won else 0.0, float(won), o["price"] * step])
    return out


def main(which):
    with Pool(4) as pool:
        if "main" in which:
            arr = run(dict(P.BASE), MAIN_RULES, R_MAIN, pool)
            write("main.csv", summarise(arr, MAIN_RULES))
            for r in summarise(arr, MAIN_RULES):
                print(r["rule"], r["revenue"], r["efficiency"], r["d_revenue_vs_ref"], r["extension"], flush=True)
        if "sens" in which:
            sensitivity(pool)
        if "cap" in which:
            cap_stress(pool)
        if "focal" in which:
            focal_payoffs(pool)


if __name__ == "__main__":
    main(sys.argv[1:] or ["main", "sens", "cap", "focal"])
