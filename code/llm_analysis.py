"""Summarise LLM bidding plans and evaluate them in the simulator.

Inputs: results/llm/decisions_<label>.jsonl (llm_bidders.py on CPU) and results/llm_kaggle/decisions_<label>.jsonl
(the same script on Kaggle GPUs, 10 repeats). Pilot models come from the CPU set, the others from the GPU set.
Outputs:
  results/llm_plans.csv   - share of valid answers, share choosing last-second timing, max_bid / value, by model x rule
  results/llm_payoffs.csv - expected surplus of the LLM plan as bidder 0 against the baseline mix of simulated rivals,
                            relative to a truthful early proxy and to a truthful sniper with the same value (paired)
  results/llm_market.csv  - auctions in which all six bidders follow plans of the same model (revenue, efficiency)
"""
import csv
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import auction as A
import population as P

ROOT = Path(__file__).resolve().parents[1]
LLM = ROOT / "results" / "llm"
KAGGLE = ROOT / "results" / "llm_kaggle"
RES = ROOT / "results"
RULE = {"hard": A.Rules("ArtSphere: hard close, ladder cap 50", cap=50),
        "soft5": A.Rules("soft close 5 min", soft=5.0), "soft30": A.Rules("soft close 30 min", soft=30.0)}
STEP, SCALE, R_FOCAL, R_MARKET = 0.01, 10.0, 400, 4000
SEED = 77_000_000
PILOT = {"qwen0.8b", "qwen2b"}                      # CPU pilot models (labelled as pilot in the paper)


def load():
    """Main set: CPU pilot answers for the models in PILOT (results/llm/) and the GPU (Kaggle) answers for the rest
    (results/llm_kaggle/). CPU answers of the larger models are used only for the CPU-GPU agreement check."""
    recs = []
    for f in sorted(LLM.glob("decisions_*.jsonl")):
        recs += [r for r in map(json.loads, f.read_text(encoding="utf-8").splitlines()) if r["model"] in PILOT]
    gpu = sorted(KAGGLE.glob("decisions_*.jsonl"))
    if not gpu:                                        # no Kaggle output: fall back to all CPU answers
        recs = [json.loads(x) for f in sorted(LLM.glob("decisions_*.jsonl"))
                for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
    for f in gpu:
        recs += [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
    for r in recs:
        r["valid"] = r["timing"] in ("early", "late") and r["max_bid_ton"] is not None and r["max_bid_ton"] >= 0
    return recs


def cpu_gpu_agreement():
    """Same prompt and seed on CPU (results/llm/) and GPU (results/llm_kaggle/): share of identical plans."""
    rows = []
    for f in sorted(KAGGLE.glob("decisions_*.jsonl")):
        cpu_f = LLM / f.name
        if not cpu_f.exists():
            continue
        g = {r["id"]: r for r in map(json.loads, f.read_text(encoding="utf-8").splitlines())}
        c = {r["id"]: r for r in map(json.loads, cpu_f.read_text(encoding="utf-8").splitlines())}
        common = sorted(set(g) & set(c))
        same_t = np.mean([g[k]["timing"] == c[k]["timing"] for k in common])
        same_b = np.mean([g[k]["max_bid_ton"] == c[k]["max_bid_ton"] for k in common])
        same = np.mean([g[k]["timing"] == c[k]["timing"] and g[k]["max_bid_ton"] == c[k]["max_bid_ton"] for k in common])
        late_c = np.mean([c[k]["timing"] == "late" for k in common])
        late_g = np.mean([g[k]["timing"] == "late" for k in common])
        rows.append({"model": f.stem.replace("decisions_", ""), "pairs": len(common), "same_timing": round(float(same_t), 4),
                     "same_max_bid": round(float(same_b), 4), "same_plan": round(float(same), 4),
                     "late_share_cpu": round(float(late_c), 4), "late_share_gpu": round(float(late_g), 4)})
    if rows:
        with open(RES / "llm_cpu_gpu_agreement.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return rows


def focal_job(args):
    """Bidder 0 has value v; rivals from the baseline mix. Returns surplus of bidder 0 under three plans."""
    rec, s = args
    b = P.draw(SEED + s, **P.BASE)
    v = rec["value"] / STEP
    b.v[0] = v
    b.plan_max = np.zeros(len(b.v))
    b.plan_late = np.zeros(len(b.v), bool)
    out = []
    for plan in ("llm", "proxy", "sniper"):
        kind = b.kind.copy()
        kind[0] = -1
        b.kind = kind
        if plan == "llm":
            b.plan_max[0], b.plan_late[0] = rec["max_bid_ton"] / SCALE / STEP, rec["timing"] == "late"
        else:
            b.plan_max[0], b.plan_late[0] = v, plan == "sniper"
        o = A.simulate(b, RULE[rec["rule"]])
        out.append((v - o["price"]) * STEP if o["sold"] and o["winner"] == 0 else 0.0)
    return out


def market_job(args):
    """Six bidders, each follows a plan drawn from the model's answers for the nearest value on the grid."""
    plans, grid, rule, s = args
    rng = np.random.default_rng([SEED, 9, s])
    b = P.draw(SEED + 1_000_000 + s, **P.BASE)
    n = len(b.v)
    b.kind = np.full(n, -1)
    b.plan_max, b.plan_late = np.zeros(n), np.zeros(n, bool)
    for i in range(n):
        g = grid[int(np.argmin(np.abs(grid - b.v[i] * STEP)))]
        cand = plans[g]
        p = cand[rng.integers(len(cand))]
        # the answer's max bid is rescaled to this bidder's value (same bid/value ratio as in the answer)
        b.plan_max[i] = b.v[i] * p[0]
        b.plan_late[i] = p[1]
    o = A.simulate(b, RULE[rule])
    return [o["price"] * STEP, o["eff"], o["ext"]]


def main():
    print(cpu_gpu_agreement())
    recs = load()
    order = ["qwen0.8b", "qwen2b", "qwen4b", "qwen9b", "qwen27b"]
    models = sorted({r["model"] for r in recs}, key=lambda m: order.index(m) if m in order else 99)
    plan_rows, pay_rows, mkt_rows = [], [], []
    with Pool(4) as pool:
        for m in models:
            for rule in RULE:
                rs = [r for r in recs if r["model"] == m and r["rule"] == rule]
                if not rs:
                    continue
                ok = [r for r in rs if r["valid"]]
                ratio = np.array([r["max_bid_ton"] / r["value_ton"] for r in ok]) if ok else np.array([np.nan])
                late = np.mean([r["timing"] == "late" for r in ok]) if ok else np.nan
                plan_rows.append({"model": m, "pilot": m in PILOT, "rule": rule, "answers": len(rs), "valid": len(ok),
                                  "late_share": round(float(late), 4),
                                  "late_share_ci": round(float(1.96 * np.sqrt(late * (1 - late) / max(len(ok), 1))), 4),
                                  "bid_value_median": round(float(np.median(ratio)), 4),
                                  "bid_value_q25": round(float(np.percentile(ratio, 25)), 4),
                                  "bid_value_q75": round(float(np.percentile(ratio, 75)), 4),
                                  "overbid_share": round(float(np.mean(ratio > 1.0)), 4),
                                  "mean_tokens": round(float(np.mean([r["tokens"] or 0 for r in rs])), 1),
                                  "mean_sec": round(float(np.mean([r["sec"] for r in rs])), 2)})
                if not ok:
                    continue
                jobs = [(r, s) for r in ok for s in range(R_FOCAL)]
                res = np.array(pool.map(focal_job, jobs, chunksize=100)).reshape(len(ok), R_FOCAL, 3)
                per = res.mean(axis=1)                        # per answer: llm, proxy, sniper
                d1, d2 = per[:, 0] - per[:, 1], per[:, 0] - per[:, 2]
                se = lambda x: 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
                pay_rows.append({"model": m, "pilot": m in PILOT, "rule": rule, "answers": len(ok),
                                 "pay_llm": round(float(per[:, 0].mean()), 5), "pay_proxy": round(float(per[:, 1].mean()), 5),
                                 "pay_sniper": round(float(per[:, 2].mean()), 5),
                                 "d_llm_minus_proxy": round(float(d1.mean()), 5), "d_llm_minus_proxy_ci": round(float(se(d1)), 5),
                                 "d_llm_minus_sniper": round(float(d2.mean()), 5), "d_llm_minus_sniper_ci": round(float(se(d2)), 5)})
                grid = np.array(sorted({r["value"] for r in ok}))
                plans = {g: [(r["max_bid_ton"] / r["value_ton"], r["timing"] == "late") for r in ok if r["value"] == g]
                         for g in grid}
                plans = {g: v for g, v in plans.items() if v}
                grid = np.array(sorted(plans))
                mk = np.array(pool.map(market_job, [(plans, grid, rule, s) for s in range(R_MARKET)], chunksize=200))
                mkt_rows.append({"model": m, "pilot": m in PILOT, "rule": rule,
                                 "revenue": round(float(mk[:, 0].mean()), 5),
                                 "revenue_ci": round(float(1.96 * mk[:, 0].std(ddof=1) / np.sqrt(len(mk))), 5),
                                 "efficiency": round(float(mk[:, 1].mean()), 5),
                                 "efficiency_ci": round(float(1.96 * mk[:, 1].std(ddof=1) / np.sqrt(len(mk))), 5),
                                 "extension": round(float(mk[:, 2].mean()), 3)})
                print(plan_rows[-1], pay_rows[-1], mkt_rows[-1], flush=True)
    # Fisher exact test: late-timing share under each soft close versus the hard close, same model
    from scipy.stats import fisher_exact
    for r in plan_rows:
        h = next(x for x in plan_rows if x["model"] == r["model"] and x["rule"] == "hard")
        if r["rule"] == "hard":
            r["p_late_vs_hard"] = ""
            continue
        a_, b_ = round(r["late_share"] * r["valid"]), round(h["late_share"] * h["valid"])
        r["p_late_vs_hard"] = float(f"{fisher_exact([[a_, r['valid'] - a_], [b_, h['valid'] - b_]])[1]:.3g}")
    for name, rows in (("llm_plans.csv", plan_rows), ("llm_payoffs.csv", pay_rows), ("llm_market.csv", mkt_rows)):
        if rows:
            with open(RES / name, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)


if __name__ == "__main__":
    main()
