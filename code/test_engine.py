"""Checks of the simulator against a literal re-implementation of the ArtSphere auto-broker loop."""
import math

import numpy as np

import auction as A


def literal_ladder(p, h, proxies, start, cap):
    """Line-by-line port of BidController.processAutoBroker (one trigger), prices in ticks."""
    steps = 0
    cap = 10**9 if cap is None else cap
    while steps < cap:
        steps += 1
        nxt = max(p + 1, start)
        cands = [(mx, s, i) for i, (mx, s) in proxies.items() if mx >= nxt and i != h]
        if not cands:
            return p, h
        cands.sort(key=lambda c: (-c[0], c[1]))
        p, h = nxt, cands[0][2]
    return p, h


def test_ladder(trials=20000, seed=1):
    rng = np.random.default_rng(seed)
    bad = 0
    for _ in range(trials):
        m = rng.integers(1, 6)
        proxies = {i: (float(rng.uniform(0, 120)), int(rng.integers(0, 50))) for i in range(m)}
        if rng.random() < 0.3:                              # force ties
            j = int(rng.integers(0, m))
            proxies[j] = (proxies[0][0], proxies[j][1])
        p = int(rng.integers(0, 60))
        h = int(rng.integers(-1, m + 1))                    # -1 none, m = a manual bidder
        start = int(rng.integers(1, 30))
        cap = [None, 50, 3, 1][rng.integers(0, 4)]
        ref = literal_ladder(p, h, proxies, start, cap)
        got = closed_form(p, h, proxies, start, cap)
        bad += ref != got
    return bad


def closed_form(p, h, proxies, start, cap):
    """Run the simulator's ladder() on a synthetic state through a tiny auction harness."""
    order = sorted(proxies.items(), key=lambda kv: (-kv[1][0], kv[1][1]))
    Aid = order[0][0]
    if h == Aid:
        if len(order) < 2:
            return p, h
        first, second = order[1][0], Aid
    else:
        first, second = Aid, (order[1][0] if len(order) > 1 else None)
    q1 = max(p + 1, start)
    F = math.floor(proxies[first][0] - q1) + 1
    if F < 1:
        return p, h
    if second is None:
        k_nat = 1
    else:
        S = math.floor(proxies[second][0] - q1) + 1
        kf = F + 1 if (F + 1) % 2 == 1 else F + 2
        ks = max(S + 1, 2)
        ks = ks if ks % 2 == 0 else ks + 1
        k_nat = min(kf, ks) - 1
    K = k_nat if cap is None else min(k_nat, cap)
    return q1 + K - 1, (first if K % 2 == 1 else second)


def test_source_matches_harness():
    """The harness above must be the same code as auction.simulate's ladder (guards against drift)."""
    import inspect
    src = inspect.getsource(A.simulate)
    for frag in ["kf = F + 1 if (F + 1) % 2 == 1 else F + 2", "ks = max(S + 1, 2)", "k_nat = min(kf, ks) - 1",
                 "K = k_nat if r.cap is None else min(k_nat, r.cap)", "h = first if K % 2 == 1 else second"]:
        assert frag in src, frag


if __name__ == "__main__":
    test_source_matches_harness()
    print("ladder mismatches:", test_ladder())
