"""Event-driven simulator of one timed ascending online auction with proxy bidding.

The bidding rules follow the ArtSphere back end (github.com/MrGorenkov/ArtSphere-Auction,
backend/Sources/App/Controllers/BidController.swift):
  * minimum next bid = max(current_bid + bid_step, starting_price);
  * a proxy ("auto-broker") holds a maximum; after every state change the server repeatedly lets the
    best non-leading proxy (highest maximum, ties -> earliest setting) bid exactly the minimum next bid;
  * the server loop stops after `maxLadderSteps = 50` steps per trigger (parameter `cap`; None = no cap);
  * the auction has a fixed end time (hard close). The soft close (extension by N minutes after a late bid)
    is the counterfactual rule studied here.

Money is measured in ticks of the bid step: prices are integers, values are floats (value / step).
Time is in minutes; the scheduled length is T.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import numpy as np

PROXY, INCR, SNIPER = 0, 1, 2
TYPE_NAMES = {PROXY: "proxy", INCR: "incremental", SNIPER: "sniper"}


@dataclass(frozen=True)
class Rules:
    name: str
    soft: float | None = None      # extension window N (minutes); None = hard close
    cap: int | None = None         # proxy ladder steps per trigger; ArtSphere uses 50
    proxy_allowed: bool = True
    T: float = 1440.0              # scheduled length, minutes
    L: float = 1.0                 # loss window: human bids placed less than L minutes before the close may fail
    q: float = 0.2                 # probability that a bid inside the loss window is not transmitted
    start: int = 1                 # starting price, ticks


@dataclass
class Bidders:
    """Everything random about one auction, drawn once and shared across rules (common random numbers)."""
    v: np.ndarray                  # values, ticks
    kind: np.ndarray               # PROXY / INCR / SNIPER, or -1 for an externally scripted bidder
    arrive: np.ndarray             # first visit, minutes
    tau: np.ndarray                # snipers: minutes before the close at which they bid
    react_mean: float              # incremental bidders: mean delay (minutes) of return after being outbid
    seed: int
    plan_max: np.ndarray | None = None   # optional: explicit proxy maximum (ticks) for scripted bidders
    plan_late: np.ndarray | None = None  # optional: scripted bidder acts at the end (True) or at arrival (False)
    streams: dict = field(default_factory=dict)

    def u(self, i):
        """Next uniform draw from bidder i's private stream (loss draws)."""
        return self._next(i, "u")

    def e(self, i):
        """Next exponential draw (mean react_mean) from bidder i's private stream (reaction delays)."""
        return -self.react_mean * math.log(1.0 - self._next(i, "e"))

    def _next(self, i, s):
        key = (i, s)
        if key not in self.streams:
            self.streams[key] = [np.random.default_rng([self.seed, i, ord(s)]).random(256), 0]
        arr, k = self.streams[key]
        if k >= len(arr):
            arr = np.concatenate([arr, np.random.default_rng([self.seed, i, ord(s), len(arr)]).random(len(arr))])
        self.streams[key] = [arr, k + 1]
        return float(arr[k])


def simulate(b: Bidders, r: Rules):
    """Run one auction. Returns a dict of outcome variables (prices in ticks)."""
    n = len(b.v)
    b.streams = {}
    E = r.T                                   # current close
    p, h = 0, -1                              # current bid (0 = no bid yet), leading bidder
    proxies = {}                              # bidder -> (max ticks, setting order)
    seq = [0]
    ev = []                                   # heap of (time, priority, seq, kind, bidder)
    pending_return = np.zeros(n, bool)
    snipe_state = ["" for _ in range(n)]      # "" / pending / done / failed
    human_times = []                          # (time, bidder) of successful human submissions
    first_human = np.full(n, np.inf)
    stats = {"auto_bids": 0, "manual_bids": 0, "lost": 0, "cap_hits": 0}
    kind = b.kind

    def push(t, k, i, prio=1):
        seq[0] += 1
        heapq.heappush(ev, (t, prio, seq[0], k, i))

    def nxt():
        return max(p + 1, r.start)

    def submit(t, i):
        """A human action at time t. Inside the loss window it fails with probability q."""
        if E - t < r.L and b.u(i) < r.q:
            stats["lost"] += 1
            return False
        human_times.append((t, i))
        first_human[i] = min(first_human[i], t)
        return True

    def extend(t):
        nonlocal E
        if r.soft is not None and E - t < r.soft:
            E = t + r.soft
            push(E, "END", -1, prio=0)
            for j in range(n):                # a sniper whose bid was lost tries again before the new close
                if snipe_state[j] == "failed":
                    snipe_state[j] = "pending"
                    push(max(t, E - b.tau[j]), "SNIPE", j)

    def notify(j, t):
        if j >= 0 and kind[j] == INCR and not pending_return[j]:
            pending_return[j] = True
            push(t + b.e(j), "RETURN", j)

    def ladder(t):
        """ArtSphere auto-broker loop, in closed form. After the first step only the two best proxies alternate."""
        nonlocal p, h
        if not proxies:
            return
        order = sorted(proxies.items(), key=lambda kv: (-kv[1][0], kv[1][1]))
        A = order[0][0]
        if h == A:
            if len(order) < 2:
                return
            first, second = order[1][0], A
        else:
            first = A
            second = order[1][0] if len(order) > 1 else None
        q1 = nxt()
        F = math.floor(proxies[first][0] - q1) + 1          # first can bid at steps k <= F (odd k)
        if F < 1:
            return
        if second is None:
            k_nat = 1
        else:
            S = math.floor(proxies[second][0] - q1) + 1      # second can bid at steps k <= S (even k)
            kf = F + 1 if (F + 1) % 2 == 1 else F + 2
            ks = max(S + 1, 2)
            ks = ks if ks % 2 == 0 else ks + 1
            k_nat = min(kf, ks) - 1
        K = k_nat if r.cap is None else min(k_nat, r.cap)
        if r.cap is not None and k_nat > r.cap:
            stats["cap_hits"] += 1
        prev = h
        p = q1 + K - 1
        h = first if K % 2 == 1 else second
        stats["auto_bids"] += K
        if prev != h:
            notify(prev, t)
        extend(t)

    def session(t, i, cap_max=None):
        """Incremental bidder: bids the minimum while outbid and the minimum does not exceed the value."""
        nonlocal p, h
        top = b.v[i] if cap_max is None else cap_max
        if h == i or nxt() > top:
            return
        if not submit(t, i):
            return
        while h != i and nxt() <= top:
            prev = h
            p, h = nxt(), i
            stats["manual_bids"] += 1
            notify(prev, t)
            extend(t)
            ladder(t)

    def set_proxy(t, i, mx):
        if not submit(t, i):
            return False
        proxies[i] = (mx, seq[0])
        seq[0] += 1
        ladder(t)
        return True

    push(E, "END", -1, prio=0)
    for i in range(n):
        if kind[i] == SNIPER or (kind[i] == -1 and b.plan_late[i]):
            snipe_state[i] = "pending"
            push(E - b.tau[i], "SNIPE", i)
        else:
            push(b.arrive[i], "ARRIVE", i)

    while ev:
        t, _, _, k, i = heapq.heappop(ev)
        if k == "END":
            if t == E:
                break
            continue
        if t >= E:
            continue
        if k == "ARRIVE":
            if kind[i] == PROXY:
                if r.proxy_allowed:
                    set_proxy(t, i, b.v[i])
                else:
                    session(t, i)
            elif kind[i] == INCR:
                session(t, i)
            else:                                           # scripted bidder acting early
                if r.proxy_allowed:
                    set_proxy(t, i, b.plan_max[i])
                else:
                    session(t, i, b.plan_max[i])
        elif k == "RETURN":
            pending_return[i] = False
            session(t, i)
        elif k == "SNIPE":
            if snipe_state[i] != "pending":
                continue
            if E - t > b.tau[i] + 1e-9:                     # close was extended: wait for the new close
                push(E - b.tau[i], "SNIPE", i)
                continue
            mx = b.v[i] if kind[i] == SNIPER else b.plan_max[i]
            if nxt() > mx:
                snipe_state[i] = "done"
                continue
            if r.proxy_allowed:
                ok = set_proxy(t, i, mx)
            else:
                before = len(human_times)
                session(t, i, mx)
                ok = len(human_times) > before
            snipe_state[i] = "done" if ok else "failed"

    vmax = float(b.v.max())
    sold = h >= 0
    late = [x for x in human_times if x[0] > r.T - 1.0]   # sniping window: last minute of the scheduled time
    return {
        "sold": sold,
        "price": p if sold else 0,
        "winner": h,
        "winner_kind": int(kind[h]) if sold else -9,
        "eff": float(b.v[h]) / vmax if sold else 0.0,
        "surplus": float(b.v[h] - p) if sold else 0.0,
        "close": E,
        "ext": E - r.T,
        "n_human": len(human_times),
        "n_late": len(late),
        "late_win": bool(sold and first_human[h] > r.T - 1.0),   # won by a bidder who entered in the last minute
        "misalloc_sniper": bool(sold and kind[h] == SNIPER and b.v[h] < vmax),
        **stats,
    }
