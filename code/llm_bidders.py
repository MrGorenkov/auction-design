"""Elicit bidding plans from a local LLM (GGUF via llama.cpp llama-server) for the timed proxy auction.

Each prompt gives the bidder's private value, the ArtSphere rules (proxy agent, bid step, lost late bids) and one
closing rule; the model answers with JSON {"timing": "early" | "last_seconds", "max_bid": x}.
Decoding: temperature 0.7, top_p 0.95, fixed seed per prompt, thinking mode off, at most 120 new tokens.

Usage: python code/llm_bidders.py <gguf path> <label> [ngl] [repeats]
Env: LLAMA_SERVER (path to llama-server), RESULTS_DIR (default: results/llm).
Writes <RESULTS_DIR>/decisions_<label>.jsonl (one line per prompt, raw answer included).
"""
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIN = os.environ.get("LLAMA_SERVER", str(Path.home() / "llama.cpp/build/bin/llama-server"))
OUT = Path(os.environ.get("RESULTS_DIR", ROOT / "results" / "llm"))
PORT = 8093
SCALE = 10.0                                          # 1 value unit = 10 TON
VALUES = [round(0.05 + 0.08 * k, 2) for k in range(12)]  # 0.05 ... 0.93 value units -> 0.5 ... 9.3 TON
RULES = {"hard": None, "soft5": 5, "soft30": 30}
TEMPERATURE, TOP_P, MAX_TOKENS = 0.7, 0.95, 120

SYSTEM = "You are a participant in an online art auction. Answer only with a JSON object."
CLOSE = {
    None: "The auction closes exactly at the scheduled end time, 24 hours after the start. Bids that arrive after "
          "that moment are rejected.",
    "soft": "If any bid is placed in the last {N} minutes before the close, the close is moved so that the auction "
            "ends only when {N} minutes have passed without a new bid.",
}


def prompt(value_ton, soft):
    close = CLOSE[None] if soft is None else CLOSE["soft"].format(N=soft)
    return f"""You want to buy a digital artwork on an online auction platform. The artwork is worth {value_ton:.1f} TON to you: if you win and pay a price P, your gain is {value_ton:.1f} - P TON; if you lose, your gain is 0. You want to maximise your expected gain.

Rules of this auction:
- It is an ascending auction that lasts 24 hours. The current price is 0.1 TON and the bid step is 0.1 TON. The highest bidder when the auction closes wins and pays the current price.
- You give a maximum bid to an automatic bidding agent. The agent bids for you only as much as needed to stay the highest bidder (one bid step above the best competitor), up to your maximum. Other bidders never see your maximum.
- {close}
- A bid submitted in the last minute before the close fails to reach the server with probability 20% because of network delays.

Competition: there are 5 other bidders. You do not know their values; each value is equally likely to be anywhere between 0 and 10 TON. Some of them give their maximum early, some bid the minimum and come back to raise their bid (usually within about 10 minutes) after being outbid, and some wait and bid in the last seconds.

Decide:
1. timing: "early" (give your maximum to the agent now) or "last_seconds" (wait and give your maximum in the last seconds before the close);
2. max_bid: your maximum bid in TON.

Reply with JSON only, for example {{"timing": "early", "max_bid": 1.0}}"""


def chat(messages, seed):
    body = {"messages": messages, "temperature": TEMPERATURE, "top_p": TOP_P, "seed": seed, "max_tokens": MAX_TOKENS,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def parse(text):
    """Return (timing, max_bid) or (None, None) if the answer is not a usable JSON plan."""
    m = re.search(r"\{[^{}]*\}", text or "", re.S)
    if not m:
        return None, None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None, None
    t = str(d.get("timing", "")).strip().lower()
    t = "late" if t.startswith("last") or "late" in t else ("early" if t.startswith("early") else None)
    try:
        b = float(d.get("max_bid"))
    except (TypeError, ValueError):
        b = None
    return t, b


def wait_ready(proc, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise SystemExit("llama-server exited")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(2)
    raise SystemExit("llama-server did not start")


def main():
    gguf, label = Path(sys.argv[1]), sys.argv[2]
    ngl = sys.argv[3] if len(sys.argv) > 3 else "0"
    repeats = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"decisions_{label}.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(line)["id"] for line in out.read_text(encoding="utf-8").splitlines() if line.strip()}
    proc = subprocess.Popen([BIN, "-m", str(gguf), "-ngl", ngl, "-c", "4096", "-np", "1", "--port", str(PORT),
                             "--jinja", "-t", str(os.cpu_count())], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_ready(proc)
        with open(out, "a", encoding="utf-8") as f:
            for rule, soft in RULES.items():
                for v in VALUES:
                    for rep in range(repeats):
                        pid = f"{label}|{rule}|{v}|{rep}"
                        if pid in done:
                            continue
                        seed = 1000 * rep + int(round(v * 100)) + {"hard": 0, "soft5": 1, "soft30": 2}[rule] * 100000
                        t0 = time.time()
                        res = chat([{"role": "system", "content": SYSTEM},
                                    {"role": "user", "content": prompt(v * SCALE, soft)}], seed)
                        text = res["choices"][0]["message"].get("content", "")
                        timing, bid = parse(text)
                        rec = {"id": pid, "model": label, "rule": rule, "value": v, "value_ton": round(v * SCALE, 1),
                               "rep": rep, "seed": seed, "timing": timing, "max_bid_ton": bid, "raw": text,
                               "tokens": res.get("usage", {}).get("completion_tokens"), "sec": round(time.time() - t0, 2)}
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        f.flush()
                        print(pid, timing, bid, rec["sec"], flush=True)
    finally:
        proc.terminate()
        proc.wait()


if __name__ == "__main__":
    main()
