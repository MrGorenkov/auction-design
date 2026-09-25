"""Figures from results/*.csv -> figures/*.png (300 dpi) and paper/figures/ copies."""
import csv
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # categorical slots 1-4 (validated palette)
MK = ["o", "s", "^", "D"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 6})


def rows(name):
    with open(RES / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig1_main():
    m = {r["rule"]: r for r in rows("main.csv")}
    soft = [(N, m[f"soft close {N} min"]) for N in (1, 2, 5, 10, 30)]
    x = [0] + [N for N, _ in soft]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for k, (col, lab) in enumerate((("revenue", "Revenue"), ("efficiency", "Allocative efficiency"))):
        y = [0.0] + [100 * float(r[f"d_{col}_vs_ref"]) for _, r in soft]
        e = [0.0] + [100 * float(r[f"d_{col}_vs_ref_ci"]) for _, r in soft]
        ax[k].errorbar(range(len(x)), y, yerr=e, color=C[0], marker=MK[0], capsize=3)
        ax[k].set_xticks(range(len(x)), ["hard"] + [f"{N}" for N in x[1:]])
        ax[k].set_xlabel("Closing rule: hard (platform) or soft, window N, min")
        ax[k].set_ylabel(f"{lab}: change vs platform rule,\npercentage points of max value" if k == 0 else
                         f"{lab}: change vs platform rule,\npercentage points")
        ax[k].axhline(0, color=MUTED, lw=0.8)
    fig.tight_layout()
    save(fig, "fig1_closing_rules.png")


def fig2_focal():
    rs = rows("focal_payoffs.csv")
    order = ["ArtSphere: hard close, ladder cap 50", "soft close 1 min", "soft close 5 min", "soft close 30 min"]
    labels = ["hard", "soft 1", "soft 5", "soft 30"]
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for k, bg in enumerate(("incremental-heavy", "baseline mix", "proxy-heavy")):
        d = {r["rule"]: r for r in rs if r["background"] == bg}
        y = [100 * float(d[o]["d_sniper_minus_proxy"]) for o in order]
        e = [100 * float(d[o]["d_sniper_minus_proxy_ci"]) for o in order]
        ax.errorbar(range(4), y, yerr=e, color=C[k], marker=MK[k], capsize=3, label=f"rivals: {bg}")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(range(4), labels)
    ax.set_xlabel("Closing rule (window, min)")
    ax.set_ylabel("Sniping premium: surplus of a sniper\nminus a truthful early proxy,\n% of max value")
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    save(fig, "fig2_sniping_premium.png")


def fig3_sensitivity():
    rs = [r for r in rows("sensitivity.csv") if r["rule"] == "soft close 5 min"]
    names = {"q": "lost late bids q", "sniper_share": "share of snipers", "react": "reaction time, min",
             "n": "number of bidders", "step": "bid step (value units)", "dist": "value distribution",
             "late_frac": "share of late first visits"}
    fig, ax = plt.subplots(figsize=(4.8, 6.4))
    ylab, y, k = [], 0, 0
    for par in names:
        sub = [r for r in rs if r["param"] == par]
        for r in sub:
            base = (par, r["value"]) in (("q", "0.2"), ("sniper_share", "0.2"), ("react", "10.0"), ("n", "6"),
                                          ("step", "0.01"), ("dist", "uniform"), ("late_frac", "0.0"))
            m, e = 100 * float(r["d_revenue_vs_hard"]), 100 * float(r["d_revenue_vs_hard_ci"])
            ax.errorbar(m, -y, xerr=e, fmt=MK[0], color=C[0] if not base else C[1], capsize=2, ms=4)
            ylab.append((-y, f"{names[par]} = " + {"beta_low": "beta(1, 3)"}.get(r["value"], r["value"])))
            y += 1
        y += 0.6
        k += 1
    ax.set_yticks([a for a, _ in ylab], [b for _, b in ylab], fontsize=7)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("Revenue: soft close 5 min minus hard close\n(no ladder cap), % of max value")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "fig3_sensitivity.png")


def fig4_cap():
    rs = [r for r in rows("cap_stress.csv") if r["rule"].startswith("ArtSphere")]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for k, n in enumerate(("2", "4", "8")):
        sub = sorted([r for r in rs if r["n"] == n], key=lambda r: float(r["step"]))
        xs = [100 * float(r["step"]) for r in sub]
        for j, (col, lab) in enumerate((("revenue", "Revenue loss"), ("efficiency", "Efficiency loss"))):
            ax[j].errorbar(xs, [-100 * float(r[f"d_{col}_vs_ref"]) for r in sub],
                           yerr=[100 * float(r[f"d_{col}_vs_ref_ci"]) for r in sub], color=C[k], marker=MK[k],
                           capsize=3, label=f"{n} bidders")
            ax[j].set_xscale("log")
            ax[j].set_xticks(xs, [f"{x:g}" for x in xs])
            ax[j].set_xlabel("Bid step, % of max value")
            ax[j].set_ylabel(f"{lab} from the 50-step cap,\npercentage points")
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    save(fig, "fig4_ladder_cap.png")


def fig5_llm():
    p = RES / "llm_plans.csv"
    if not p.exists():
        return
    rs = rows("llm_plans.csv")
    models = list(dict.fromkeys(r["model"] for r in rs))
    rules = ["hard", "soft5", "soft30"]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
    w = 0.8 / len(rules)
    for j, rule in enumerate(rules):
        d = {r["model"]: r for r in rs if r["rule"] == rule}
        xs = [i + (j - 1) * w for i in range(len(models))]
        ax[0].bar(xs, [100 * float(d[m]["late_share"]) for m in models], width=w * 0.9, color=C[j],
                  yerr=[100 * float(d[m]["late_share_ci"]) for m in models], capsize=2,
                  error_kw={"ecolor": MUTED, "elinewidth": 1},
                  label={"hard": "hard close", "soft5": "soft 5 min", "soft30": "soft 30 min"}[rule])
        ax[1].errorbar(xs, [float(d[m]["bid_value_median"]) for m in models],
                       yerr=[[float(d[m]["bid_value_median"]) - float(d[m]["bid_value_q25"]) for m in models],
                             [float(d[m]["bid_value_q75"]) - float(d[m]["bid_value_median"]) for m in models]],
                       fmt=MK[j], color=C[j], capsize=2)
    lab = [m.replace("qwen", "").replace("b", "B") + ("*" if next(x for x in rs if x["model"] == m)["pilot"] == "True"
                                                               else "") for m in models]
    for a in ax:
        a.set_xticks(range(len(models)), lab)
        a.set_xlabel("Qwen3.5 model size")
    ax[0].set_ylabel("Chose last-second timing, %")
    ax[1].set_ylabel("Maximum bid / value\n(median, interquartile range)")
    ax[1].axhline(1.0, color=MUTED, lw=0.8)
    fig.legend(*ax[0].get_legend_handles_labels(), loc="upper center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    save(fig, "fig5_llm_plans.png")


def main():
    fig1_main()
    fig2_focal()
    fig3_sensitivity()
    fig4_cap()
    fig5_llm()
    dst = ROOT / "paper" / "figures"
    dst.mkdir(parents=True, exist_ok=True)
    for f in FIG.glob("*.png"):
        shutil.copy(f, dst / f.name)


if __name__ == "__main__":
    main()
