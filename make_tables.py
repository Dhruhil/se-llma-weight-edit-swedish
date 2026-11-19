#!/usr/bin/env python
"""Turn results/results.json into a markdown table (results/summary.md) and a plot
(results/fidelity_vs_ppl.png). Run after run_experiment.py:  python make_tables.py"""
import json
import sys
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else "results/results.json")
R = json.loads(path.read_text(encoding="utf-8"))
T, sel = R["test"], R["selected"]


def pct(r):
    lo, hi = r["ci95"]
    return f"{100 * r['sv_rate']:.0f}% [{100 * lo:.0f}-{100 * hi:.0f}]"


rows = [("Baseline (unchanged)", T["baseline"]),
        ('Prompt "Svara alltid på svenska."', T["prompt_sv"])]
if "edit" in T:
    e = sel["edit"]
    rows.append((f"Rank-1 weight edit (layers {e['layers'][0]}-{e['layers'][-1]}, lambda={e['lam']})", T["edit"]))
if "steer" in T:
    s = sel["steer"]
    rows.append((f"Additive steering (layer {s['center']}, alpha={s['alpha']})", T["steer"]))

L = [f"Model: {R['config']['model_name']} @ {R.get('commit')}, seed {R['config']['seed']}, "
     f"n_test={R['n_test_prompts']}", "",
     "| Method | Swedish rate % [95% CI] | Swedish ppl | English ppl |", "|---|---|---|---|"]
for name, r in rows:
    L.append(f"| {name} | {pct(r)} | {r['ppl_sv']:.2f} | {r['ppl_en']:.2f} |")
if "random_edit" in T:
    x = T["random_edit"]
    L.append(f"| Random-direction edit (mean ± sd, n={len(x['runs'])}) | "
             f"{100 * x['sv_rate_mean']:.0f} ± {100 * x['sv_rate_sd']:.0f}% | "
             f"{x['ppl_sv_mean']:.2f} | {x['ppl_en_mean']:.2f} |")
L += ["", "Detected languages of test outputs:"]
for name, r in rows:
    L.append(f"- {name}: {r['lang_counts']}")
b = R["dev_baseline"]
L += ["", f"Dev baseline (for reference): Swedish {100 * b['sv_rate']:.0f}%, "
          f"Swedish ppl {b['ppl_sv']:.2f}, English ppl {b['ppl_en']:.2f}"]
out = path.parent / "summary.md"
out.write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for g in R["dev_grid"]:
        ax.scatter(g["ppl_sv"], 100 * g["sv_rate"], s=18, alpha=0.6,
                   marker="o" if g["method"] == "edit" else "^",
                   color="tab:blue" if g["method"] == "edit" else "tab:orange")
    ax.axvline(R["dev_baseline"]["ppl_sv"] * (1 + R["config"]["max_ppl_increase"]),
               ls="--", c="gray", lw=1)
    ax.scatter(R["dev_baseline"]["ppl_sv"], 100 * R["dev_baseline"]["sv_rate"],
               c="k", s=40, label="baseline")
    ax.scatter([], [], c="tab:blue", marker="o", label="weight edit (dev grid)")
    ax.scatter([], [], c="tab:orange", marker="^", label="steering (dev grid)")
    ax.set_xlabel("Swedish perplexity (dev)")
    ax.set_ylabel("Swedish responses (%) (dev)")
    ax.set_title("Dev grid: fidelity vs. perplexity (dashed = 10% limit)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path.parent / "fidelity_vs_ppl.png", dpi=160)
    print("saved plot")
except ImportError:
    print("matplotlib not installed: skipped plot (pip install matplotlib)")