#!/usr/bin/env python3
"""Plot & tabulate the static-SiLU vs soft-dynActivation ablation.

Reads the JSON summaries written by
``scripts/atomic/experiment_learnable_activations.py`` and produces:

  * a two-panel validation-loss curve figure (one panel per model), with the
    per-seed mean drawn solid and a +/-1 std band, and
  * a short text table of best validation loss (mean +/- std over seeds) and
    the relative improvement of soft dynActivation over static SiLU.
"""

import os
import sys
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXP_DIR = os.path.join(REPO_ROOT, "experiments", "learnable_activations")

SILU_C = "#4C78A8"
DYN_C = "#E45756"


def curves(runs, dynamic):
    cs = [r["val_total_curve"] for r in runs if r["dynamic"] == dynamic]
    if not cs:
        return None, None
    n = min(len(c) for c in cs)
    arr = np.array([c[:n] for c in cs])
    return arr.mean(0), arr.std(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["SGNN", "EGNN"])
    ap.add_argument("--max-neighbors", type=int, default=12)
    ap.add_argument("--out", default=os.path.join(EXP_DIR, "learnable_activations.png"))
    args = ap.parse_args()

    summaries = {}
    for m in args.models:
        p = os.path.join(EXP_DIR, f"{m}_max{args.max_neighbors}.json")
        if os.path.exists(p):
            with open(p) as f:
                summaries[m] = json.load(f)

    if not summaries:
        print("No result JSONs found in", EXP_DIR)
        sys.exit(1)

    # ---- text table ----
    print(f"\n{'model':<6} {'nbrs':>4} {'static SiLU':>18} {'soft dynAct':>18} "
          f"{'rel. impr.':>11}")
    print("-" * 62)
    for m, s in summaries.items():
        b, d = s["static_silu_best_val"], s["soft_dynact_best_val"]
        imp = s.get("relative_improvement_pct")
        imp_s = f"{imp:+.2f}%" if imp is not None else "n/a"
        bs = f"{b['mean']:.4f}±{b['std']:.4f}" if b["mean"] is not None else "n/a"
        ds = f"{d['mean']:.4f}±{d['std']:.4f}" if d["mean"] is not None else "n/a"
        print(f"{m:<6} {s['max_neighbors']:>4} {bs:>18} {ds:>18} {imp_s:>11}")
    print()

    # ---- figure ----
    n = len(summaries)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 4.4), squeeze=False)
    for ax, (m, s) in zip(axes[0], summaries.items()):
        runs = s["runs"]
        tail_max = 0.0  # y-limit that focuses on the informative region
        for dyn, color, lab in [(False, SILU_C, "static SiLU"),
                                (True, DYN_C, "soft dynActivation")]:
            mean, std = curves(runs, dyn)
            if mean is None:
                continue
            x = np.arange(1, len(mean) + 1)
            ax.plot(x, mean, color=color, lw=2, label=lab)
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18)
            # skip the first few epochs (huge initial loss) when setting ylim
            if len(mean) > 10:
                tail_max = max(tail_max, float(np.max((mean + std)[10:])))
        if tail_max > 0:
            ax.set_ylim(top=tail_max * 1.05)
        ax.set_title(f"{m}  (max_neighbors={s['max_neighbors']})")
        ax.set_xlabel("epoch")
        ax.set_ylabel("validation loss (total)")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        imp = s.get("relative_improvement_pct")
        if imp is not None:
            better = "dynAct better" if imp > 0 else "SiLU better"
            ax.text(0.97, 0.95, f"{imp:+.2f}%  ({better})",
                    transform=ax.transAxes, ha="right", va="top", fontsize=10,
                    bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))

    fig.suptitle("Static SiLU vs. soft dynActivation on QTAIM atomic targets "
                 "(fold 0, mean ± std over seeds)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.out, dpi=140)
    print("saved figure ->", args.out)


if __name__ == "__main__":
    main()
