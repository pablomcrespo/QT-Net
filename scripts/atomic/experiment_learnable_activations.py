#!/usr/bin/env python3
"""Simple ablation: static SiLU vs. soft dynActivation (arXiv:2603.22154).

Trains an atomic QT-Net model twice under an otherwise identical setup -- once
with the network's usual static SiLU activations, once with the paper's soft
(SiLU/Swish-based, C-infinity) ``dynActivation`` in which every activation site
gets two learnable scalars ``alpha, beta`` and computes

    SiLU(x) * (alpha - beta) + beta * x     (init alpha=1, beta=0 == SiLU).

The only thing that changes between the two runs is
``dynamic_activations.set_dynamic_activations(...)`` at construction time, so
any difference in validation error is attributable to making the activations
learnable.

This is intentionally a *small* experiment (subsampled folds, CPU-friendly):
it is meant to indicate whether learnable activations are worth pursuing on the
QTAIM atomic task, not to produce paper-grade numbers.

Usage:
    python experiment_learnable_activations.py --model EGNN --max-neighbors 12
    python experiment_learnable_activations.py --model SGNN --max-neighbors 12 \
        --n-train 2000 --n-val 500 --epochs 120 --seeds 0 1 2
"""

import os
import sys
import json
import time
import pickle
import argparse

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from flax import nnx

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
SRC = os.path.join(REPO_ROOT, "src")
for p in (REPO_ROOT, SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

from qtnet.jax_models import dynamic_activations as da
from qtnet.jax_models.models_scalar import ScalarGNN
from qtnet.jax_models.models_equivariant import EquivariantGNN
from qtnet.data_utils import (
    create_cv_splits,
    compute_molecular_stats,
    compute_per_atom_stats,
    apply_z_regularization,
)
from qtnet.jax_models.representations import (
    precompute_complexes,
    prepare_padded_batches,
)
from qtnet.jax_models.train_utils import (
    make_optimizer,
    train_multitask,
    count_parameters,
)

ALL_ELEMENTS = ["H", "C", "N", "O"]
TARGET_COLUMNS = [
    "N", "LI", "Mu_X", "Mu_Y", "Mu_Z",
    "Q_XY", "Q_XZ", "Q_YZ", "Q_aniso", "Q_ZZ",
]
MODELS = {"EGNN": EquivariantGNN, "SGNN": ScalarGNN}

DATA_PKL = os.path.join(
    REPO_ROOT, "data_curation", "atomic", "cluster_analysis", "train_and_val_compat.pkl"
)


def build_subset(args):
    """Load data, take fold-0 scaffold split, subsample deterministically."""
    df = pd.read_pickle(DATA_PKL)
    fold0 = next(f for f in create_cv_splits(df, group_col="Murcko_Scaffold")
                 if f["fold"] == 0)
    train_idx, val_idx = fold0["train_idx"], fold0["test_idx"]

    rng = np.random.default_rng(1234)
    if args.n_train and len(train_idx) > args.n_train:
        train_idx = np.sort(rng.choice(train_idx, args.n_train, replace=False))
    if args.n_val and len(val_idx) > args.n_val:
        val_idx = np.sort(rng.choice(val_idx, args.n_val, replace=False))

    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()
    return train_df, val_df


def prepare(args, train_df, val_df):
    """Precompute complexes (cached) and build padded, regularized batches."""
    element_to_idx = {e: i for i, e in enumerate(ALL_ELEMENTS)}
    sub_df = pd.concat([train_df, val_df])

    cache = os.path.join(
        REPO_ROOT, "data_curation", "atomic",
        f"expt_complexes_max{args.max_neighbors}_cut{args.cutoff}"
        f"_n{len(sub_df)}.pkl",
    )
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            complexes = pickle.load(f)
    else:
        complexes = precompute_complexes(
            sub_df, element_to_idx=element_to_idx,
            cutoff=args.cutoff, max_neighbors=args.max_neighbors,
            fully_connected=False, max_dim=2, output_path=cache, verbose=True,
        )

    molecular_stats = compute_molecular_stats(train_df)
    atomic_stats = compute_per_atom_stats(train_df)
    reg_train = apply_z_regularization(train_df.copy(), molecular_stats, atomic_stats)
    reg_val = apply_z_regularization(val_df.copy(), molecular_stats, atomic_stats)

    train_batches = prepare_padded_batches(
        complexes, reg_train, TARGET_COLUMNS,
        batch_size=args.batch_size, verbose=False, as_numpy=True)
    val_batches = prepare_padded_batches(
        complexes, reg_val, TARGET_COLUMNS,
        batch_size=args.batch_size, verbose=False, as_numpy=True)

    element_weight_array = jnp.array(
        [float(atomic_stats.at[el, "weight"]) for el in ALL_ELEMENTS],
        dtype=jnp.float32)
    return train_batches, val_batches, element_weight_array


def run_one(cls, dynamic, seed, args, train_batches, val_batches, weights):
    """Train a single model and return its validation history + metadata."""
    da.set_dynamic_activations(dynamic)
    model = cls(num_species=len(ALL_ELEMENTS), rngs=nnx.Rngs(seed))
    da.set_dynamic_activations(False)  # reset global immediately after build
    n_params = count_parameters(model)

    tx = make_optimizer(lr=args.lr, weight_decay=args.weight_decay,
                        use_reduce_on_plateau=True,
                        plateau_accumulation_size=len(train_batches))
    optimizer = nnx.Optimizer(model, tx)

    t0 = time.perf_counter()
    _, val_hist = train_multitask(
        model=model, optimizer=optimizer,
        train_batches=train_batches, val_batches=val_batches,
        epochs=args.epochs, verbose=args.verbose, print_every=args.print_every,
        disable_saving=True, element_weight_array=weights,
    )
    dt = time.perf_counter() - t0

    best_epoch = int(np.argmin(val_hist["total"]))
    return {
        "dynamic": dynamic,
        "seed": seed,
        "n_params": n_params,
        "train_time_s": round(dt, 1),
        "best_epoch": best_epoch,
        "best_val_total": float(val_hist["total"][best_epoch]),
        "best_val_by_prop": {k: float(val_hist[k][best_epoch])
                             for k in ("N", "LI", "Mu", "Q")},
        "final_val_total": float(val_hist["total"][-1]),
        "val_total_curve": [float(v) for v in val_hist["total"]],
    }


def _save(out, args, train_df, val_df, results):
    """Write (or overwrite) the summary JSON from the runs collected so far."""
    def agg(dynamic):
        vals = [r["best_val_total"] for r in results if r["dynamic"] == dynamic]
        if not vals:
            return None, None
        return float(np.mean(vals)), float(np.std(vals))

    base_mean, base_std = agg(False)
    dyn_mean, dyn_std = agg(True)
    improvement = None
    if base_mean is not None and dyn_mean is not None and base_mean != 0:
        improvement = 100.0 * (base_mean - dyn_mean) / base_mean

    summary = {
        "model": args.model,
        "max_neighbors": args.max_neighbors,
        "cutoff": args.cutoff,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "epochs": args.epochs,
        "seeds": args.seeds,
        "static_silu_best_val": {"mean": base_mean, "std": base_std},
        "soft_dynact_best_val": {"mean": dyn_mean, "std": dyn_std},
        "relative_improvement_pct": improvement,
        "runs": results,
    }
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--max-neighbors", type=int, default=12)
    ap.add_argument("--cutoff", type=float, default=5.25)
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-val", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--print-every", type=int, default=1)
    ap.add_argument("--out-dir", default=os.path.join(
        REPO_ROOT, "experiments", "learnable_activations"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cls = MODELS[args.model]

    print(f"=== {args.model}: static SiLU vs soft dynActivation "
          f"(max_neighbors={args.max_neighbors}) ===")
    train_df, val_df = build_subset(args)
    print(f"train molecules: {len(train_df)}  val molecules: {len(val_df)}")
    train_batches, val_batches, weights = prepare(args, train_df, val_df)
    print(f"train batches: {len(train_batches)}  val batches: {len(val_batches)}"
          f"  epochs: {args.epochs}")

    out = os.path.join(args.out_dir, f"{args.model}_max{args.max_neighbors}.json")

    # Resume: keep any runs already completed in a previous (possibly
    # interrupted) invocation so a container restart doesn't lose progress.
    results = []
    done = set()
    if os.path.exists(out):
        try:
            with open(out) as f:
                prev = json.load(f)
            # only resume when the stored run used the same configuration
            same_cfg = (prev.get("epochs") == args.epochs
                        and prev.get("n_train") == len(train_df)
                        and prev.get("n_val") == len(val_df)
                        and prev.get("max_neighbors") == args.max_neighbors)
            if same_cfg:
                for r in prev.get("runs", []):
                    results.append(r)
                    done.add((r["seed"], r["dynamic"]))
                if results:
                    print(f"resuming: {len(results)} run(s) already complete")
        except (json.JSONDecodeError, KeyError):
            pass

    for seed in args.seeds:
        for dynamic in (False, True):
            if (seed, dynamic) in done:
                continue
            tag = "dynAct" if dynamic else "SiLU  "
            r = run_one(cls, dynamic, seed, args,
                        train_batches, val_batches, weights)
            results.append(r)
            print(f"  seed {seed} | {tag} | params={r['n_params']:>7,} | "
                  f"best_val={r['best_val_total']:.4f} @ep{r['best_epoch']:>3} | "
                  f"{r['train_time_s']}s")
            _save(out, args, train_df, val_df, results)  # checkpoint after each run

    summary = _save(out, args, train_df, val_df, results)
    b = summary["static_silu_best_val"]
    d = summary["soft_dynact_best_val"]
    imp = summary["relative_improvement_pct"]
    print(f"\n  static SiLU     best val: {b['mean']:.4f} ± {b['std']:.4f}")
    print(f"  soft dynAct     best val: {d['mean']:.4f} ± {d['std']:.4f}")
    if imp is not None:
        print(f"  relative improvement    : {imp:+.2f}%  "
              f"({'dynAct better' if imp > 0 else 'SiLU better'})")
    print(f"  saved -> {out}")


if __name__ == "__main__":
    main()
