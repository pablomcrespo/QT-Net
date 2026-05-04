#!/usr/bin/env python3
"""
Benchmark forward-pass FLOPs and wall-time for every trained atomic model.

Walks experiments/atomic/<MODEL>/fold_<N>/config.json, instantiates each model
fresh from the saved config (no checkpoint load — timing is weight-independent),
and runs the existing FLOPs / timing utilities on a representative batch.

Output: tests/benchmark_results.csv + a printed markdown table sorted by time.

Usage:
    python tests/benchmark_models.py
    python tests/benchmark_models.py --models EGNN SGN2 --fold 0
    python tests/benchmark_models.py --num-runs 50 --warmup 5
"""

import os
import sys
import json
import argparse
import pickle
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path bootstrap (mirror train_multitask.py)
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
PROJECT_ROOT = os.path.dirname(REPO_ROOT)

for p in (REPO_ROOT, os.path.join(REPO_ROOT, 'src'), os.path.join(REPO_ROOT, 'scripts', 'atomic')):
    if p not in sys.path:
        sys.path.insert(0, p)

import jax  # noqa: E402

from qtnet.data_utils import (  # noqa: E402
    compute_molecular_stats,
    compute_per_atom_stats,
    apply_z_regularization,
)
from qtnet.jax_models.representations import (  # noqa: E402
    precompute_complexes,
    prepare_padded_batches,
)
from qtnet.jax_models.optuna_hpo import (  # noqa: E402
    count_flops,
    benchmark_model,
    format_flops,
    format_time,
)
from qtnet.jax_models.train_utils import count_parameters  # noqa: E402

# Reuse model registry + factory from the training script
from train_multitask import (  # noqa: E402
    MODEL_CLASSES,
    create_model,
    get_fold,
    TARGET_COLUMNS,
    ALL_ELEMENTS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def connectivity_suffix(cutoff, max_neighbors, fully_connected) -> str:
    if fully_connected:
        return "fc"
    s = f"cutoff{cutoff}"
    if max_neighbors is not None:
        s += f"_max{max_neighbors}"
    return s


def find_model_dirs(experiments_dir: str, models_filter=None):
    """Return list of (model_type, fold_dir) for the chosen fold of each model."""
    if not os.path.isdir(experiments_dir):
        raise FileNotFoundError(f"experiments dir not found: {experiments_dir}")
    out = []
    for name in sorted(os.listdir(experiments_dir)):
        if models_filter and name not in models_filter:
            continue
        mdir = os.path.join(experiments_dir, name)
        if not os.path.isdir(mdir):
            continue
        out.append((name, mdir))
    return out


def pick_fold_dir(model_dir: str, preferred_fold: int) -> Optional[str]:
    """Return path to fold_<preferred_fold> if it has config.json, else first usable fold."""
    candidates = [f"fold_{preferred_fold}"] + sorted(
        d for d in os.listdir(model_dir) if d.startswith("fold_")
    )
    seen = set()
    for d in candidates:
        if d in seen:
            continue
        seen.add(d)
        cfg = os.path.join(model_dir, d, "config.json")
        if os.path.isfile(cfg):
            return os.path.join(model_dir, d)
    return None


def build_batch(df, complexes_cache, atomic_data_dir, cutoff, max_neighbors,
                fully_connected, fold, batch_size, element_to_idx):
    """Build (or reuse) one padded batch matching this connectivity setting."""
    suffix = connectivity_suffix(cutoff, max_neighbors, fully_connected)
    if suffix in complexes_cache:
        complexes = complexes_cache[suffix]
    else:
        path = os.path.join(atomic_data_dir, f"precomputed_complexes_{suffix}.pkl")
        if os.path.exists(path):
            with open(path, 'rb') as f:
                complexes = pickle.load(f)
        else:
            complexes = precompute_complexes(
                df, element_to_idx=element_to_idx,
                cutoff=cutoff, max_neighbors=max_neighbors,
                fully_connected=fully_connected, max_dim=2,
                output_path=path, verbose=False,
            )
        complexes_cache[suffix] = complexes

    train_idx, _ = get_fold(df, fold)
    train_df = df.iloc[train_idx].copy()
    mol_stats = compute_molecular_stats(train_df)
    atomic_stats = compute_per_atom_stats(train_df)
    reg = apply_z_regularization(train_df, mol_stats, atomic_stats)
    batches = prepare_padded_batches(
        complexes, reg, TARGET_COLUMNS,
        batch_size=batch_size, verbose=False, as_numpy=True,
    )
    return batches[0]


def batch_size_info(batch):
    """Return (n_atoms, n_edges) for the first cochain dim of the batch."""
    cb0 = batch.cochain_batches[0]
    n_atoms = int(np.sum(np.asarray(cb0.num_cells)))
    n_edges = 0
    for cb in batch.cochain_batches:
        for name in ('up_senders', 'down_senders', 'boundary_senders', 'coboundary_senders'):
            arr = getattr(cb, name, None)
            if arr is not None:
                n_edges += int(arr.shape[0])
    return n_atoms, n_edges


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiments-dir", default=os.path.join(PROJECT_ROOT, "experiments", "atomic"))
    ap.add_argument("--data-pkl",
                    default=os.path.join(REPO_ROOT, "data_curation", "atomic", "train_and_val.pkl"))
    ap.add_argument("--atomic-data-dir",
                    default=os.path.join(REPO_ROOT, "data_curation", "atomic"))
    ap.add_argument("--fold", type=int, default=0,
                    help="Fold to use for the benchmark batch (default 0).")
    ap.add_argument("--models", nargs="*", default=None,
                    help="Restrict to these model_type names (default: all dirs).")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="Override the per-config batch_size (default: use config's).")
    ap.add_argument("--num-runs", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--out-csv", default=os.path.join(SCRIPT_DIR, "benchmark_results.csv"))
    args = ap.parse_args()

    print(f"JAX devices: {jax.devices()}")
    print(f"Experiments dir: {args.experiments_dir}")

    print("\nLoading dataset...")
    df = pd.read_pickle(args.data_pkl)
    element_to_idx = {e: i for i, e in enumerate(ALL_ELEMENTS)}

    model_dirs = find_model_dirs(args.experiments_dir, models_filter=args.models)
    print(f"Found {len(model_dirs)} model directories.")

    complexes_cache: dict = {}
    batch_cache: dict = {}  # keyed by (suffix, batch_size)
    rows = []

    for model_type, mdir in model_dirs:
        fold_dir = pick_fold_dir(mdir, args.fold)
        if fold_dir is None:
            print(f"[skip] {model_type}: no fold/config.json")
            continue
        with open(os.path.join(fold_dir, "config.json")) as f:
            cfg = json.load(f)

        model_class = cfg["model_class"]
        if model_class not in MODEL_CLASSES:
            print(f"[skip] {model_type}: unknown model_class {model_class!r}")
            continue
        model_kwargs = cfg["model_kwargs"]
        cutoff = cfg.get("cutoff", 5.25)
        max_neighbors = cfg.get("max_neighbors", 5)
        fully_connected = cfg.get("fully_connected", False)
        seed = cfg.get("seed", 0)
        bsz = args.batch_size or cfg.get("batch_size", 1024)

        print(f"\n=== {model_type} ({model_class}) — fold_dir={os.path.basename(fold_dir)} ===")
        print(f"  cutoff={cutoff}, max_neighbors={max_neighbors}, fc={fully_connected}, batch_size={bsz}")

        suffix = connectivity_suffix(cutoff, max_neighbors, fully_connected)
        cache_key = (suffix, bsz)
        if cache_key in batch_cache:
            batch = batch_cache[cache_key]
        else:
            batch = build_batch(df, complexes_cache, args.atomic_data_dir,
                                cutoff, max_neighbors, fully_connected,
                                args.fold, bsz, element_to_idx)
            batch_cache[cache_key] = batch

        n_atoms, n_edges = batch_size_info(batch)
        print(f"  batch: n_atoms={n_atoms}, n_edges={n_edges}")

        try:
            model = create_model(model_class, model_kwargs, seed=seed)
        except Exception as e:
            print(f"  [error] could not instantiate: {type(e).__name__}: {e}")
            continue

        n_params = count_parameters(model)

        try:
            flops = count_flops(model, batch)
        except Exception as e:
            print(f"  [warn] count_flops failed: {e}")
            flops = 0

        try:
            tinfo = benchmark_model(model, batch,
                                    num_warmup=args.warmup, num_runs=args.num_runs)
        except Exception as e:
            print(f"  [error] benchmark_model failed: {e}")
            continue

        flops_per_sec = (flops / (tinfo["mean_time_ms"] / 1000.0)) if flops and tinfo["mean_time_ms"] > 0 else 0.0
        print(f"  params={n_params:,}  flops={format_flops(flops)}  "
              f"time={format_time(tinfo['mean_time_ms'])} ± {tinfo['std_time_ms']:.2f} ms  "
              f"throughput={format_flops(flops_per_sec)}/s")

        rows.append({
            "model_type": model_type,
            "model_class": model_class,
            "fold_used": os.path.basename(fold_dir),
            "n_atoms_in_batch": n_atoms,
            "n_edges": n_edges,
            "flops": flops,
            "mean_time_ms": tinfo["mean_time_ms"],
            "std_time_ms": tinfo["std_time_ms"],
            "flops_per_sec": flops_per_sec,
            "n_params": n_params,
            "cutoff": cutoff,
            "max_neighbors": max_neighbors,
            "fully_connected": fully_connected,
            "batch_size": bsz,
        })

    if not rows:
        print("\nNo models benchmarked.")
        return

    out = pd.DataFrame(rows).sort_values("mean_time_ms").reset_index(drop=True)
    out.to_csv(args.out_csv, index=False)
    print(f"\nWrote {args.out_csv}")

    print("\n| model | class | params | flops | time (ms) | FLOP/s |")
    print("|---|---|---:|---:|---:|---:|")
    for r in out.to_dict("records"):
        print(f"| {r['model_type']} | {r['model_class']} | {r['n_params']:,} | "
              f"{format_flops(r['flops'])} | {r['mean_time_ms']:.2f} ± {r['std_time_ms']:.2f} | "
              f"{format_flops(r['flops_per_sec'])}/s |")


if __name__ == "__main__":
    main()
