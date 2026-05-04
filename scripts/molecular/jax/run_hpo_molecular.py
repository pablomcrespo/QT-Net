#!/usr/bin/env python3
"""
Optuna HPO for molecular property prediction.

Supports ScalarGNNMolecular (topology only, no rings) and
ScalarTPaiNNMolecular (topology + ring cells).  Run with
--use-atom-features to enable the informed variant that injects
per-atom N, LI, Mu, Q descriptors.

Only fold 0 (first fold of first repeat) is used, consistent with
using fold 1 / repeat 1 in 1-indexed notation.

Val loss is reported to Optuna at every epoch so the MedianPruner can
cut unpromising trials early.

Usage examples:
    python run_hpo_molecular.py --model-name ScalarGNNMolecular
    python run_hpo_molecular.py --model-name ScalarTPaiNNMolecular --use-atom-features
    python run_hpo_molecular.py --model-name ScalarGNNMolecular --n-trials 60 --epochs-per-trial 200
"""

import os
import sys
import re
import time
import json
import argparse
import pickle
import warnings
import functools

import numpy as np
import pandas as pd
import optax
import optuna
from optuna.samplers import TPESampler
import jax
import jax.numpy as jnp
from flax import nnx

# ---------------------------------------------------------------------------
# Locate repository root so that qtnet is importable regardless of cwd
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _find_repo_root(start_dir: str) -> str:
    cur = os.path.abspath(start_dir)
    while True:
        if os.path.isdir(os.path.join(cur, 'data_curation')):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return os.path.abspath(os.path.join(start_dir, '..', '..', '..'))
        cur = parent


REPO_ROOT = _find_repo_root(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
for _root, _dirs, _ in os.walk(REPO_ROOT):
    if 'qtnet' in _dirs:
        if _root not in sys.path:
            sys.path.insert(0, _root)
        break

from qtnet.data_utils import create_cv_splits, MOLECULAR_PROPERTIES_PRED
from qtnet.jax_models.representations import row_to_molecular_complex, prepare_padded_batches
from qtnet.jax_models.models_molecular import ScalarGNNMolecular, ScalarTPaiNNMolecular
from qtnet.jax_models.train_utils import (
    compute_molecular_multitask_loss,
    count_parameters,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALL_ELEMENTS   = ['H', 'C', 'N', 'O']
TARGET_COLUMNS = MOLECULAR_PROPERTIES_PRED   # ['alpha', 'gap', 'U0', 'Cv']
DEFAULT_PKL    = os.path.join(
    REPO_ROOT, 'data_curation', 'molecular', 'aimel_clustered_molecular.pkl'
)

MODEL_CLASSES = {
    'ScalarGNNMolecular':    ScalarGNNMolecular,
    'ScalarTPaiNNMolecular': ScalarTPaiNNMolecular,
}

# ---------------------------------------------------------------------------
# Hyperparameter search spaces
# ---------------------------------------------------------------------------
HP_SPACES = {
    'ScalarGNNMolecular': {
        'num_node_scalars':    ('categorical', {'choices': [8,16]}),
        'num_edge_scalars':    ('categorical', {'choices': [8,16]}),
        'num_complex_scalars': ('categorical', {'choices': [16,32]}),
        'embedding_dim':       ('categorical', {'choices': [32]}),
        'hidden_dim':          ('categorical', {'choices': [32]}),
        'num_layers':          ('int',         {'low': 2, 'high': 6}),
        'lr':                  ('float',       {'low': 1e-5, 'high': 1e-2, 'log': True}),
        'weight_decay':        ('float',       {'low': 1e-6, 'high': 1e-3, 'log': True}),
    },
    'ScalarTPaiNNMolecular': {
        'num_node_scalars':    ('categorical', {'choices': [8,16]}),
        'num_edge_scalars':    ('categorical', {'choices': [8,16]}),
        'num_ring_scalars':    ('categorical', {'choices': [8,16]}),
        'num_complex_scalars': ('categorical', {'choices': [16,32,48]}),
        'embedding_dim':       ('categorical', {'choices': [32]}),
        'hidden_dim':          ('categorical', {'choices': [48, 64, 96]}),
        'num_layers':          ('int',         {'low': 2, 'high': 6}),
        'lr':                  ('float',       {'low': 5e-5, 'high': 1e-3, 'log': True}),
        'weight_decay':        ('float',       {'low': 5e-6, 'high': 1e-4, 'log': True}),
    },
}

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def norm_stats_from_train(df_train: pd.DataFrame) -> dict:
    """Compute z-score stats for each target property from training rows."""
    stats = {}
    for prop in TARGET_COLUMNS:
        vals = df_train[prop].values.astype(np.float64)
        std  = float(vals.std())
        if std < 1e-8:
            std = 1.0
        stats[prop] = {'mean': float(vals.mean()), 'std': std}
    return stats


def normalize_targets(df: pd.DataFrame, stats: dict) -> np.ndarray:
    """Return (n_mols, n_targets) float32 array z-normalised by stats."""
    out = np.zeros((len(df), len(TARGET_COLUMNS)), dtype=np.float32)
    for i, prop in enumerate(TARGET_COLUMNS):
        vals = df[prop].values.astype(np.float32)
        out[:, i] = (vals - stats[prop]['mean']) / stats[prop]['std']
    return out


# ---------------------------------------------------------------------------
# FLOPs counting (molecular-aware)
# ---------------------------------------------------------------------------

def _format_flops(flops: int) -> str:
    if flops == 0:
        return "N/A"
    for unit, threshold in [('TFLOPs', 1e12), ('GFLOPs', 1e9),
                            ('MFLOPs', 1e6),  ('KFLOPs', 1e3)]:
        if flops >= threshold:
            return f"{flops / threshold:.2f} {unit}"
    return f"{flops:.0f} FLOPs"


def count_molecular_flops(model, batch) -> int:
    """Count FLOPs for one molecular forward pass via JAX HLO cost analysis."""
    graphdef, state = nnx.split(model)
    graph_idx  = batch.cochain_batches[0].owner_cochains
    num_graphs = int(batch.num_complexes[0])

    def _forward(state):
        mdl = nnx.merge(graphdef, state)
        return mdl(batch, graph_idx=graph_idx, num_graphs=num_graphs)

    try:
        lowered  = jax.jit(_forward).lower(state)
        compiled = lowered.compile()
        cost     = compiled.cost_analysis()
        if isinstance(cost, dict):
            flops = cost.get('flops', 0)
        elif isinstance(cost, (list, tuple)) and len(cost) > 0:
            flops = cost[0].get('flops', 0)
        else:
            flops = 0
        return int(flops)
    except Exception as exc:
        print(f"  cost_analysis failed ({type(exc).__name__}: {exc})")
        return 0


# ---------------------------------------------------------------------------
# Per-trial forward-pass timing (molecular-aware)
# ---------------------------------------------------------------------------

def _benchmark_molecular(model, batch, num_warmup: int = 1, num_runs: int = 3) -> dict:
    """Time a molecular model forward pass; returns ms mean/std and throughput."""
    graph_idx  = batch.cochain_batches[0].owner_cochains
    num_graphs = int(batch.num_complexes[0])

    @nnx.jit
    def _forward(model, batch):
        return model(batch, graph_idx=graph_idx, num_graphs=num_graphs)

    for _ in range(num_warmup):
        jax.block_until_ready(_forward(model, batch))

    times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        jax.block_until_ready(_forward(model, batch))
        times.append(time.perf_counter() - t0)

    times = np.array(times)
    return {
        'mean_time_ms': float(np.mean(times) * 1000),
        'std_time_ms':  float(np.std(times)  * 1000),
        'throughput':   float(1.0 / np.mean(times)),
    }


# ---------------------------------------------------------------------------
# Optuna sampling helper
# ---------------------------------------------------------------------------

def _sample_hp(trial: optuna.Trial, hp_space: dict) -> dict:
    sampled = {}
    for name, (suggest_type, kwargs) in hp_space.items():
        if suggest_type == 'int':
            sampled[name] = trial.suggest_int(name, **kwargs)
        elif suggest_type == 'float':
            sampled[name] = trial.suggest_float(name, **kwargs)
        elif suggest_type == 'categorical':
            sampled[name] = trial.suggest_categorical(name, **kwargs)
        else:
            raise ValueError(f"Unknown suggest_type '{suggest_type}' for '{name}'")
    return sampled


# ---------------------------------------------------------------------------
# MolecularHPO
# ---------------------------------------------------------------------------

class MolecularHPO:
    """
    Optuna HPO for ScalarGNNMolecular / ScalarTPaiNNMolecular.

    Val loss is reported to Optuna at every epoch so the MedianPruner can
    cut bad trials after the warmup window.  Construct and call .run().

    Args:
        model_name         : 'ScalarGNNMolecular' or 'ScalarTPaiNNMolecular'.
        use_atom_features  : Whether to pass atom-level descriptors.
        num_species        : Vocabulary size (len(element_to_idx)).
        train_batches      : Padded ComplexBatch list (targets separate).
        train_targets      : (N_train, 4) z-normalised numpy array.
        val_batches        : Padded ComplexBatch list for validation.
        val_targets        : (N_val, 4) z-normalised numpy array.
        hp_space           : Search-space dict (default: HP_SPACES[model_name]).
        n_trials           : Number of Optuna trials.
        epochs_per_trial   : Training epochs per trial.
        seed               : Base random seed.
        verbose            : Print per-trial summaries.
        benchmark_per_trial: Time and count FLOPs before training each trial.
    """

    def __init__(
        self,
        model_name: str,
        use_atom_features: bool,
        num_species: int,
        train_batches: list,
        train_targets: np.ndarray,
        val_batches: list,
        val_targets: np.ndarray,
        hp_space: dict = None,
        n_trials: int = 60,
        epochs_per_trial: int = 200,
        seed: int = 0,
        verbose: bool = True,
        benchmark_per_trial: bool = True,
    ):
        self.model_name          = model_name
        self.use_atom_features   = use_atom_features
        self.num_species         = num_species
        self.train_batches       = train_batches
        self.train_targets       = train_targets
        self.val_batches         = val_batches
        self.val_targets         = val_targets
        self.hp_space            = hp_space or HP_SPACES[model_name]
        self.n_trials            = n_trials
        self.epochs_per_trial    = epochs_per_trial
        self.seed                = seed
        self.verbose             = verbose
        self.benchmark_per_trial = benchmark_per_trial
        self.model_class         = MODEL_CLASSES[model_name]

    @property
    def study_name(self) -> str:
        variant = 'informed' if self.use_atom_features else 'blind'
        return f"{self.model_name}_{variant}"

    def _build_model(self, hp: dict, seed: int):
        return self.model_class(
            num_species=self.num_species,
            num_outputs=len(TARGET_COLUMNS),
            use_atom_features=self.use_atom_features,
            rngs=nnx.Rngs(seed),
            **hp,
        )

    def _objective(self, trial: optuna.Trial) -> float:
        hp           = _sample_hp(trial, self.hp_space)
        lr           = hp.pop('lr')
        weight_decay = hp.pop('weight_decay', 1e-4)

        model     = self._build_model(hp, seed=self.seed + trial.number)
        tx        = optax.adamw(learning_rate=lr, weight_decay=weight_decay)
        optimizer = nnx.Optimizer(model, tx)

        # --- optional per-trial benchmark -----------------------------------
        bench = {}
        flops = 0
        if self.benchmark_per_trial and self.train_batches:
            try:
                perf_batch = self.train_batches[0]
                bench  = _benchmark_molecular(model, perf_batch, num_warmup=1, num_runs=3)
                flops  = count_molecular_flops(model, perf_batch)
                n_params = count_parameters(model)
                trial.set_user_attr('bench',    bench)
                trial.set_user_attr('flops',    flops)
                trial.set_user_attr('n_params', n_params)
                if self.verbose:
                    print(
                        f"[{self.study_name}] trial {trial.number:>3d} pre-train: "
                        f"time={bench['mean_time_ms']:.1f}ms "
                        f"std={bench['std_time_ms']:.1f}ms "
                        f"thr={bench['throughput']:.1f}it/s "
                        f"flops={_format_flops(flops)} "
                        f"params={n_params:,}"
                    )
            except Exception as exc:
                warnings.warn(f"Benchmark failed (trial {trial.number}): {exc}")

        # --- jit-compiled step functions ------------------------------------
        # num_graphs must be a compile-time constant for jax.ops.segment_sum.
        # We pass it as a static keyword argument and extract it from the batch
        # outside the jit call (where it is a concrete Python int).

        @functools.partial(nnx.jit, static_argnames=('num_graphs',))
        def _train_step(model, optimizer, batch, batch_targets, num_graphs):
            graph_idx = batch.cochain_batches[0].owner_cochains

            def loss_fn(model):
                out   = model(batch, graph_idx=graph_idx, num_graphs=num_graphs)
                total, losses = compute_molecular_multitask_loss(
                    out['predictions'], batch_targets, TARGET_COLUMNS
                )
                return total, losses

            grads, losses = nnx.grad(loss_fn, has_aux=True)(model)
            optimizer.update(grads)
            return losses

        @functools.partial(nnx.jit, static_argnames=('num_graphs',))
        def _val_step(model, batch, batch_targets, num_graphs):
            graph_idx = batch.cochain_batches[0].owner_cochains
            out = model(batch, graph_idx=graph_idx, num_graphs=num_graphs)
            _, losses = compute_molecular_multitask_loss(
                out['predictions'], batch_targets, TARGET_COLUMNS
            )
            return losses

        # --- epoch loop with per-epoch Optuna reporting ---------------------
        loss_keys = ['total'] + list(TARGET_COLUMNS)
        best_val  = float('inf')

        try:
            for epoch in range(self.epochs_per_trial):
                # train
                mol_idx = 0
                for batch in self.train_batches:
                    n_mols        = int(batch.num_complexes[0])
                    batch_targets = jnp.asarray(
                        self.train_targets[mol_idx:mol_idx + n_mols]
                    )
                    mol_idx += n_mols
                    _train_step(model, optimizer, batch, batch_targets,
                                num_graphs=n_mols)

                # validate
                vl = {k: [] for k in loss_keys}
                mol_idx = 0
                for batch in self.val_batches:
                    n_mols        = int(batch.num_complexes[0])
                    batch_targets = jnp.asarray(
                        self.val_targets[mol_idx:mol_idx + n_mols]
                    )
                    mol_idx += n_mols
                    losses = _val_step(model, batch, batch_targets,
                                       num_graphs=n_mols)
                    for k in loss_keys:
                        if k in losses:
                            vl[k].append(float(losses[k]))

                avg_val = float(np.mean(vl['total'])) if vl['total'] else float('inf')
                best_val = min(best_val, avg_val)

                # report to Optuna every epoch so the pruner has full visibility
                trial.report(avg_val, step=epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

        except optuna.exceptions.TrialPruned:
            raise
        except Exception as exc:
            print(f"[{self.study_name}] trial {trial.number:>3d} "
                  f"training failed: {exc}")
            raise optuna.exceptions.TrialPruned()

        # --- post-trial logging ---------------------------------------------
        if self.verbose:
            n_params = count_parameters(model)
            hp_str   = " | ".join(f"{k}={v}" for k, v in hp.items())
            print(
                f"[{self.study_name}] trial {trial.number:>3d} | "
                f"val={best_val:.5f} | params={n_params:,} | lr={lr:.2e} | "
                + hp_str
            )

        return best_val

    def run(self, save_top_n: int = 5) -> optuna.Study:
        """Run all trials and return the finished study.

        Parameters
        ----------
        save_top_n : int
            Write the top-N trial configs + FLOPs + timing to
            ``{study_name}_optuna.json`` when > 0.
        """
        study = optuna.create_study(
            direction='minimize',
            study_name=self.study_name,
            sampler=TPESampler(seed=self.seed),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5, n_warmup_steps=30, interval_steps=2
            ),
        )
        study.optimize(self._objective, n_trials=self.n_trials)

        print(f"\n{'='*60}")
        print(f"[{self.study_name}] HPO complete — best val: {study.best_value:.5f}")
        print(f"  Best params: {study.best_params}")
        print(f"{'='*60}\n")

        if save_top_n > 0:
            trials = sorted(
                [t for t in study.trials if t.value is not None],
                key=lambda t: t.value,
            )[:save_top_n]
            records = []
            for t in trials:
                flops = t.user_attrs.get('flops', 0)
                rec = {
                    'trial':         t.number,
                    'value':         t.value,
                    'params':        t.params,
                    'n_params':      t.user_attrs.get('n_params'),
                    'flops':         _format_flops(flops),
                    'flops_raw':     flops,
                    **t.user_attrs.get('bench', {}),
                }
                records.append(rec)
            fname = f"{self.study_name}_optuna.json"
            with open(fname, 'w') as fp:
                json.dump(records, fp, indent=2)
            print(f"Saved top-{save_top_n} trial info to {fname}")

        return study


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Optuna HPO for molecular property prediction models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Models
------
  ScalarGNNMolecular    — BCP bond topology only (no rings)
  ScalarTPaiNNMolecular — BCP bond topology + RCP ring topology

Target properties: alpha, gap, U0, Cv

The search uses fold 0 (= fold 1, repeat 1 in 1-indexed notation).
Val loss is reported to Optuna every epoch; the MedianPruner cuts bad trials.
        """,
    )
    p.add_argument(
        '--model-name', required=True,
        choices=list(MODEL_CLASSES.keys()),
        help='Model class to search',
    )
    p.add_argument(
        '--use-atom-features', action='store_true',
        help='Inject per-atom N, LI, Mu, Q descriptors (informed variant)',
    )
    p.add_argument(
        '--pkl-file', default=DEFAULT_PKL,
        help='Path to the molecular DataFrame pickle',
    )
    p.add_argument(
        '--complexes-pkl', default=os.path.join(
    REPO_ROOT, 'data_curation', 'molecular', 'precomputed_blind.pkl'
),
        help='Pre-built complexes cache; saves/loads to skip complex construction',
    )
    p.add_argument(
        '--batch-size', type=int, default=32,
        help='Padded batch size (default: 32)',
    )
    p.add_argument(
        '--n-trials', type=int, default=None,
        help='Number of Optuna trials (model-specific default if omitted)',
    )
    p.add_argument(
        '--epochs-per-trial', type=int, default=None,
        help='Training epochs per trial (model-specific default if omitted)',
    )
    p.add_argument(
        '--save-top-n', type=int, default=5,
        help='Number of best-trial configs to dump to JSON',
    )
    p.add_argument(
        '--seed', type=int, default=0,
        help='Base random seed for reproducibility',
    )
    p.add_argument(
        '--n-splits', type=int, default=5,
        help='Number of CV folds (default: 5)',
    )
    p.add_argument(
        '--group-col', type=str, default='Murcko_Scaffold',
        help="DataFrame column for grouped k-fold; 'none' for plain KFold",
    )
    p.add_argument(
        '--no-benchmark', action='store_true',
        help='Skip per-trial forward-pass timing and FLOPs counting',
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Default HPO budgets (n_trials × epochs ≈ 12 h on one GPU for these models)
# ---------------------------------------------------------------------------
MODEL_DEFAULTS = {
    'ScalarGNNMolecular':    {'n_trials': 60, 'epochs': 200},
    'ScalarTPaiNNMolecular': {'n_trials': 60, 'epochs': 200},
}


def main():
    args = parse_args()

    # ---- load DataFrame ----------------------------------------------------
    print(f"Loading {args.pkl_file} ...")
    t0 = time.perf_counter()
    df = pd.read_pickle(args.pkl_file)
    print(f"  {len(df)} molecules ({time.perf_counter() - t0:.1f}s)")

    #missing = [c for c in ['a_name', 'BCP_connectivity', 'RCP_connectivity'] + TARGET_COLUMNS
    #           if c not in df.columns]
    #if missing:
    #    raise ValueError(f"DataFrame missing required columns: {missing}")

    element_to_idx = {e: i for i, e in enumerate(ALL_ELEMENTS)}
    num_species    = len(element_to_idx)

    # ----  load complexes -------------------------------------------
    if args.complexes_pkl and os.path.exists(args.complexes_pkl):
        print(f"Loading pre-built complexes from {args.complexes_pkl} ...")
        with open(args.complexes_pkl, 'rb') as f:
            cpx_data = pickle.load(f)
        complexes      = cpx_data['complexes']
        element_to_idx = cpx_data['element_to_idx']
        num_species    = len(element_to_idx)
        cached_af      = cpx_data.get('use_atom_features', args.use_atom_features)
        if cached_af != args.use_atom_features:
            warnings.warn(
                f"Cached complexes have use_atom_features={cached_af} but "
                f"--use-atom-features={args.use_atom_features}. "
                f"Using cached complexes as-is (atom features are embedded in the Complex)."
            )
        print(f"  Loaded {len(complexes)} complexes")
    
    # ---- fold 0: first fold of first repeat --------------------------------
    group_col = None if args.group_col.lower() == 'none' else args.group_col

    train_idx = val_idx = None
    for split_info in create_cv_splits(
        df,
        n_splits=args.n_splits,
        n_repeats=1,
        group_col=group_col,
        base_seed=args.seed,
        training_fractions=[1.0],
        val_fraction = 0.1
    ):
        if split_info['fold'] == 0:
            train_idx = split_info['train_idx']
            val_idx   = split_info['val_idx']
            break

    if train_idx is None:
        raise RuntimeError("No fold 0 found — check n_splits >= 1")

    df_train = df.iloc[train_idx].copy()
    df_val   = df.iloc[val_idx].copy()
    print(f"\nFold 0: {len(df_train)} train / {len(df_val)} val molecules")

    # ---- z-score normalisation (fit on training set only) ------------------
    norm_stats    = norm_stats_from_train(df_train)
    train_targets = normalize_targets(df_train, norm_stats)
    val_targets   = normalize_targets(df_val,   norm_stats)

    # ---- padded ComplexBatch lists -----------------------------------------
    print("Building padded train batches ...", end=" ", flush=True)
    t0 = time.perf_counter()
    train_batches = prepare_padded_batches(
        complexes, df_train, target_columns=[],
        batch_size=args.batch_size, verbose=False, as_numpy=True,
    )
    print(f"{len(train_batches)} batches ({time.perf_counter() - t0:.1f}s)")

    print("Building padded val batches   ...", end=" ", flush=True)
    t0 = time.perf_counter()
    val_batches = prepare_padded_batches(
        complexes, df_val, target_columns=[],
        batch_size=args.batch_size, verbose=False, as_numpy=True,
    )
    print(f"{len(val_batches)} batches ({time.perf_counter() - t0:.1f}s)")

    # ---- resolve trial / epoch budget --------------------------------------
    defaults = MODEL_DEFAULTS.get(args.model_name, {'n_trials': 60, 'epochs': 200})
    n_trials = args.n_trials         if args.n_trials         is not None else defaults['n_trials']
    epochs   = args.epochs_per_trial if args.epochs_per_trial is not None else defaults['epochs']

    # ---- run HPO -----------------------------------------------------------
    variant = 'informed' if args.use_atom_features else 'blind'
    print(f"\n{'='*60}")
    print(f"HPO: {args.model_name} ({variant})")
    print(f"  n_trials={n_trials}  epochs_per_trial={epochs}  seed={args.seed}")
    print(f"  batch_size={args.batch_size}  num_species={num_species}")
    print(f"{'='*60}\n")

    hpo = MolecularHPO(
        model_name=args.model_name,
        use_atom_features=args.use_atom_features,
        num_species=num_species,
        train_batches=train_batches,
        train_targets=train_targets,
        val_batches=val_batches,
        val_targets=val_targets,
        n_trials=n_trials,
        epochs_per_trial=epochs,
        seed=args.seed,
        benchmark_per_trial=not args.no_benchmark,
    )

    study = hpo.run(save_top_n=args.save_top_n)
    print(f"HPO complete — study name: '{study.study_name}'")


if __name__ == '__main__':
    main()
