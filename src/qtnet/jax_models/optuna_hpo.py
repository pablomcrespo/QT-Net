"""
Optuna hyperparameter optimisation utilities for atomic multitask models.

Public API
----------
DEFAULT_HP_SPACES : dict
    Pre-built search-space specs for TPaiNN, EquivariantGNN, ScalarGNN,
    ScalarTPaiNN.

make_factories(num_species) -> dict
    Returns a FACTORIES dict whose values are (hp, seed) -> model callables.

OptunaHPO
    Fluent-builder class.  Construct with the four required arguments, then
    chain optional configurators before calling .run():

        OptunaHPO(model_name, factory, train_batches, val_batches)
            .with_training(n_trials=30, epochs_per_trial=20, seed=0)
            .with_search_space({...})           # optional override
            .with_optuna(pruner=..., sampler=...) # optional override
            .run()  -> optuna.Study
"""

import os
import sys
import optuna
from optuna.samplers import TPESampler
import jax
import time
import re
from flax import nnx
import optax
import numpy as np


from qtnet.jax_models.models_equivariant import TPaiNN, EquivariantGNN
from qtnet.jax_models.models_scalar import ScalarGNN, ScalarTPaiNN, ScalarBaseline, ScalarBaselineEdges
import qtnet.jax_models.train_utils
from qtnet.jax_models.train_utils import make_optimizer


# ---------------------------------------------------------------------------
# Hyperparameter search-space specifications
#
# Each entry maps a parameter name to (suggest_type, kwargs).
#   suggest_type : 'int' | 'float' | 'categorical'
#   kwargs       : forwarded to the corresponding trial.suggest_* call.
#
# Must always include an 'lr' key for the learning rate.
# ---------------------------------------------------------------------------

DEFAULT_HP_SPACES = {
    'TPaiNN': {
        'num_node_scalars':     ('categorical', {'choices': [16,24]}),
        'num_node_vectors':     ('categorical', {'choices': [8]}),
        'num_node_tensors':     ('categorical', {'choices': [8]}),
        'num_edge_scalars':     ('categorical', {'choices': [16,24]}),
        'num_edge_vectors':     ('categorical', {'choices': [8]}),
        'num_edge_tensors':     ('categorical', {'choices': [8]}),
        'num_bag_scalars':      ('categorical', {'choices': [16]}),
        'num_bag_vectors':      ('categorical', {'choices': [8]}),
        'num_bag_tensors':      ('categorical', {'choices': [8]}),
        'embedding_dim':        ('categorical', {'choices': [24,32]}),
        'hidden_dim':           ('categorical', {'choices': [32,48]}),
        'hidden_l1_channels':   ('categorical', {'choices': [12]}),
        'hidden_l2_channels':   ('categorical', {'choices': [12]}),
        'geometric_filter_dim': ('categorical', {'choices': [24,32]}),
        'geo_basis_dim':        ('categorical', {'choices': [8,16]}),
        'num_layers':           ('int',         {'low': 1, 'high': 4}),
        'lr':                   ('float',       {'low': 1e-4, 'high': 1e-2, 'log': True}),
        'weight_decay':         ('float',       {'low': 1e-5, 'high': 1e-3, 'log': True}),
    },
    'EquivariantGNN': {
        'num_node_scalars':     ('categorical', {'choices': [24]}),
        'num_node_vectors':     ('categorical', {'choices': [8, 12]}),
        'num_node_tensors':     ('categorical', {'choices': [8, 12]}),
        'num_edge_scalars':     ('categorical', {'choices': [24, 32]}),
        'num_edge_vectors':     ('categorical', {'choices': [8, 12,16]}),
        'num_edge_tensors':     ('categorical', {'choices': [8, 12]}),
        'embedding_dim':        ('categorical', {'choices': [24,32]}),
        'hidden_dim':           ('categorical', {'choices': [32,48]}),
        'hidden_l1_channels':   ('categorical', {'choices': [12]}),
        'hidden_l2_channels':   ('categorical', {'choices': [12]}),
        'geometric_filter_dim': ('categorical', {'choices': [24,32]}),
        'geo_basis_dim':        ('categorical', {'choices': [8,16]}),
        'num_layers':           ('int',         {'low': 1, 'high': 4}),
        'lr':                   ('float',       {'low': 1e-4, 'high': 1e-2, 'log': True}),
        'weight_decay':         ('float',       {'low': 1e-5, 'high': 1e-3, 'log': True}),
    },
    'ScalarGNN': {
        'num_node_scalars':     ('categorical', {'choices': [24, 32]}),
        'num_edge_scalars':     ('categorical', {'choices': [24, 32]}),
        'embedding_dim':        ('categorical', {'choices': [ 24,32]}),
        'hidden_dim':           ('categorical', {'choices': [ 48,64]}),
        'num_layers':           ('int',         {'low': 2, 'high': 6}),
        'lr':                   ('float',       {'low': 1e-4, 'high': 1e-2, 'log': True}),
        'weight_decay':         ('float',       {'low': 1e-5, 'high': 1e-3, 'log': True}),
    },
    'ScalarTPaiNN': {
        'num_node_scalars':     ('categorical', {'choices': [16, 24]}),
        'num_edge_scalars':     ('categorical', {'choices': [16, 24]}),
        'num_bag_scalars':     ('categorical',  {'choices': [16, 24]}),
        'embedding_dim':        ('categorical', {'choices': [24,32]}),
        'hidden_dim':           ('categorical', {'choices': [48,64]}),
        'num_layers':           ('int',         {'low': 2, 'high': 6}),
        'lr':                   ('float',       {'low': 1e-4, 'high': 1e-2, 'log': True}),
        'weight_decay':         ('float',       {'low': 1e-5, 'high': 1e-3, 'log': True}),
    },
    'ScalarBaseline': {
        'num_node_scalars':     ('categorical', {'choices': [64,96]}),
        'embedding_dim':        ('categorical', {'choices': [64,96]}),
        'hidden_dim':           ('categorical', {'choices': [64, 96, 128]}),
        'num_layers':           ('int',         {'low': 2, 'high': 10}),
        'lr':                   ('float',       {'low': 1e-4, 'high': 1e-2, 'log': True}),
        'weight_decay':         ('float',       {'low': 1e-5, 'high': 1e-3, 'log': True}),
    },
    'ScalarBaselineEdges': {
        'num_node_scalars':     ('categorical', {'choices': [48, 64]}),
        'num_edge_scalars':     ('categorical', {'choices': [48, 64]}),
        'embedding_dim':        ('categorical', {'choices': [64, 96]}),
        'hidden_dim':           ('categorical', {'choices': [64, 96, 128]}),
        'num_layers':           ('int',         {'low': 2, 'high': 10}),
        'lr':                   ('float',       {'low': 1e-4, 'high': 1e-2, 'log': True}),
        'weight_decay':         ('float',       {'low': 1e-5, 'high': 1e-3, 'log': True}),
    },
    # Fully-connected variants (graph connectivity is all-pairs; cutoff only
    # controls the RBF basis inside the model for EquivariantGNN_FC)
    'EquivariantGNN_FC': {
        'num_node_scalars':     ('categorical', {'choices': [16]}),
        'num_node_vectors':     ('categorical', {'choices': [8]}),
        'num_node_tensors':     ('categorical', {'choices': [8]}),
        'num_edge_scalars':     ('categorical', {'choices': [16]}),
        'num_edge_vectors':     ('categorical', {'choices': [8]}),
        'num_edge_tensors':     ('categorical', {'choices': [8]}),
        'embedding_dim':        ('categorical', {'choices': [24]}),
        'hidden_dim':           ('categorical', {'choices': [32,48]}),
        'hidden_l1_channels':   ('categorical', {'choices': [12]}),
        'hidden_l2_channels':   ('categorical', {'choices': [12]}),
        'geometric_filter_dim': ('categorical', {'choices': [16]}),
        'geo_basis_dim':        ('categorical', {'choices': [8]}),
        'num_layers':           ('int',         {'low': 1, 'high': 4}),
        'cutoff':               ('float',       {'low': 4.0, 'high': 9.0}),
        'lr':                   ('float',       {'low': 1e-4, 'high': 1e-2, 'log': True}),
        'weight_decay':         ('float',       {'low': 1e-5, 'high': 1e-3, 'log': True}),
    },
    'ScalarGNN_FC': {
        'num_node_scalars':     ('categorical', {'choices': [16,24]}),
        'num_edge_scalars':     ('categorical', {'choices': [16,24]}),
        'embedding_dim':        ('categorical', {'choices': [24, 32]}),
        'hidden_dim':           ('categorical', {'choices': [32,48,64]}),
        'num_layers':           ('int',         {'low': 2, 'high': 5}),
        'lr':                   ('float',       {'low': 1e-4, 'high': 1e-2, 'log': True}),
        'weight_decay':         ('float',       {'low': 1e-5, 'high': 1e-3, 'log': True}),
    },
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sample_hp(trial: optuna.Trial, hp_space: dict) -> dict:
    """Sample one configuration from a space spec using an Optuna trial."""
    sampled = {}
    for name, (suggest_type, kwargs) in hp_space.items():
        if suggest_type == 'int':
            sampled[name] = trial.suggest_int(name, **kwargs)
        elif suggest_type == 'float':
            sampled[name] = trial.suggest_float(name, **kwargs)
        elif suggest_type == 'categorical':
            sampled[name] = trial.suggest_categorical(name, **kwargs)
        else:
            raise ValueError(f"Unknown suggest type '{suggest_type}' for param '{name}'")
    return sampled


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def make_factories(num_species: int) -> dict:
    """
    Return a dict of model factory callables keyed by model name.

    Each factory has the signature ``(hp: dict, seed: int) -> nnx.Module``,
    where ``hp`` contains all architecture hyperparameters *except* ``'lr'``
    (which is consumed by OptunaHPO before calling the factory).

    Parameters
    ----------
    num_species : int
        Number of chemical species / atom types (vocabulary size).

    Returns
    -------
    dict[str, callable]
    """

    def _tpainn(hp: dict, seed: int = 0):
        # ``use_compression`` is part of the hyperparameter space so it
        # should come from ``hp`` rather than be hard‑coded here.  dropping
        # the explicit keyword avoids the "multiple values" TypeError from
        # Optuna.
        return TPaiNN(
            num_species=num_species,
            rngs=nnx.Rngs(seed),
            **hp,
        )

    def _egnn(hp: dict, seed: int = 0):
        # see comment in _tpainn; compression setting should originate from
        # the sampled hp dictionary.
        return EquivariantGNN(
            num_species=num_species,
            rngs=nnx.Rngs(seed),
            **hp,
        )

    def _sgnn(hp: dict, seed: int = 0):
        return ScalarGNN(num_species=num_species, rngs=nnx.Rngs(seed), **hp)

    def _stpainn(hp: dict, seed: int = 0):
        return ScalarTPaiNN(num_species=num_species, rngs=nnx.Rngs(seed), **hp)


    def _scalar_baseline(hp: dict, seed: int = 0):
        return ScalarBaseline(num_species=num_species, rngs=nnx.Rngs(seed), **hp)

    def _scalar_baseline_edges(hp: dict, seed: int = 0):
        return ScalarBaselineEdges(num_species=num_species, rngs=nnx.Rngs(seed), **hp)

    # Fully-connected variants reuse the same model classes; 'cutoff' (when
    # present) is forwarded as an RBF-basis cutoff, not a graph cutoff.
    def _egnn_fc(hp: dict, seed: int = 0):
        return EquivariantGNN(num_species=num_species, rngs=nnx.Rngs(seed), **hp)

    def _sgnn_fc(hp: dict, seed: int = 0):
        return ScalarGNN(num_species=num_species, rngs=nnx.Rngs(seed), **hp)

    return {
        'TPaiNN':            _tpainn,
        'EquivariantGNN':    _egnn,
        'ScalarGNN':         _sgnn,
        'ScalarTPaiNN':      _stpainn,
        'ScalarBaseline':    _scalar_baseline,
        'ScalarBaselineEdges': _scalar_baseline_edges,
        'EquivariantGNN_FC': _egnn_fc,
        'ScalarGNN_FC':      _sgnn_fc,
    }


def benchmark_model(model, batch, num_warmup=1, num_runs=5):
    """
    Benchmark a model's forward pass throughput.

    This function uses a small number of warmup/real runs by default to make
    it lightweight during HPO summary estimation.

    Returns:
        dict with 'mean_time_ms', 'std_time_ms', 'throughput' (runs/s)
    """
    @nnx.jit
    def forward(model):
        return model(batch)

    # Warmup (compile + fill caches)
    for _ in range(num_warmup):
        out = forward(model)
        jax.block_until_ready(out)

    # Timed runs
    times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        out = forward(model)
        jax.block_until_ready(out)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    
    times = np.array(times)
    return {
        'mean_time_ms': float(np.mean(times) * 1000),
        'std_time_ms': float(np.std(times) * 1000),
        'throughput': float(1.0 / np.mean(times)),
    }


def count_flops(model, batch):
    """
    Count FLOPs for a single forward pass using JAX's HLO cost analysis.
    
    Uses nnx.split/merge to make the model compatible with jax.jit tracing.
    """
    graphdef, state = nnx.split(model)
    
    def forward_fn(state):
        mdl = nnx.merge(graphdef, state)
        return mdl(batch)
    
    try:
        lowered = jax.jit(forward_fn).lower(state)
        compiled = lowered.compile()
        cost = compiled.cost_analysis()
        
        # cost_analysis() returns a dict (not a list) on this JAX version
        if isinstance(cost, dict):
            flops = cost.get('flops', 0)
        elif isinstance(cost, (list, tuple)) and len(cost) > 0:
            flops = cost[0].get('flops', 0)
        else:
            flops = 0
        
        flops = int(flops)
            
    except Exception as e:
        print(f"  cost_analysis failed ({type(e).__name__}: {e})")
        flops = 0
    
    return flops


def count_batch_cells(batch):
    """Return the total cell count across all cochain dimensions in a batch."""
    if batch is None or not hasattr(batch, 'cochain_batches'):
        return 0
    total = 0
    for cb in batch.cochain_batches:
        num_cells = cb.num_cells
        # num_cells may be a vector per example when batched
        if hasattr(num_cells, 'shape'):
            total += int(np.sum(np.asarray(num_cells)))
        else:
            total += int(num_cells)
    return total


def count_batch_adjacencies(batch):
    """Return total adjacency-type entries across all cochain dimensions."""
    if batch is None or not hasattr(batch, 'cochain_batches'):
        return 0
    total = 0
    for cb in batch.cochain_batches:
        for name in ('up_senders', 'down_senders', 'boundary_senders', 'coboundary_senders'):
            arr = getattr(cb, name, None)
            if arr is not None:
                total += int(arr.shape[0])
    return total


def format_flops(flops):
    """Format FLOP count in human-readable form."""
    if flops == 0:
        return "N/A"
    elif flops >= 1e12:
        return f"{flops/1e12:.2f} TFLOPs"
    elif flops >= 1e9:
        return f"{flops/1e9:.2f} GFLOPs"
    elif flops >= 1e6:
        return f"{flops/1e6:.2f} MFLOPs"
    elif flops >= 1e3:
        return f"{flops/1e3:.2f} KFLOPs"
    else:
        return f"{flops:.0f} FLOPs"


def format_time(ms):
    """Format time in ms."""
    if ms >= 1000:
        return f"{ms/1000:.2f} s"
    elif ms >= 1:
        return f"{ms:.2f} ms"
    else:
        return f"{ms*1000:.1f} µs"
# ---------------------------------------------------------------------------
# OptunaHPO — fluent builder
# ---------------------------------------------------------------------------

class OptunaHPO:
    """
    Fluent-builder wrapper around an Optuna hyperparameter search.

    Constructor (required)
    ----------------------
    model_name : str
        Key used for logging and default HP-space lookup in
        ``DEFAULT_HP_SPACES``.
    model_factory : callable  (hp: dict, seed: int) -> nnx.Module
        Builds and returns a model given a hyperparameter dict (without
        'lr') and a random seed.  Use ``make_factories(num_species)``
        to obtain ready-made factories.
    train_batches : list
        Padded training batches from the first CV fold.
    val_batches : list
        Padded validation batches from the first CV fold.

    Configurators (optional, each returns self)
    -------------------------------------------
    .with_search_space(hp_space)
        Override the hyperparameter search-space dict.
        Defaults to ``DEFAULT_HP_SPACES[model_name]``.

    .with_training(n_trials, epochs_per_trial, seed, verbose)
        Set trial count, epoch budget per trial, random seed,
        and per-trial console output.

    .with_optuna(pruner, sampler)
        Plug in custom Optuna pruner and/or sampler objects.

    Execute
    -------
    .run() -> optuna.Study
    """

    def __init__(
        self,
        model_name: str,
        model_factory,
        train_batches: list,
        val_batches: list,
    ):
        self.model_name    = model_name
        self.model_factory = model_factory
        self.train_batches = train_batches
        self.val_batches   = val_batches

        # defaults — overridden by the with_* methods below
        self.hp_space         = DEFAULT_HP_SPACES.get(model_name, {})
        self.n_trials         = 50
        self.epochs_per_trial = 50
        self.seed             = 0
        self.verbose          = True
        self.pruner           = optuna.pruners.MedianPruner(
                                        n_startup_trials=5,
                                        n_warmup_steps=30,
                                        interval_steps=2
                                        )
        self.sampler          = TPESampler(seed=0)
        # AdamW + reduce_on_plateau scheduler config
        self.weight_decay             = 1e-4
        self.use_reduce_on_plateau    = True
        self.plateau_factor           = 0.7
        self.plateau_patience         = 10
        self.plateau_accumulation_size = 1
        self.plateau_min_scale        = 1e-5

        # trial-level benchmarking metrics (optional)
        self.benchmark_per_trial = True

    # --- configurators -------------------------------------------------------

    def with_search_space(self, hp_space: dict) -> 'OptunaHPO':
        """Replace the hyperparameter search-space dictionary."""
        self.hp_space = hp_space
        return self

    def with_training(
        self,
        n_trials: int = 30,
        epochs_per_trial: int = 20,
        seed: int = 0,
        verbose: bool = True,
        benchmark_per_trial: bool = True,
    ) -> 'OptunaHPO':
        """Set trial count, epoch budget, random seed, and verbosity."""
        self.benchmark_per_trial = benchmark_per_trial

        self.n_trials         = n_trials
        self.epochs_per_trial = epochs_per_trial
        self.seed             = seed
        self.verbose          = verbose
        self.sampler          = TPESampler(seed=seed)   # re-seed the default sampler
        return self

    def with_optuna(self, pruner=None, sampler=None) -> 'OptunaHPO':
        """Plug in a custom Optuna pruner and/or sampler."""
        if pruner  is not None: self.pruner  = pruner
        if sampler is not None: self.sampler = sampler
        return self

    def with_scheduler_config(
        self,
        weight_decay: float = 1e-4,
        use_reduce_on_plateau: bool = True,
        plateau_factor: float = 0.5,
        plateau_patience: int = 5,
        plateau_accumulation_size: int = 1,
        plateau_min_scale: float = 1e-4,
    ) -> 'OptunaHPO':
        """Configure AdamW weight-decay and the reduce_on_plateau scheduler.

        Parameters
        ----------
        weight_decay : float
            L2 regularisation for AdamW.
        use_reduce_on_plateau : bool
            Toggle the plateau scheduler on/off.
        plateau_factor : float
            LR reduction factor when a plateau is detected.
        plateau_patience : int
            Steps without improvement before reducing.
        plateau_accumulation_size : int
            Loss observations to average before each plateau check.
        plateau_min_scale : float
            Minimum allowed LR scale factor.
        """
        self.weight_decay              = weight_decay
        self.use_reduce_on_plateau     = use_reduce_on_plateau
        self.plateau_factor            = plateau_factor
        self.plateau_patience          = plateau_patience
        self.plateau_accumulation_size = plateau_accumulation_size
        self.plateau_min_scale         = plateau_min_scale
        return self

    # --- internal objective --------------------------------------------------

    def _objective(self, trial: optuna.Trial) -> float:
        hp = _sample_hp(trial, self.hp_space)
        lr = hp.pop('lr')
        # allow weight_decay & patience to be part of the search space; fall
        # back to builder defaults if they aren't specified.
        weight_decay = hp.pop('weight_decay', self.weight_decay)
        patience = hp.pop('patience', self.plateau_patience)

        model     = self.model_factory(hp, seed=self.seed + trial.number)
        tx = make_optimizer(
            lr,
            weight_decay=weight_decay,
            use_reduce_on_plateau=self.use_reduce_on_plateau,
            plateau_factor=self.plateau_factor,
            plateau_patience=patience,
            plateau_accumulation_size=self.plateau_accumulation_size,
            plateau_min_scale=self.plateau_min_scale,
        )
        optimizer = nnx.Optimizer(model, tx)

        if self.benchmark_per_trial and self.train_batches:
            perf_batch = self.train_batches[0]
            bench = benchmark_model(model, perf_batch, num_warmup=1, num_runs=3)
            flops = count_flops(model, perf_batch)
            num_cells = count_batch_cells(perf_batch)
            num_adjacencies = count_batch_adjacencies(perf_batch)
            trial.set_user_attr('bench', bench)
            trial.set_user_attr('flops', flops)
            trial.set_user_attr('n_params', train_utils.count_parameters(model))
            trial.set_user_attr('num_cells', num_cells)
            trial.set_user_attr('num_adjacencies', num_adjacencies)
        else:
            bench = {}
            flops = 0
            num_cells = 0
            num_adjacencies = 0

        if self.verbose:
            mean_time_ms = bench.get('mean_time_ms')
            mean_time_per_cell = float(mean_time_ms) / num_cells if (mean_time_ms is not None and num_cells > 0) else None
            flops_per_cell = float(flops) / num_cells if num_cells > 0 else None
            mean_time_per_adj = float(mean_time_ms) / num_adjacencies if (mean_time_ms is not None and num_adjacencies > 0) else None
            flops_per_adj = float(flops) / num_adjacencies if num_adjacencies > 0 else None
            print(
                f"[{self.model_name}] trial {trial.number:>3d} pre-train benchmark "
                f"time={mean_time_ms:.1f}ms std={bench.get('std_time_ms', 0):.1f}ms thr={bench.get('throughput', 0):.1f}it/s "
                f"cells={num_cells} adj={num_adjacencies} "
                f"ms/cell={mean_time_per_cell if mean_time_per_cell is not None else 'N/A'} "
                f"flop/cell={flops_per_cell if flops_per_cell is not None else 'N/A'} "
                f"ms/adj={mean_time_per_adj if mean_time_per_adj is not None else 'N/A'} "
                f"flop/adj={flops_per_adj if flops_per_adj is not None else 'N/A'}"
            )

        try:
            _, val_hist = train_utils.train_multitask(
                model, optimizer,
                self.train_batches,
                val_batches=self.val_batches,
                epochs=self.epochs_per_trial,
                disable_saving=True,
                verbose=False,
            )
        except Exception as e:
            print(f"[{self.model_name}] trial {trial.number:>3d} failed during training: {e}")
            raise

        best_val = min(val_hist['total']) if val_hist else float('inf')

        trial.report(best_val, step=self.epochs_per_trial)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if self.verbose:
            # Pull benchmark metrics from trial attributes if available.
            bench = trial.user_attrs.get('bench', {}) if hasattr(trial, 'user_attrs') else {}
            mean_time_ms = bench.get('mean_time_ms')
            std_time_ms = bench.get('std_time_ms')
            throughput = bench.get('throughput')

            n_params = train_utils.count_parameters(model)
            num_cells = trial.user_attrs.get('num_cells', 0) if hasattr(trial, 'user_attrs') else 0
            num_adjacencies = trial.user_attrs.get('num_adjacencies', 0) if hasattr(trial, 'user_attrs') else 0
            flops = trial.user_attrs.get('flops', 0) if hasattr(trial, 'user_attrs') else 0

            mean_time_per_cell = float(mean_time_ms) / num_cells if (mean_time_ms is not None and num_cells > 0) else None
            flops_per_cell = float(flops) / num_cells if num_cells > 0 else None
            mean_time_per_adj = float(mean_time_ms) / num_adjacencies if (mean_time_ms is not None and num_adjacencies > 0) else None
            flops_per_adj = float(flops) / num_adjacencies if num_adjacencies > 0 else None

            log_parts = [
                f"[{self.model_name}] trial {trial.number:>3d}",
                f"val={best_val:.5f}",
                f"params={n_params:,}",
                f"lr={lr:.2e}",
            ]
            if mean_time_ms is not None:
                log_parts.append(f"time={mean_time_ms:.1f}ms")
            if std_time_ms is not None:
                log_parts.append(f"std={std_time_ms:.1f}ms")
            if throughput is not None:
                log_parts.append(f"thr={throughput:.1f}it/s")
            if num_cells > 0:
                log_parts.append(f"cells={num_cells}")
            if num_adjacencies > 0:
                log_parts.append(f"adj={num_adjacencies}")
            if mean_time_per_cell is not None:
                log_parts.append(f"ms/cell={mean_time_per_cell:.4f}")
            if flops_per_cell is not None:
                log_parts.append(f"flop/cell={flops_per_cell:.1f}")
            if mean_time_per_adj is not None:
                log_parts.append(f"ms/adj={mean_time_per_adj:.4f}")
            if flops_per_adj is not None:
                log_parts.append(f"flop/adj={flops_per_adj:.1f}")

            # include raw hp in log
            hp_str = ", ".join(f"{k}={v}" for k, v in hp.items())
            if hp_str:
                log_parts.append(hp_str)

            print(" | ".join(log_parts))

        return best_val

    # --- execute -------------------------------------------------------------

    def run(self, save_top_n: int = 0) -> optuna.Study:
        """Create the study, run all trials, print a summary, return the study.

        Parameters
        ----------
        save_top_n : int
            If >0, compute additional statistics (parameter count, FLOPs,
            forward-pass time) for the top-N trials and write them to a
            JSON file named ``{model_name}_optuna.json``.  The first training
            batch is used for benchmarking; the file contains an array of
            trial summaries.
        """
        study = optuna.create_study(
            direction='minimize',
            study_name=self.model_name,
            sampler=self.sampler,
            pruner=self.pruner,
        )
        study.optimize(self._objective, n_trials=self.n_trials)

        print(f"\n{'='*60}")
        print(f"[{self.model_name}] HPO complete — best val loss: {study.best_value:.5f}")
        print(f"  Best params: {study.best_params}")
        print(f"{'='*60}\n")

        # optionally save top-N trial configurations with benchmarks
        if save_top_n > 0:
            try:
                import json
            except ImportError:
                print("WARNING: benchmark_utils unavailable, cannot save top trials info")
            else:
                # pick first batch for profiling
                batch = self.train_batches[0] if self.train_batches else None
                trials = sorted([t for t in study.trials if t.value is not None],
                                key=lambda t: t.value)[:save_top_n]
                records = []
                for t in trials:
                    hp = t.params.copy()
                    # the optimisation loop already consumes lr and may
                    # remove weight_decay/patience; when re‑building the
                    # model for profiling we must do the same so that the
                    # factory is not passed unexpected keywords.  note that
                    # some spaces do not include weight_decay or patience,
                    # hence the default None.
                    lr = hp.pop('lr', None)
                    hp.pop('weight_decay', None)
                    hp.pop('patience', None)

                    if t.user_attrs and 'n_params' in t.user_attrs:
                        n_params = t.user_attrs.get('n_params')
                        flops = t.user_attrs.get('flops', 0)
                        bench = t.user_attrs.get('bench', {})
                        num_cells = t.user_attrs.get('num_cells', 0)
                        num_adjacencies = t.user_attrs.get('num_adjacencies', 0)
                    else:
                        model = self.model_factory(hp, seed=self.seed + t.number)
                        n_params = train_utils.count_parameters(model)
                        flops = count_flops(model, batch) if batch is not None else 0
                        bench = benchmark_model(model, batch) if batch is not None else {}
                        num_cells = count_batch_cells(batch) if batch is not None else 0
                        num_adjacencies = count_batch_adjacencies(batch) if batch is not None else 0

                    mean_time_ms = bench.get('mean_time_ms') if bench is not None else None
                    if num_cells > 0 and mean_time_ms is not None:
                        mean_time_ms_per_cell = float(mean_time_ms) / float(num_cells)
                    else:
                        mean_time_ms_per_cell = None
                    flops_per_cell = float(flops) / float(num_cells) if num_cells > 0 else None

                    if num_adjacencies > 0 and mean_time_ms is not None:
                        mean_time_ms_per_adj = float(mean_time_ms) / float(num_adjacencies)
                    else:
                        mean_time_ms_per_adj = None
                    flops_per_adj = float(flops) / float(num_adjacencies) if num_adjacencies > 0 else None

                    rec = {
                        'trial': t.number,
                        'value': t.value,
                        'params': t.params,
                        'n_params': n_params,
                        'flops': format_flops(flops),
                        'num_cells': num_cells,
                        'num_adjacencies': num_adjacencies,
                        'flops_per_cell': flops_per_cell,
                        'mean_time_ms_per_cell': mean_time_ms_per_cell,
                        'flops_per_adjacency': flops_per_adj,
                        'mean_time_ms_per_adjacency': mean_time_ms_per_adj,
                        **bench,
                    }
                    records.append(rec)
                fname = f"{self.model_name}_optuna.json"
                with open(fname, 'w') as fp:
                    json.dump(records, fp, indent=2)
                print(f"Saved top-{save_top_n} trial info to {fname}")

        return study
