"""
Training utilities for equivariant per-atom predictions.

Sections:
    1. Checkpointing   — save / load / count_parameters
    2. Optimizer        — make_optimizer (AdamW + ReduceOnPlateau)
    3. Loss             — compute_multitask_loss (4-property norm-based)
    4. Training loop    — train_multitask (composed from small helpers)
    5. Legacy I/O       — save_loss_history / load_loss_history (text files)

L=2 Tensor basis: [xy, xz, yz, (xx-yy)/2, zz]  (traceless).
Frobenius weights [2, 2, 2, 2, 1.5] ensure rotation-invariant loss.
"""

import functools
import json
import os
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax import nnx

from qtnet.jax_models.representations import compute_frobenius_norm

# ═══════════════════════════════════════════════════════════════════════
# 1. Checkpointing
# ═══════════════════════════════════════════════════════════════════════

_CHECKPOINTER = ocp.PyTreeCheckpointer()


def _state_to_numpy(state):
    """Convert JAX arrays in a pytree to numpy for Orbax serialization."""
    def _convert(x):
        if hasattr(x, 'dtype') and jnp.issubdtype(x.dtype, jax.dtypes.prng_key):
            return np.asarray(jax.random.key_data(x))
        if hasattr(x, 'shape'):
            return np.asarray(x)
        return x
    return jax.tree_util.tree_map(_convert, state)


def _resolve_checkpoint_path(path: str) -> str:
    """Strip legacy .pkl suffix when the directory form exists."""
    if path.endswith('.pkl') and not os.path.exists(path):
        stripped = path[:-4]
        if os.path.exists(stripped):
            return stripped
    return path


def save_checkpoint(model: nnx.Module, checkpoint_dir: str,
                    filename: str = "model") -> str:
    """Save model state to *checkpoint_dir/filename* (Orbax directory)."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    if filename.endswith('.pkl'):
        filename = filename[:-4]
    path = os.path.join(checkpoint_dir, filename)
    np_state = _state_to_numpy(nnx.state(model))
    _CHECKPOINTER.save(path, np_state, force=os.path.exists(path))
    print(f"Saved checkpoint to {path}")
    return path


def save_state_checkpoint(state, checkpoint_dir: str,
                          filename: str = "model") -> str:
    """Save a raw state pytree (e.g. best-so-far snapshot)."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    if filename.endswith('.pkl'):
        filename = filename[:-4]
    path = os.path.join(checkpoint_dir, filename)
    np_state = _state_to_numpy(state)
    _CHECKPOINTER.save(path, np_state, force=os.path.exists(path))
    return path


def load_checkpoint(model: nnx.Module, checkpoint_path: str) -> nnx.Module:
    """Load model state from an Orbax checkpoint directory."""
    checkpoint_path = _resolve_checkpoint_path(checkpoint_path)
    ref_state = _state_to_numpy(nnx.state(model))
    restored = _CHECKPOINTER.restore(checkpoint_path, item=ref_state)
    nnx.update(model, restored)
    print(f"Loaded checkpoint from {checkpoint_path}")
    return model


def count_parameters(model: nnx.Module) -> int:
    """Count total trainable parameters."""
    return int(sum(
        np.prod(leaf.shape) for leaf in jax.tree_util.tree_leaves(nnx.state(model))
        if hasattr(leaf, 'shape')
    ))


# ═══════════════════════════════════════════════════════════════════════
# 2. Optimizer
# ═══════════════════════════════════════════════════════════════════════

def make_optimizer(
    lr: float,
    weight_decay: float = 1e-4,
    use_reduce_on_plateau: bool = True,
    plateau_factor: float = 0.5,
    plateau_patience: int = 50,
    plateau_accumulation_size: int = 1,
    plateau_min_scale: float = 5e-3,
) -> optax.GradientTransformation:
    """Build AdamW, optionally chained with ReduceOnPlateau.

    When *use_reduce_on_plateau* is True the returned transform is an
    ``optax.chain(adamw, reduce_on_plateau)`` whose ``update`` expects a
    ``value=`` keyword (the training loss).  The scheduler state is
    embedded in the optimizer state so it is automatically saved and
    restored with regular optimizer checkpoints.

    Set *plateau_accumulation_size* to the number of training batches
    per epoch so the scheduler evaluates once per epoch.
    """
    base = optax.adamw(learning_rate=lr, weight_decay=weight_decay)
    if not use_reduce_on_plateau:
        return base
    return optax.chain(
        base,
        optax.contrib.reduce_on_plateau(
            factor=plateau_factor,
            patience=plateau_patience,
            accumulation_size=plateau_accumulation_size,
            min_scale=plateau_min_scale,
        ),
    )


def get_lr_scale(optimizer: nnx.Optimizer) -> float:
    """Extract the ReduceOnPlateau scale from a chained optimizer.

    Returns 1.0 when the optimizer does not contain a plateau scheduler.
    """
    try:
        # chain(adamw, reduce_on_plateau) -> opt_state is a 2-tuple;
        # second element is the ReduceLROnPlateauState.
        return float(optimizer.opt_state[1].scale.value)
    except (IndexError, AttributeError, TypeError):
        return 1.0


# ═══════════════════════════════════════════════════════════════════════
# 3. Loss
# ═══════════════════════════════════════════════════════════════════════

# Frobenius weights for L=2 tensor in 5-component basis
_FROBENIUS_L2 = jnp.array([2.0, 2.0, 2.0, 2.0, 1.5])

# 4 rotation-invariant physical property groups
PHYSICAL_PROPERTY_NAMES = ['N', 'LI', 'Mu', 'Q']
ATOMIC_PROPERTY_NAMES = PHYSICAL_PROPERTY_NAMES  # backward compat

LOSS_KEYS = ['total'] + PHYSICAL_PROPERTY_NAMES


def compute_multitask_loss(
    predictions: Dict[str, jnp.ndarray],
    y: jnp.ndarray,
    node_mask: jnp.ndarray,
    element_weights: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Norm-based multitask loss over 4 physical properties.

    L = (1/4) [MSE(N) + MSE(LI) + mean(||Δμ||²) + mean(||ΔQ||²_F)]
    """
    pred_concat = jnp.concatenate(
        [predictions['scalars'], predictions['vectors'], predictions['tensors']],
        axis=-1,
    )
    delta = pred_concat - y

    N_err  = delta[:, 0] ** 2
    LI_err = delta[:, 1] ** 2
    Mu_err = jnp.sum(delta[:, 2:5] ** 2, axis=-1)
    Q_err  = jnp.sum(delta[:, 5:] ** 2 * _FROBENIUS_L2[None, :], axis=-1)

    errors = jnp.stack([N_err, LI_err, Mu_err, Q_err], axis=-1)  # (n, 4)
    mask = node_mask[:, None]

    if element_weights is not None:
        w = element_weights[:, None]
        weighted = jnp.where(mask, w * errors, 0.0)
        w_sum = jnp.sum(jnp.where(node_mask, element_weights, 0.0)) + 1e-8
        per_property = jnp.sum(weighted, axis=0) / w_sum
    else:
        n_valid = jnp.sum(node_mask) + 1e-8
        per_property = jnp.sum(jnp.where(mask, errors, 0.0), axis=0) / n_valid

    total_loss = jnp.mean(per_property)

    losses = {name: per_property[i] for i, name in enumerate(PHYSICAL_PROPERTY_NAMES)}
    losses['total'] = total_loss
    return total_loss, losses

def compute_multitask_loss_mse(
    predictions: Dict[str, jnp.ndarray],
    y: jnp.ndarray,
    node_mask: jnp.ndarray,
    element_weights: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Per-property MSE loss over 4 physical properties.

    Each property group is averaged uniformly over atoms *and* components,
    with no Frobenius weighting on the tensor components.

    L = (1/4) [MSE(N) + MSE(LI) + MSE_flat(Mu) + MSE_flat(Q)]
    """
    pred_concat = jnp.concatenate(
        [predictions['scalars'], predictions['vectors'], predictions['tensors']],
        axis=-1,
    )
    delta = pred_concat - y
    mask = node_mask[:, None]

    N_sq  = delta[:, 0:1] ** 2           # (n, 1)
    LI_sq = delta[:, 1:2] ** 2           # (n, 1)
    Mu_sq = delta[:, 2:5] ** 2           # (n, 3)
    Q_sq  = delta[:, 5:] ** 2            # (n, 5)

    if element_weights is not None:
        w = element_weights[:, None]
        w_sum = jnp.sum(jnp.where(node_mask, element_weights, 0.0)) + 1e-8
        N_loss  = jnp.sum(jnp.where(mask, w * N_sq,  0.0)) / w_sum
        LI_loss = jnp.sum(jnp.where(mask, w * LI_sq, 0.0)) / w_sum
        Mu_loss = jnp.sum(jnp.where(mask, w * Mu_sq, 0.0)) / (w_sum * 3)
        Q_loss  = jnp.sum(jnp.where(mask, w * Q_sq,  0.0)) / (w_sum * 5)
    else:
        n_valid = jnp.sum(node_mask) + 1e-8
        N_loss  = jnp.sum(jnp.where(mask, N_sq,  0.0)) / n_valid
        LI_loss = jnp.sum(jnp.where(mask, LI_sq, 0.0)) / n_valid
        Mu_loss = jnp.sum(jnp.where(mask, Mu_sq, 0.0)) / (n_valid * 3)
        Q_loss  = jnp.sum(jnp.where(mask, Q_sq,  0.0)) / (n_valid * 5)

    per_property = jnp.stack([N_loss, LI_loss, Mu_loss, Q_loss])
    total_loss = jnp.mean(per_property)

    losses = {name: per_property[i] for i, name in enumerate(PHYSICAL_PROPERTY_NAMES)}
    losses['total'] = total_loss
    return total_loss, losses


# ═══════════════════════════════════════════════════════════════════════
# 4. Training loop
# ═══════════════════════════════════════════════════════════════════════

def prefetch_to_device(iterable, num_prefetch=2):
    """Overlap host→device transfer with computation via a look-ahead queue."""
    queue = deque()
    it = iter(iterable)

    def _enqueue(n):
        for _ in range(n):
            try:
                queue.append(jax.device_put(next(it)))
            except StopIteration:
                break

    _enqueue(num_prefetch)
    while queue:
        yield queue.popleft()
        _enqueue(1)


# ── Tiny I/O helpers used inside the training loop ────────────────────

def save_stats(molecular_stats: Dict, atomic_stats: Dict, output_path: str):
    """Write fold statistics to JSON."""
    if hasattr(molecular_stats, 'to_dict'):
        molecular_stats = molecular_stats.to_dict()
    if hasattr(atomic_stats, 'to_dict'):
        atomic_stats = atomic_stats.to_dict()
    with open(output_path, 'w') as f:
        json.dump({'molecular_stats': molecular_stats,
                    'atomic_stats': atomic_stats}, f, indent=2)


def save_loss_json(loss_dict: Dict[str, float], output_path: str):
    """Dump a single-step loss dictionary to JSON."""
    with open(output_path, 'w') as f:
        json.dump(loss_dict, f, indent=2)


# ── History management ────────────────────────────────────────────────

def _load_history(loss_dir: Optional[str], verbose: bool) -> Optional[Dict]:
    """Try to load loss_history.json from *loss_dir*; return None on failure."""
    if loss_dir is None:
        return None
    path = os.path.join(loss_dir, "loss_history.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r') as f:
            hist = json.load(f)
        if verbose:
            print(f"Loaded existing loss history from {path}")
        return hist
    except Exception:
        return None


def _init_histories(
    initial_history: Optional[Dict],
    has_val: bool,
) -> Tuple[Dict[str, List], Optional[Dict[str, List]], float]:
    """Build empty train/val history dicts, optionally warm-started.

    Returns (train_history, val_history, best_val_loss_so_far).
    """
    train_history = {k: [] for k in LOSS_KEYS}
    val_history = {k: [] for k in LOSS_KEYS} if has_val else None

    best_val = float('inf')

    if initial_history is not None and isinstance(initial_history, dict):
        for k in LOSS_KEYS:
            src = initial_history.get('train', {}).get(k)
            if isinstance(src, list):
                train_history[k] = list(src)

        if val_history is not None:
            for k in LOSS_KEYS:
                src = initial_history.get('val', {}).get(k)
                if isinstance(src, list):
                    val_history[k] = list(src)

        try:
            best_val = float(initial_history.get('best_val_so_far', float('inf')))
        except Exception:
            pass

    return train_history, val_history, best_val


def _save_history(
    train_history: Dict,
    val_history: Optional[Dict],
    best_val: float,
    loss_dir: Optional[str],
    verbose: bool = False,
    lr_scale: float = 1.0,
):
    """Write the full loss history JSON to *loss_dir*."""
    if loss_dir is None:
        return
    os.makedirs(loss_dir, exist_ok=True)
    hist: Dict[str, Any] = {'train': train_history}
    if val_history is not None:
        hist['val'] = val_history
        hist['best_val_so_far'] = best_val
    if lr_scale < 1.0:
        hist['lr_scale'] = lr_scale
    path = os.path.join(loss_dir, "loss_history.json")
    with open(path, 'w') as f:
        json.dump(hist, f, indent=2)
    if verbose:
        print(f"  Loss history saved to {path}")


# ── JIT step builders ─────────────────────────────────────────────────

def _make_train_step(element_weight_array, pass_loss_value=False, use_mse_loss=False):
    """Return a JIT-compiled single training step function.

    When *pass_loss_value* is True (chained ReduceOnPlateau), the batch
    training loss is forwarded to the optimizer via ``value=`` so that
    the plateau scheduler can track it.
    When *use_mse_loss* is True, uses per-component MSE instead of the
    default norm-based loss (no Frobenius weighting on tensors).
    """
    _loss_fn = compute_multitask_loss_mse if use_mse_loss else compute_multitask_loss

    def _step(model, optimizer, batch):
        def loss_fn(model):
            preds = model(batch)
            y = batch.cochain_batches[0].y
            ew = None
            if element_weight_array is not None:
                Z = batch.cochain_batches[0].static['Z']
                ew = element_weight_array[Z]
            total, losses = _loss_fn(
                preds, y, preds['x_mask'], element_weights=ew,
            )
            return total, losses

        grads, losses = nnx.grad(loss_fn, has_aux=True)(model)
        if pass_loss_value:
            optimizer.update(grads, value=losses['total'])
        else:
            optimizer.update(grads)
        return losses

    return nnx.jit(_step)


def _make_val_step(element_weight_array, use_mse_loss=False):
    """Return a JIT-compiled single validation step function."""
    _loss_fn = compute_multitask_loss_mse if use_mse_loss else compute_multitask_loss
    def _step(model, batch):
        preds = model(batch)
        y = batch.cochain_batches[0].y
        ew = None
        if element_weight_array is not None:
            Z = batch.cochain_batches[0].static['Z']
            ew = element_weight_array[Z]
        _, losses = _loss_fn(
            preds, y, preds['x_mask'], element_weights=ew,
        )
        return losses

    return nnx.jit(_step)


# ── Per-epoch runners ─────────────────────────────────────────────────

def _run_train_epoch(train_step, model, optimizer, batches):
    """Run one training epoch, return per-key loss lists."""
    epoch_losses = {k: [] for k in LOSS_KEYS}
    for batch in prefetch_to_device(batches):
        losses = train_step(model, optimizer, batch)
        for k in LOSS_KEYS:
            epoch_losses[k].append(float(losses[k]))
    return epoch_losses


def _run_val_epoch(val_step, model, batches):
    """Run one validation epoch, return per-key loss lists."""
    epoch_losses = {k: [] for k in LOSS_KEYS}
    for batch in prefetch_to_device(batches):
        losses = val_step(model, batch)
        for k in LOSS_KEYS:
            epoch_losses[k].append(float(losses[k]))
    return epoch_losses


# ── Optuna integration ────────────────────────────────────────────────

def _optuna_report(trial, avg_val_total: float, epoch: int):
    """Report to Optuna and raise TrialPruned when appropriate."""
    if trial is None:
        return
    try:
        trial.report(float(avg_val_total), epoch)
        import optuna
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    except Exception as e:
        # Re-raise pruning; swallow everything else
        if 'TrialPruned' in type(e).__name__:
            raise


# ── Main entry point ──────────────────────────────────────────────────

def train_multitask(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    train_batches: List,
    val_batches: Optional[List] = None,
    epochs: int = 100,
    save_every: int = 100,
    checkpoint_dir: Optional[str] = None,
    stats_output: Optional[str] = None,
    molecular_stats: Optional[Dict] = None,
    atomic_stats: Optional[Dict] = None,
    verbose: bool = True,
    print_every: int = 10,
    disable_saving: bool = False,
    loss_dir: Optional[str] = None,
    initial_history: Optional[Dict] = None,
    use_mse_loss: bool = False,
    trial: object = None,
    element_weight_array: Optional[jnp.ndarray] = None,
    augment_fn: Optional[Callable] = None,
    augment_key: Optional[jnp.ndarray] = None,
    rotate_every: int = 0,
    start_epoch: int = 0,
    **kwargs,
) -> Tuple[Dict[str, List[float]], Optional[Dict[str, List[float]]]]:
    """Multitask training loop with checkpoint-resume support.

    ReduceOnPlateau is auto-detected from the optimizer: when
    ``make_optimizer`` was called with ``use_reduce_on_plateau=True`` the
    scheduler is chained inside the optimizer and its state is
    automatically saved and restored with regular optimizer checkpoints.

    Args:
        model / optimizer:  Flax NNX model and wrapped optimizer.
        train_batches / val_batches:  Pre-padded batch lists (numpy or jax).
        epochs:             *Total* number of epochs (including resumed ones).
        start_epoch:        Epoch offset when resuming (skips to the right
                            position; history loaded from ``loss_dir`` is kept
                            and new epochs are appended).
        save_every:         Checkpoint interval (epochs).
        checkpoint_dir:     Where to write model snapshots.
        loss_dir:           Where to write ``loss_history.json``.
        initial_history:    Explicit history dict; auto-loaded from *loss_dir*
                            when None.
        element_weight_array: Per-species weighting for the loss. Pass ``None``
                            to disable element weights (uniform weighting).
        use_mse_loss:       When True, use per-component MSE loss instead of
                            the default norm-based loss (no Frobenius weighting).
        augment_fn / augment_key / rotate_every: SO(3) augmentation.
        trial:              Optuna trial for pruning support.
        disable_saving:     Skip all file I/O (useful for quick debug runs).

    Returns:
        (train_history, val_history) — dicts mapping loss keys to per-epoch
        value lists.
    """
    # ── optional stats dump ───────────────────────────────────────────
    if (not disable_saving and stats_output is not None
            and molecular_stats is not None and atomic_stats is not None):
        save_stats(molecular_stats, atomic_stats, stats_output)
        if verbose:
            print(f"Fold stats written to {stats_output}")

    # ── detect chained ReduceOnPlateau in optimizer ───────────────────
    has_plateau = isinstance(
        optimizer.tx, optax.GradientTransformationExtraArgs,
    )
    

    # ── JIT-compiled step functions ───────────────────────────────────
    train_step = _make_train_step(
        element_weight_array, pass_loss_value=has_plateau, use_mse_loss=use_mse_loss,
    )
    val_step = _make_val_step(element_weight_array, use_mse_loss=use_mse_loss)

    # ── history (auto-load only on resume) ──────────────────────────────
    if initial_history is None and start_epoch > 0:
        initial_history = _load_history(loss_dir, verbose)

    train_history, val_history, best_val_loss = _init_histories(
        initial_history, has_val=(val_batches is not None),
    )
    best_state = None

    remaining_epochs = epochs - start_epoch
    if verbose:
        if start_epoch > 0:
            print(f"Resuming from epoch {start_epoch}")
        print(f"Training for {remaining_epochs} epochs "
              f"({len(train_batches)} batches/epoch)")
        print("=" * 72)

    # ── epoch loop ────────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs):
        # SO(3) augmentation
        if augment_fn is not None and rotate_every > 0 and epoch % rotate_every == 0:
            augment_key_local, augment_key = jax.random.split(augment_key)
            train_batches = augment_fn(train_batches, augment_key_local)

        # train
        epoch_losses = _run_train_epoch(
            train_step, model, optimizer, train_batches,
        )
        for k in LOSS_KEYS:
            train_history[k].append(np.mean(epoch_losses[k]))

        # validate
        avg_val_total = None
        if val_batches is not None:
            val_losses = _run_val_epoch(val_step, model, val_batches)
            for k in LOSS_KEYS:
                val_history[k].append(np.mean(val_losses[k]))
            avg_val_total = val_history['total'][-1]

            # best-model tracking
            if avg_val_total < best_val_loss:
                best_val_loss = avg_val_total
                best_state = jax.tree_util.tree_map(
                    lambda x: x.copy(), nnx.state(model),
                )

            # Optuna
            _optuna_report(trial, avg_val_total, epoch)

        # current LR scale (for logging / history)
        cur_lr_scale = get_lr_scale(optimizer)

        # checkpoint
        if (checkpoint_dir and not disable_saving
                and (epoch + 1) % save_every == 0):
            save_checkpoint(model, checkpoint_dir,
                            filename=f"model_epoch{epoch + 1}")
            save_state_checkpoint(
                _state_to_numpy(nnx.state(optimizer)),
                checkpoint_dir, filename=f"optimizer_epoch{epoch + 1}",
            )
            if best_state is not None:
                save_state_checkpoint(best_state, checkpoint_dir,
                                      filename="model_best_so_far")
            _save_history(train_history, val_history, best_val_loss,
                          loss_dir, verbose,
                          lr_scale=cur_lr_scale)
            if verbose:
                print(f"  Checkpoint saved at epoch {epoch + 1}")

        # logging
        if verbose and ((epoch + 1) % print_every == 0 or epoch == start_epoch):
            parts = [f"{k}={train_history[k][-1]:.4f}" for k in LOSS_KEYS]
            msg = f"Epoch {epoch + 1}/{epochs}: " + ", ".join(parts)
            if avg_val_total is not None:
                msg += f" | Val total {avg_val_total:.4f}"
            if cur_lr_scale < 1.0:
                msg += f" | LR scale {cur_lr_scale:.4f}"
            print(msg)

    # ── post-training saves ───────────────────────────────────────────
    if not disable_saving:
        # final best-state update
        if val_history is not None:
            final_val = val_history['total'][-1]
            if final_val < best_val_loss:
                best_val_loss = final_val
                best_state = jax.tree_util.tree_map(
                    lambda x: x.copy(), nnx.state(model),
                )

        _save_history(train_history, val_history, best_val_loss,
                      loss_dir, verbose,
                      lr_scale=get_lr_scale(optimizer))

        if checkpoint_dir:
            save_checkpoint(model, checkpoint_dir,
                            filename=f"model_epoch{epochs}")
            save_state_checkpoint(
                _state_to_numpy(nnx.state(optimizer)),
                checkpoint_dir, filename=f"optimizer_epoch{epochs}",
            )
            if best_state is not None:
                save_state_checkpoint(best_state, checkpoint_dir,
                                      filename="model_best_so_far")
            if verbose:
                print("Saved final checkpoint")

    if verbose:
        print("=" * 72)
        print("Training complete")

    return train_history, val_history


# ═══════════════════════════════════════════════════════════════════════
# 5. Molecular multi-property training utilities
# ═══════════════════════════════════════════════════════════════════════

def compute_molecular_multitask_loss(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    target_names: List[str],
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Weighted MSE loss for molecular multi-property prediction.

    Args:
        predictions : (G, num_targets)  model output, z-normalised scale
        targets     : (G, num_targets)  z-normalised ground-truth values
        target_names: list of property names matching column order
        weights     : optional per-property weight dict (default all 1.0)

    Returns:
        (total_loss, {prop: loss_val, 'total': total_loss})
    """
    losses: Dict[str, jnp.ndarray] = {}
    total = jnp.array(0.0)

    for i, name in enumerate(target_names):
        prop_loss = jnp.mean((predictions[:, i] - targets[:, i]) ** 2)
        losses[name] = prop_loss
        w = weights.get(name, 1.0) if weights else 1.0
        total = total + w * prop_loss

    losses['total'] = total
    return total, losses


def train_molecular_multitask(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    train_batches: List,
    train_targets: np.ndarray,
    val_batches: Optional[List],
    val_targets: Optional[np.ndarray],
    target_names: List[str],
    epochs: int = 200,
    loss_weights: Optional[Dict[str, float]] = None,
    save_every: int = 100,
    checkpoint_dir: Optional[str] = None,
    loss_dir: Optional[str] = None,
    verbose: bool = True,
    print_every: int = 10,
    disable_saving: bool = False,
    start_epoch: int = 0,
) -> Tuple[Dict[str, List[float]], Optional[Dict[str, List[float]]]]:
    """Training loop for molecular multi-property prediction.

    Args:
        model / optimizer   : Flax NNX model and wrapped optimizer.
        train_batches       : List of ComplexBatch (padded).
        train_targets       : (N_train, num_targets) z-normalised numpy array,
                              rows ordered to match the batch sequence.
        val_batches / val_targets : Validation counterparts (or None).
        target_names        : Ordered property names, e.g. ['alpha','gap','U0','Cv'].
        epochs              : Total training epochs.
        loss_weights        : Per-property loss weight dict.
        save_every          : Checkpoint interval.
        checkpoint_dir      : Where to write model checkpoints.
        loss_dir            : Where to write loss_history.json.
        start_epoch         : Epoch offset for resuming.

    Returns:
        (train_history, val_history) dicts mapping loss keys → list of floats.
    """
    loss_keys = ['total'] + list(target_names)

    # Detect chained ReduceOnPlateau before defining JIT step functions so
    # the Python-level branch is resolved at trace time (not at runtime).
    has_plateau = isinstance(optimizer.tx, optax.GradientTransformationExtraArgs)

    # num_graphs must be a compile-time constant for jax.ops.segment_sum inside
    # the model, so it is declared static and extracted outside the jit call.
    @functools.partial(nnx.jit, static_argnames=('num_graphs',))
    def _train_step(model, optimizer, batch, batch_targets, num_graphs):
        graph_idx = batch.cochain_batches[0].owner_cochains

        def loss_fn(model):
            output = model(batch, graph_idx=graph_idx, num_graphs=num_graphs)
            preds  = output['predictions']
            total, losses = compute_molecular_multitask_loss(
                preds, batch_targets, target_names, loss_weights,
            )
            return total, losses

        grads, losses = nnx.grad(loss_fn, has_aux=True)(model)
        if has_plateau:
            optimizer.update(grads, value=losses['total'])
        else:
            optimizer.update(grads)
        return losses

    @functools.partial(nnx.jit, static_argnames=('num_graphs',))
    def _val_step(model, batch, batch_targets, num_graphs):
        graph_idx = batch.cochain_batches[0].owner_cochains
        output = model(batch, graph_idx=graph_idx, num_graphs=num_graphs)
        preds  = output['predictions']
        _, losses = compute_molecular_multitask_loss(
            preds, batch_targets, target_names, loss_weights,
        )
        return losses

    train_history: Dict[str, List[float]] = {k: [] for k in loss_keys}
    val_history: Optional[Dict[str, List[float]]] = (
        {k: [] for k in loss_keys} if val_batches is not None else None
    )
    best_val_loss = float('inf')
    best_state = None

    if verbose:
        print(f"Molecular multitask training: {epochs - start_epoch} epochs, "
              f"{len(train_batches)} batches/epoch, properties={target_names}")
        print("=" * 72)

    for epoch in range(start_epoch, epochs):
        # ---- train epoch ------------------------------------------------
        epoch_losses: Dict[str, List] = {k: [] for k in loss_keys}
        mol_idx = 0
        for batch in train_batches:
            n_mols = int(batch.num_complexes[0])
            batch_targets = jnp.asarray(train_targets[mol_idx:mol_idx + n_mols])
            mol_idx += n_mols
            losses = _train_step(model, optimizer, batch, batch_targets,
                                 num_graphs=n_mols)
            for k in loss_keys:
                if k in losses:
                    epoch_losses[k].append(float(losses[k]))
        for k in loss_keys:
            if epoch_losses[k]:
                train_history[k].append(float(np.mean(epoch_losses[k])))

        # ---- val epoch --------------------------------------------------
        avg_val = None
        if val_batches is not None and val_targets is not None:
            vl: Dict[str, List] = {k: [] for k in loss_keys}
            mol_idx = 0
            for batch in val_batches:
                n_mols = int(batch.num_complexes[0])
                batch_targets_v = jnp.asarray(val_targets[mol_idx:mol_idx + n_mols])
                mol_idx += n_mols
                losses = _val_step(model, batch, batch_targets_v,
                                   num_graphs=n_mols)
                for k in loss_keys:
                    if k in losses:
                        vl[k].append(float(losses[k]))
            for k in loss_keys:
                if vl[k]:
                    val_history[k].append(float(np.mean(vl[k])))
            avg_val = val_history['total'][-1]

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_state = jax.tree_util.tree_map(
                    lambda x: x.copy(), nnx.state(model)
                )

        # ---- checkpointing ----------------------------------------------
        if not disable_saving and checkpoint_dir and (epoch + 1) % save_every == 0:
            save_checkpoint(model, checkpoint_dir, f"model_epoch{epoch + 1}")
            if best_state is not None:
                save_state_checkpoint(best_state, checkpoint_dir, "model_best_so_far")
            if loss_dir:
                _save_history(train_history, val_history, best_val_loss,
                              loss_dir, verbose=False, lr_scale=get_lr_scale(optimizer))

        # ---- logging ----------------------------------------------------
        if verbose and ((epoch + 1) % print_every == 0 or epoch == start_epoch):
            parts = [f"{k}={train_history[k][-1]:.4f}" for k in loss_keys if train_history[k]]
            msg = f"Epoch {epoch + 1}/{epochs}: " + ", ".join(parts)
            if avg_val is not None:
                msg += f" | Val total {avg_val:.4f}"
            print(msg)

    # ---- post-training saves -------------------------------------------
    if not disable_saving:
        if loss_dir:
            _save_history(train_history, val_history, best_val_loss,
                          loss_dir, verbose, lr_scale=get_lr_scale(optimizer))
        if checkpoint_dir:
            save_checkpoint(model, checkpoint_dir, f"model_epoch{epochs}")
            if best_state is not None:
                save_state_checkpoint(best_state, checkpoint_dir, "model_best_so_far")

    if verbose:
        print("=" * 72)
        print("Training complete")

    return train_history, val_history


# ═══════════════════════════════════════════════════════════════════════
# 6. Legacy I/O (text-file histories)
# ═══════════════════════════════════════════════════════════════════════

def save_loss_history(loss_history: Dict[str, List[float]],
                      output_dir: str,
                      prefix: str = "train_loss"):
    """Save loss history to text files (one per loss key)."""
    os.makedirs(output_dir, exist_ok=True)
    for key, values in loss_history.items():
        with open(os.path.join(output_dir, f"{prefix}_{key}.txt"), 'w') as f:
            for v in values:
                f.write(f"{v}\n")
    print(f"Saved loss history to {output_dir}/")


def load_loss_history(input_dir: str,
                      prefix: str = "train_loss") -> Dict[str, List[float]]:
    """Load loss history from text files."""
    history = {}
    for fname in os.listdir(input_dir):
        if fname.startswith(prefix) and fname.endswith('.txt'):
            key = fname[len(prefix) + 1:-4]
            with open(os.path.join(input_dir, fname), 'r') as f:
                history[key] = [float(line) for line in f if line.strip()]
    return history
