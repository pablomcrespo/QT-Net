"""Test utilities for QT-Net.

Importable helpers for the notebooks under ``tests/``:

* Rotation utilities and equivariance checks (ported from the legacy
  ``unit_tests_legacy.ipynb`` notebook).
* Synthetic-data factories that build DataFrames matching the AIMEl atomic
  schema, plus convenience builders for ``Complex`` / ``ComplexBatch`` objects.
* Lightweight pipeline assertions used by ``data_loading.ipynb``.

The module is deliberately self-contained so it can be imported from
notebooks without ``pytest``.
"""

from __future__ import annotations

import dataclasses
import pickle
import sys
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp

# Ensure the repository ``src/`` directory is importable when the file is
# loaded from inside the ``tests/`` notebook directory.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
for _candidate in (_REPO_ROOT, os.path.join(_REPO_ROOT, 'src')):
    if os.path.isdir(_candidate) and _candidate not in sys.path:
        sys.path.insert(0, _candidate)

from qtnet.jax_models.representations import (  # noqa: E402
    Cochain,
    Complex,
    CochainBatch,
    ComplexBatch,
    row_to_complex,
    precompute_complexes,
    prepare_padded_batches,
    compute_frobenius_norm,
)


# Constants mirroring scripts/atomic/train_multitask.py
ALL_ELEMENTS = ['H', 'C', 'N', 'O']
TARGET_COLUMNS = [
    'N', 'LI', 'Mu_X', 'Mu_Y', 'Mu_Z',
    'Q_XY', 'Q_XZ', 'Q_YZ', 'Q_aniso', 'Q_ZZ',
]


# ---------------------------------------------------------------------------
# 1. Rotation & equivariance utilities
# ---------------------------------------------------------------------------

def random_rotation_matrix(key: jax.Array) -> jnp.ndarray:
    """Generate a uniformly-distributed proper 3x3 rotation matrix via QR.

    ``det(R) == +1``.
    """
    A = jax.random.normal(key, (3, 3))
    Q, _ = jnp.linalg.qr(A)
    Q = Q * jnp.sign(jnp.linalg.det(Q))
    return Q


def random_improper_rotation_matrix(key: jax.Array) -> jnp.ndarray:
    """Generate a 3x3 orthogonal matrix with ``det(R) == -1``.

    Constructed by drawing a proper rotation and flipping the sign of its
    first column.  Useful for testing parity / reflection equivariance.
    """
    R = random_rotation_matrix(key)
    return R.at[:, 0].set(-R[:, 0])


def random_orthogonal_matrix(key: jax.Array) -> jnp.ndarray:
    """Sample uniformly from O(3): proper rotation w.p. 1/2, improper otherwise."""
    k_sign, k_R = jax.random.split(key)
    R = random_rotation_matrix(k_R)
    flip = jax.random.bernoulli(k_sign).astype(R.dtype)  # 0 or 1
    return jnp.where(flip == 1, R.at[:, 0].set(-R[:, 0]), R)


def parity_matrix() -> jnp.ndarray:
    """Spatial inversion ``-I`` (det = -1)."""
    return -jnp.eye(3)


def rotation_matrix_axis_angle(axis: jnp.ndarray, angle: float) -> jnp.ndarray:
    axis = axis / jnp.linalg.norm(axis)
    K = jnp.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    return jnp.eye(3) + jnp.sin(angle) * K + (1 - jnp.cos(angle)) * (K @ K)


def rotate_positions(pos: jnp.ndarray, R: jnp.ndarray) -> jnp.ndarray:
    """Apply pos' = pos @ R.T (rows are 3-vectors)."""
    return pos @ R.T


def rotate_vectors(v: jnp.ndarray, R: jnp.ndarray, num_channels: int) -> jnp.ndarray:
    """Rotate L=1 features stored as ``(N, num_channels * 3)``."""
    n = v.shape[0]
    v_reshaped = v.reshape(n, num_channels, 3)
    return jnp.einsum('ij,ncj->nci', R, v_reshaped).reshape(n, -1)


def rotate_l2_tensor(T: jnp.ndarray, R: jnp.ndarray) -> jnp.ndarray:
    """Rotate one symmetric-traceless L=2 tensor in 5-component form.

    Component layout: ``[xy, xz, yz, (xx-yy)/2, zz]``.
    """
    xy, xz, yz, aniso, zz = T[0], T[1], T[2], T[3], T[4]
    xx_minus_yy = 2 * aniso
    xx = (-zz + xx_minus_yy) / 2
    yy = (-zz - xx_minus_yy) / 2
    T_full = jnp.array([
        [xx, xy, xz],
        [xy, yy, yz],
        [xz, yz, zz],
    ])
    T_rot = R @ T_full @ R.T
    return jnp.array([
        T_rot[0, 1],
        T_rot[0, 2],
        T_rot[1, 2],
        (T_rot[0, 0] - T_rot[1, 1]) / 2,
        T_rot[2, 2],
    ])


def rotate_l2_tensor_batch(T: jnp.ndarray, R: jnp.ndarray) -> jnp.ndarray:
    """Vectorised version: ``T`` shape ``(N, 5)``."""
    return jax.vmap(lambda t: rotate_l2_tensor(t, R))(T)


def rotate_l2_channels(T: jnp.ndarray, R: jnp.ndarray, num_channels: int) -> jnp.ndarray:
    """Rotate a stack of L=2 features stored as ``(N, num_channels * 5)``."""
    n = T.shape[0]
    T_reshaped = T.reshape(n, num_channels, 5)
    rotated = jax.vmap(
        lambda Tn: jax.vmap(lambda t: rotate_l2_tensor(t, R))(Tn)
    )(T_reshaped)
    return rotated.reshape(n, -1)


@dataclasses.dataclass
class EquivarianceResult:
    name: str
    passed: bool
    scalar_err: float
    vector_err: float
    tensor_err: float
    atol: float

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] {self.name}: "
            f"L0={self.scalar_err:.2e}  "
            f"L1={self.vector_err:.2e}  "
            f"L2={self.tensor_err:.2e}  "
            f"(atol={self.atol:.0e})"
        )


def check_equivariance(
    out_orig: Dict[str, jnp.ndarray],
    out_rot: Dict[str, jnp.ndarray],
    R: jnp.ndarray,
    num_vector_channels: int,
    num_tensor_channels: int,
    name: str,
    atol: float = 1e-4,
    mask: Optional[jnp.ndarray] = None,
) -> EquivarianceResult:
    """Compare ``model(rotate(x))`` against ``rotate(model(x))``.

    Each output dict must contain ``'scalars'`` (invariant), ``'vectors'``
    (L=1) and ``'tensors'`` (L=2).  When ``mask`` is provided, comparisons
    are restricted to rows where the mask is true (used to ignore padded
    cells whose outputs are unconstrained).
    """
    expected_v = rotate_vectors(out_orig['vectors'], R, num_vector_channels)
    expected_t = rotate_l2_channels(out_orig['tensors'], R, num_tensor_channels)

    def _err(a: jnp.ndarray, b: jnp.ndarray) -> float:
        if mask is None:
            return float(jnp.max(jnp.abs(a - b)))
        m = mask.astype(bool)
        if not bool(jnp.any(m)):
            return 0.0
        return float(jnp.max(jnp.abs((a - b)[m])))

    s_err = _err(out_orig['scalars'], out_rot['scalars'])
    v_err = _err(expected_v, out_rot['vectors'])
    t_err = _err(expected_t, out_rot['tensors'])
    passed = max(s_err, v_err, t_err) < atol
    return EquivarianceResult(name, passed, s_err, v_err, t_err, atol)


def _replace_static(c: Cochain, new_static: Dict[str, jnp.ndarray]) -> Cochain:
    """Return a new ``Cochain`` with ``static`` replaced (flax dataclass)."""
    return dataclasses.replace(c, static=new_static)


def _replace_cochainbatch_static(cb: CochainBatch, new_static: Dict[str, jnp.ndarray]) -> CochainBatch:
    return dataclasses.replace(cb, static=new_static)


def rotate_complex_batch(batch: ComplexBatch, R: jnp.ndarray) -> ComplexBatch:
    """Return a new ``ComplexBatch`` whose geometry is rotated by ``R``.

    Rotates node positions and edge / 2-cell gyration tensors in-place;
    leaves all index / mask / scalar arrays untouched.  Padded rows
    (zero-filled OOB slots) stay zero — rotation of zero is zero.

    ``R`` only needs to be orthogonal (``R Rᵀ = I``); the same routine
    handles proper rotations and improper rotations / parity (det = -1):
    positions transform as ``R · pos`` and gyration tensors as
    ``R G Rᵀ`` regardless of ``det(R)``.  This is consistent with full
    O(3) covariance.
    """
    R = jnp.asarray(R)
    new_cochain_batches = []
    for cb in batch.cochain_batches:
        if cb.static is None:
            new_cochain_batches.append(cb)
            continue

        new_static = dict(cb.static)
        if 'pos' in new_static:
            pos = jnp.asarray(new_static['pos'])
            new_static['pos'] = rotate_positions(pos, R)
        if 'G' in new_static:
            G = jnp.asarray(new_static['G'])
            new_static['G'] = rotate_l2_tensor_batch(G, R)
            # G_norm is invariant; recompute defensively if present
            if 'G_norm' in new_static:
                new_static['G_norm'] = compute_frobenius_norm(new_static['G'])
        new_cochain_batches.append(_replace_cochainbatch_static(cb, new_static))

    return dataclasses.replace(batch, cochain_batches=new_cochain_batches)


def translate_complex_batch(batch: ComplexBatch, t: jnp.ndarray) -> ComplexBatch:
    """Return a new ``ComplexBatch`` whose node positions are shifted by ``t``.

    Only the **real** (unmasked) atoms are translated; padded OOB rows stay
    at zero so they remain neutral inside ``segment_sum`` operations.

    ``EquivariantGNN`` consumes positions only through the relative
    differences ``pos_j - pos_i`` (and the gyration tensors built from
    them), so adding a constant ``t`` should leave **every** output
    unchanged.  This is the translation-invariance half of full E(3)
    coverage (the rotation/reflection half is handled by
    ``rotate_complex_batch``).
    """
    t = jnp.asarray(t).reshape(3)
    new_cochain_batches = []
    for cb in batch.cochain_batches:
        if cb.static is None or 'pos' not in cb.static:
            new_cochain_batches.append(cb)
            continue
        pos = jnp.asarray(cb.static['pos'])
        mask = jnp.asarray(cb.x_mask).astype(pos.dtype)[:, None]
        new_static = dict(cb.static)
        new_static['pos'] = pos + mask * t[None, :]
        new_cochain_batches.append(_replace_cochainbatch_static(cb, new_static))
    return dataclasses.replace(batch, cochain_batches=new_cochain_batches)


# ---------------------------------------------------------------------------
# 2. Synthetic data factories
# ---------------------------------------------------------------------------

# Per-element target distributions chosen so post-regularization assertions
# have well-defined finite means/stds across a small synthetic dataset.
_ELEMENT_PARAMS: Dict[str, Dict[str, Tuple[float, float]]] = {
    # element -> {field: (mean, std)}
    'H': {'N': (1.05, 0.30), 'LI': (0.40, 0.10), 'Mu': (0.10, 0.05), 'Q': (0.20, 0.10)},
    'C': {'N': (5.90, 0.40), 'LI': (3.50, 0.30), 'Mu': (0.30, 0.10), 'Q': (0.50, 0.20)},
    'N': {'N': (7.20, 0.50), 'LI': (4.30, 0.30), 'Mu': (0.40, 0.15), 'Q': (0.70, 0.25)},
    'O': {'N': (8.40, 0.55), 'LI': (5.10, 0.35), 'Mu': (0.50, 0.20), 'Q': (0.90, 0.30)},
}


def _sample_atom_targets(rng: np.random.Generator, atoms: Sequence[str]) -> Dict[str, np.ndarray]:
    """Sample per-atom QTAIM target arrays for a single molecule."""
    out: Dict[str, List[float]] = {col: [] for col in TARGET_COLUMNS}
    for sym in atoms:
        params = _ELEMENT_PARAMS[sym]
        n_mean, n_std = params['N']
        li_mean, li_std = params['LI']
        mu_mean, mu_std = params['Mu']
        q_mean, q_std = params['Q']
        out['N'].append(rng.normal(n_mean, n_std))
        out['LI'].append(rng.normal(li_mean, li_std))
        for axis in ('Mu_X', 'Mu_Y', 'Mu_Z'):
            out[axis].append(rng.normal(mu_mean, mu_std))
        for axis in ('Q_XY', 'Q_XZ', 'Q_YZ', 'Q_aniso', 'Q_ZZ'):
            out[axis].append(rng.normal(q_mean, q_std))
    return {k: np.asarray(v, dtype=np.float32) for k, v in out.items()}


def make_synthetic_row(
    rng: np.random.Generator,
    n_atoms: int = 5,
    elements: Sequence[str] = tuple(ALL_ELEMENTS),
    scaffold: str = 'scaffold_0',
) -> Dict[str, object]:
    """Build a single AIMEl-schema row as a plain dict.

    Always includes at least one of every element in ``elements`` so the
    per-atom statistics are well-defined; remaining atoms are sampled
    uniformly.  Positions are drawn from a Gaussian (Bohr units, the
    convention of ``row_to_complex``).
    """
    elements = list(elements)
    if n_atoms < len(elements):
        raise ValueError(f"n_atoms ({n_atoms}) must be >= len(elements) ({len(elements)})")
    atoms: List[str] = list(elements)
    extra = rng.choice(elements, size=n_atoms - len(elements)).tolist()
    atoms.extend(extra)
    rng.shuffle(atoms)

    # Spread positions over a few Bohr so cutoff=5.25 yields several edges.
    pos = rng.normal(loc=0.0, scale=2.0, size=(n_atoms, 3)).astype(np.float32)

    targets = _sample_atom_targets(rng, atoms)
    row: Dict[str, object] = {
        'atom': atoms,
        'position_x': pos[:, 0].tolist(),
        'position_y': pos[:, 1].tolist(),
        'position_z': pos[:, 2].tolist(),
        'Murcko_Scaffold': scaffold,
    }
    for col, arr in targets.items():
        row[col] = arr.tolist()
    return row


def make_synthetic_dataframe(
    n_molecules: int = 24,
    seed: int = 0,
    n_atoms_range: Tuple[int, int] = (8, 12),
    n_scaffolds: int = 6,
    elements: Sequence[str] = tuple(ALL_ELEMENTS),
) -> pd.DataFrame:
    """Build a small AIMEl-schema DataFrame for tests.

    Uses several distinct scaffold groups so ``GroupKFold`` produces
    non-trivial splits.
    """
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, object]] = []
    for i in range(n_molecules):
        n_atoms = int(rng.integers(n_atoms_range[0], n_atoms_range[1] + 1))
        scaffold = f'scaffold_{i % n_scaffolds}'
        rows.append(make_synthetic_row(rng, n_atoms=n_atoms, elements=elements, scaffold=scaffold))
    df = pd.DataFrame(rows)
    return df


def make_minimal_complex(
    seed: int = 0,
    n_atoms: int = 9,
    fully_connected: bool = False,
) -> Tuple[Complex, Dict[str, int]]:
    """Build a single ``Complex`` from a synthetic row."""
    rng = np.random.default_rng(seed)
    row = make_synthetic_row(rng, n_atoms=n_atoms)
    element_to_idx = {el: i for i, el in enumerate(ALL_ELEMENTS)}
    pd_row = pd.Series(row)
    cx = row_to_complex(
        pd_row,
        element_to_idx=element_to_idx,
        cutoff=5.25,
        max_neighbors=5 if not fully_connected else None,
        fully_connected=fully_connected,
        max_dim=2,
    )
    return cx, element_to_idx


def make_minimal_batch(
    n_molecules: int = 8,
    batch_size: int = 8,
    seed: int = 0,
    cutoff: float = 5.25,
    max_neighbors: Optional[int] = 5,
    fully_connected: bool = False,
    as_numpy: bool = False,
) -> ComplexBatch:
    """Run the synthetic df → ``precompute_complexes`` → ``prepare_padded_batches``
    pipeline and return the **first** padded batch.

    The targets are the 10 ``TARGET_COLUMNS`` listed above (matching the
    atomic training pipeline).
    """
    df = make_synthetic_dataframe(n_molecules=n_molecules, seed=seed)
    element_to_idx = {el: i for i, el in enumerate(ALL_ELEMENTS)}
    complexes = precompute_complexes(
        df, element_to_idx=element_to_idx,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        fully_connected=fully_connected,
        max_dim=2,
        verbose=False,
    )
    batches = prepare_padded_batches(
        complexes, df, target_columns=TARGET_COLUMNS,
        batch_size=batch_size, verbose=False, as_numpy=as_numpy,
    )
    return batches[0]


# ---------------------------------------------------------------------------
# 3. Pipeline assertion helpers
# ---------------------------------------------------------------------------

def assert_complex_invariants(c: Complex) -> None:
    """Cheap structural sanity checks for a single ``Complex``.

    Catches the common mistakes: ill-typed ``num_cells``, mismatched static
    shapes, out-of-range index arrays, self-loop edges.
    """
    assert c.dimension == len(c.cochains) - 1
    n_atoms = c.cochains[0].num_cells
    n_edges = c.cochains[1].num_cells
    assert n_atoms >= 0
    assert n_edges >= 0

    # Node static fields agree with num_cells
    node = c.cochains[0]
    assert node.static is not None and 'pos' in node.static and 'Z' in node.static
    assert node.static['pos'].shape == (n_atoms, 3)
    assert node.static['Z'].shape == (n_atoms,)

    # Edge static fields agree with num_edges
    edge = c.cochains[1]
    if n_edges > 0:
        assert edge.static is not None
        assert edge.static['G'].shape == (n_edges, 5)
        assert edge.static['atoms'].shape == (n_edges, 2)
        # No self-loops
        a, b = edge.static['atoms'][:, 0], edge.static['atoms'][:, 1]
        assert bool(jnp.all(a != b)), "edge connects an atom to itself"
        # Index arrays in range (Cochain.__post_init__ already validates,
        # but assert here for explicitness)
        assert bool(jnp.all((a >= 0) & (a < n_atoms)))
        assert bool(jnp.all((b >= 0) & (b < n_atoms)))


def assert_cv_disjoint(folds: List[Dict[str, np.ndarray]], n_rows: int) -> None:
    """Verify that train/test partitions are disjoint and exhaustive."""
    seen_folds = set()
    for f in folds:
        train = np.asarray(f['train_idx'])
        test = np.asarray(f['test_idx'])
        assert len(set(train.tolist()) & set(test.tolist())) == 0, "train/test overlap"
        union = sorted(set(train.tolist()) | set(test.tolist()))
        assert union == list(range(n_rows)), "train ∪ test does not cover all rows"
        seen_folds.add(int(f['fold']))
    assert seen_folds == set(range(len(folds))), "fold indices not contiguous"


def assert_padded_batch_shapes(batch: ComplexBatch, target_columns: Sequence[str]) -> None:
    """Sanity-check the padded ``ComplexBatch`` produced by
    ``prepare_padded_batches``.

    * Each cochain has at least one slot beyond the real cells (the OOB row).
    * ``x``, ``y`` and ``x_mask`` all share the leading dimension.
    * Padded rows are zero (masked off) — ``x_mask`` is False for at least
      the final OOB slot.
    * Target ``y`` (dim-0) has width ``len(target_columns)``.
    """
    for d, cb in enumerate(batch.cochain_batches):
        assert cb.x is not None, f"dim {d} has no features"
        assert cb.x_mask is not None, f"dim {d} has no x_mask"
        assert cb.x.shape[0] == cb.x_mask.shape[0]
        real_cells = int(jnp.sum(cb.x_mask))
        # Padded layout reserves at least one OOB row; the global max across
        # batches sets the row count, so per-batch real count may be lower.
        assert cb.x.shape[0] >= real_cells + 1, (
            f"dim {d}: rows {cb.x.shape[0]} < real_cells+1 ({real_cells + 1})"
        )
        # The last row must be the OOB slot — masked off and zero.
        assert not bool(cb.x_mask[-1]), f"dim {d}: trailing OOB row is unmasked"
        assert bool(jnp.allclose(jnp.asarray(cb.x[-1]), 0.0)), (
            f"dim {d}: trailing OOB row is non-zero"
        )

    node_batch = batch.cochain_batches[0]
    assert node_batch.y is not None
    assert node_batch.y.shape[1] == len(target_columns), (
        f"y width {node_batch.y.shape[1]} != len(target_columns) {len(target_columns)}"
    )
    assert node_batch.y.shape[0] == node_batch.x.shape[0]


def assert_post_regularization_stats(
    df_reg: pd.DataFrame,
    atomic_stats: pd.DataFrame,
    atol_mean: float = 5e-2,
    atol_std: float = 1e-1,
) -> None:
    """For each element present, regularized N/LI should have ~zero mean
    and ~unit std on the training rows; Mu_*/Q_* should have ~unit RMS.

    Tolerances are loose because the synthetic dataset is small (so sample
    statistics differ slightly from the population stats fed into the
    regulariser).
    """
    # Flatten per-row arrays into per-atom records keyed by element.
    rows: List[Dict[str, object]] = []
    for _, r in df_reg.iterrows():
        for i, sym in enumerate(r['atom']):
            rec = {'atom': sym}
            for col in TARGET_COLUMNS:
                rec[col] = float(r[col][i])
            rows.append(rec)
    flat = pd.DataFrame(rows)

    for sym in flat['atom'].unique():
        sub = flat[flat['atom'] == sym]
        # N and LI are z-scored
        for col in ('N', 'LI'):
            assert abs(sub[col].mean()) < atol_mean, (
                f"{sym}.{col}: mean {sub[col].mean():.3f} not ~0"
            )
            assert abs(sub[col].std(ddof=0) - 1.0) < atol_std, (
                f"{sym}.{col}: std {sub[col].std(ddof=0):.3f} not ~1"
            )
        # Mu_* / Q_* components are *all* divided by the same per-element
        # scalar (Mu_rms = sqrt(mean(|Mu|^2)), Q_rms = sqrt(mean(||Q||_F^2))).
        # After that scaling the per-element mean of (sum_components^2) is 1
        # by construction.
        mu_sq = sub[['Mu_X', 'Mu_Y', 'Mu_Z']].pow(2).sum(axis=1)
        q_sq = sub[['Q_XY', 'Q_XZ', 'Q_YZ', 'Q_aniso', 'Q_ZZ']].pow(2).sum(axis=1)
        mu_rms = float(np.sqrt(mu_sq.mean()))
        q_rms = float(np.sqrt(q_sq.mean()))
        assert abs(mu_rms - 1.0) < 0.3, (
            f"{sym}.Mu_*: RMS norm {mu_rms:.3f} not ~1"
        )
        assert abs(q_rms - 1.0) < 0.3, (
            f"{sym}.Q_*: RMS norm {q_rms:.3f} not ~1"
        )


# ---------------------------------------------------------------------------
# 4. Tiny test runner so notebooks can collect PASS/FAIL counts inline
# ---------------------------------------------------------------------------

class TestRunner:
    """Bare-bones aggregator used by the notebooks.

    Each ``run`` call executes a no-arg callable that should raise on failure
    (e.g. via ``assert``) or return without raising on success.  The runner
    captures the exception and continues so a notebook cell can report the
    full set of results before failing.
    """

    def __init__(self) -> None:
        self.results: List[Tuple[str, bool, str]] = []

    def run(self, name: str, fn) -> None:
        try:
            fn()
        except AssertionError as exc:
            self.results.append((name, False, f"AssertionError: {exc}"))
        except Exception as exc:  # noqa: BLE001
            self.results.append((name, False, f"{type(exc).__name__}: {exc}"))
        else:
            self.results.append((name, True, ""))

    def report(self) -> bool:
        n_pass = sum(1 for _, ok, _ in self.results if ok)
        n_fail = len(self.results) - n_pass
        for name, ok, msg in self.results:
            tag = "PASS" if ok else "FAIL"
            line = f"  [{tag}] {name}"
            if msg:
                line += f"  -- {msg}"
            print(line)
        print(f"\nTotal: {n_pass} passed, {n_fail} failed")
        return n_fail == 0
