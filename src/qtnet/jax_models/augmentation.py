"""
SO(3) data augmentation for cell-complex batches.

Provides Haar-uniform random rotations that act on:
- Atom positions  (pos)
- Edge gyration tensors  (G)
- L=1 targets  (Mu_X, Mu_Y, Mu_Z  – columns 2-4 of y)
- L=2 targets  (Q_XY, Q_XZ, Q_YZ, Q_aniso, Q_ZZ – columns 5-9 of y)

Rotation-invariant quantities (distance, G_norm) are left unchanged.
Connectivity (senders, receivers, masks, etc.) is unchanged.
"""

import jax
import jax.numpy as jnp
from qtnet.jax_models.representations import (
    compute_gyration_tensor,
    ComplexBatch,
)


# ============================================================================
# Rotation Matrix Generation
# ============================================================================

def random_rotation_matrix(key: jnp.ndarray) -> jnp.ndarray:
    """Sample a Haar-uniform random 3×3 rotation matrix.

    Uses the Arvo (1992) subgroup algorithm:
    1. Draw three uniform random numbers u1, u2, u3 ∈ [0, 1).
    2. Build a rotation by angle 2π·u1 around z-axis.
    3. Build a Householder reflection that maps z to a random direction
       on S², then combine to get a uniform SO(3) element.

    Returns:
        R: jnp.ndarray of shape (3, 3).
    """
    u = jax.random.uniform(key, shape=(3,))
    u1, u2, u3 = u[0], u[1], u[2]

    # Rotation by angle theta = 2*pi*u1 about z
    theta = 2.0 * jnp.pi * u1
    ct, st = jnp.cos(theta), jnp.sin(theta)
    Rz = jnp.array([[ct, st, 0.0],
                     [-st, ct, 0.0],
                     [0.0, 0.0, 1.0]])

    # Random point on S2 for Householder reflection
    phi = 2.0 * jnp.pi * u2
    z = u3  # cos(inclination) uniform ~ U(0,1) for hemisphere
    r = jnp.sqrt(1.0 - z * z)
    v = jnp.array([jnp.cos(phi) * r, jnp.sin(phi) * r, z])

    # Householder matrix H = 2 * v v^T - I
    H = 2.0 * jnp.outer(v, v) - jnp.eye(3)

    # Combine: uniform SO(3) = H @ Rz  (det = +1)
    return H @ Rz


# ============================================================================
# Wigner-D Matrix for L=2 in the 5-component basis
# ============================================================================

def wigner_d_l2(R: jnp.ndarray) -> jnp.ndarray:
    """Compute the 5×5 Wigner-D matrix that transforms our L=2 basis
    under rotation R.

    Our basis ordering is:
        [xy, xz, yz, (xx-yy)/2, zz - r²/3]

    Instead of using Euler angles, we directly compute the transformation
    by applying R to each basis element (Cartesian tensors) and reading
    off the coefficients.

    Given a 3×3 rotation R, the traceless symmetric tensor T transforms as
    T' = R T R^T.  We express this action in the 5-component representation.

    Args:
        R: Rotation matrix of shape (3, 3).

    Returns:
        D: jnp.ndarray of shape (5, 5).
    """
    # Build the 5 basis tensors in 3x3 symmetric traceless form,
    # then rotate each and project back.
    # 
    # Basis (same convention as compute_gyration_tensor / compute_traceless_outer_product):
    # b0 = xy:        T0[0,1] = T0[1,0] = 1   (others 0)
    # b1 = xz:        T1[0,2] = T1[2,0] = 1
    # b2 = yz:        T2[1,2] = T2[2,1] = 1
    # b3 = (xx-yy)/2: T3[0,0] = 1,  T3[1,1] = -1  (divided by 2 below)
    # b4 = zz-r²/3:   T4[2,2] = 2/3,  T4[0,0] = T4[1,1] = -1/3

    def _rotate_and_project(T):
        """Rotate T → R T R^T, then extract 5 coefficients."""
        T_rot = R @ T @ R.T
        c0 = T_rot[0, 1]               # xy
        c1 = T_rot[0, 2]               # xz
        c2 = T_rot[1, 2]               # yz
        c3 = (T_rot[0, 0] - T_rot[1, 1]) / 2.0  # (xx-yy)/2
        c4 = T_rot[2, 2]               # zz (tracelessness means xx+yy+zz=0, so zz = zz - r²/3 is just zz for traceless inputs)
        return jnp.array([c0, c1, c2, c3, c4])

    # Build basis tensors.  Each Bᵢ is the 3×3 matrix you get when
    # component cᵢ = 1, all others = 0, and the traceless symmetric
    # matrix is reconstructed via:
    #   T[0,1] = c₀, T[0,2] = c₁, T[1,2] = c₂,
    #   T[2,2] = c₄,  T[0,0] = c₃ - c₄/2,  T[1,1] = -c₃ - c₄/2.
    T0 = jnp.array([[0., 1., 0.], [1., 0., 0.], [0., 0., 0.]])   # xy
    T1 = jnp.array([[0., 0., 1.], [0., 0., 0.], [1., 0., 0.]])   # xz
    T2 = jnp.array([[0., 0., 0.], [0., 0., 1.], [0., 1., 0.]])   # yz
    # c₃ = 1 → T[0,0]=1, T[1,1]=-1, T[2,2]=0
    T3 = jnp.array([[1., 0., 0.], [0., -1., 0.], [0., 0., 0.]])
    # c₄ = 1 → T[0,0]=-1/2, T[1,1]=-1/2, T[2,2]=1
    T4 = jnp.array([[-0.5, 0., 0.], [0., -0.5, 0.], [0., 0., 1.]])

    D = jnp.stack([
        _rotate_and_project(T0),
        _rotate_and_project(T1),
        _rotate_and_project(T2),
        _rotate_and_project(T3),
        _rotate_and_project(T4),
    ], axis=0)  # (5, 5): D[i, j] = coefficient of j-th basis in rotation of i-th basis

    # Transpose: we want D such that new_coeffs = D @ old_coeffs
    # _rotate_and_project gives row i = image of basis i → each row is the
    # decomposition of the rotated i-th basis tensor.
    # For a vector v in the basis: v' = D @ v where D[j, i] = proj_j(R(b_i)).
    # So we actually need D.T.
    return D.T


# ============================================================================
# Batch Augmentation
# ============================================================================

def augment_batch(batch: ComplexBatch, key: jnp.ndarray) -> ComplexBatch:
    """Apply a random SO(3) rotation to a ComplexBatch.

    Modifies:
    - ``node_static['pos']`` → R @ pos
    - ``edge_static['G']`` → recomputed from rotated displacements
    - ``node_y`` columns 2-4 (Mu vector) → R @ Mu
    - ``node_y`` columns 5-9 (Q tensor) → D_2 @ Q

    Unchanged (rotation-invariant):
    - ``edge_static['distance']``
    - ``edge_static['G_norm']`` (Frobenius norm of traceless tensor)
    - ``edge_static['atoms']`` (topology)
    - All adjacency indices and masks

    Args:
        batch: A ComplexBatch.
        key: A JAX PRNGKey.

    Returns:
        A new ComplexBatch with rotated geometry and targets.
    """
    R = random_rotation_matrix(key)
    D2 = wigner_d_l2(R)

    node_cb = batch.cochain_batches[0]
    edge_cb = batch.cochain_batches[1]

    # --- Rotate node positions ---
    pos = node_cb.static['pos']           # (n_nodes, 3)
    pos_rot = pos @ R.T                    # (n_nodes, 3)

    new_node_static = dict(node_cb.static)
    new_node_static['pos'] = pos_rot

    # --- Recompute edge G from rotated positions (G_norm is invariant) ---
    edge_atoms = edge_cb.static['atoms']   # (n_edges, 2)
    r_ij_rot = pos_rot[edge_atoms[:, 1]] - pos_rot[edge_atoms[:, 0]]  # (n_edges, 3)
    new_G, _ = compute_gyration_tensor(r_ij_rot)  # G_norm unchanged

    new_edge_static = dict(edge_cb.static)
    new_edge_static['G'] = new_G
    # G_norm, distance, atoms: all invariant — keep as-is

    # --- Rotate targets (y) ---
    y = node_cb.y  # (n_nodes, 10)

    # Scalars (N, LI) — columns 0,1 — invariant
    scalars = y[:, :2]

    # L=1 vector (Mu_X, Mu_Y, Mu_Z) — columns 2-4
    mu_rot = y[:, 2:5] @ R.T                  # (n_nodes, 3)

    # L=2 tensor (Q_XY, Q_XZ, Q_YZ, Q_aniso, Q_ZZ) — columns 5-9
    Q_rot = y[:, 5:10] @ D2.T                 # (n_nodes, 5)

    new_y = jnp.concatenate([scalars, mu_rot, Q_rot], axis=-1)

    # --- Reconstruct batch ---
    new_node_cb = node_cb.replace(static=new_node_static, y=new_y)
    new_edge_cb = edge_cb.replace(static=new_edge_static)

    # Rebuild cochain_batches list (preserve higher dimensions if any)
    new_cochain_batches = [new_node_cb, new_edge_cb] + list(batch.cochain_batches[2:])
    new_batch = batch.replace(cochain_batches=new_cochain_batches)

    return new_batch


def augment_batches(
    batches: list,
    key: jnp.ndarray,
) -> list:
    """Apply independent random SO(3) rotations to each batch in a list.

    Args:
        batches: List of ComplexBatch objects.
        key: A JAX PRNGKey (will be split for each batch).

    Returns:
        List of augmented ComplexBatch objects.
    """
    keys = jax.random.split(key, len(batches))
    return [augment_batch(b, k) for b, k in zip(batches, keys)]


# ============================================================================
# Fast numpy-only augmentation (for batches already living on host)
# ============================================================================

def augment_batches_fast(
    batches: list,
    key: jnp.ndarray,
) -> list:
    """Fast SO(3) augmentation for numpy-backed padded batches.

    Avoids reconstructing the full batch structure.  Only the three arrays
    that actually change under rotation are touched:

    - ``node_static['pos']``   ← pos @ R^T
    - ``edge_static['G']``     ← G  @ D2^T   (tensor rotation, no recompute)
    - ``node_cochain.y``       ← rotate Mu (L=1) and Q (L=2) columns

    All other fields (adjacency, masks, distances, G_norm, …) are
    rotation-invariant and left untouched.

    Args:
        batches: List of ComplexBatch with numpy arrays (``as_numpy=True``).
        key: A JAX PRNGKey (split internally for each batch).

    Returns:
        The *same* list, mutated in-place.
    """
    import numpy as np

    keys = jax.random.split(key, len(batches))
    for batch, k in zip(batches, keys):
        # Generate rotation on device, then pull to host (3×3 + 5×5 — tiny)
        R = np.asarray(random_rotation_matrix(k))          # (3, 3)
        D2 = np.asarray(wigner_d_l2(jnp.asarray(R)))      # (5, 5)

        node_cb = batch.cochain_batches[0]

        # ── positions (in-place dict mutation) ──
        node_cb.static['pos'] = node_cb.static['pos'] @ R.T

        # ── G: rotate via D2 instead of recomputing from positions ──
        # G is a rank-2 traceless symmetric tensor in 5-component basis;
        # the Wigner-D L=2 matrix rotates it exactly like Q.
        batch.cochain_batches[1].static['G'] = (
            batch.cochain_batches[1].static['G'] @ D2.T
        )

        # ── targets y (need .replace because y is an immutable attribute) ──
        y = node_cb.y                                      # (n_nodes, 10)
        new_y = np.empty_like(y)
        new_y[:, :2]   = y[:, :2]                          # N, LI — invariant
        new_y[:, 2:5]  = y[:, 2:5]  @ R.T                  # Mu (L=1)
        new_y[:, 5:10] = y[:, 5:10] @ D2.T                 # Q  (L=2)
        batch.cochain_batches[0] = node_cb.replace(y=new_y)

    return batches
