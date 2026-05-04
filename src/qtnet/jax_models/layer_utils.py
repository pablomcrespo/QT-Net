"""
Utility functions for equivariant layers.

This module contains pure functions (no learnable parameters) that are shared
across multiple layer classes. These include:
- Feature extraction (scalars, L=1, L=2)
- Norm computations
- Frobenius inner products and tensor alignments
- Radial basis functions
- Index building utilities for irreps

All functions are JIT-compatible and operate on JAX arrays.
"""

import jax.numpy as jnp
from typing import Optional, Tuple, List, Dict
import e3nn_jax as e3nn
from qtnet.jax_models.representations import compute_frobenius_norm


# =============================================================================
# IrrepsInfo — precomputed metadata for an irreps specification
# =============================================================================

class IrrepsInfo:
    """Precomputed metadata for an irreps specification.
    
    Stores channel counts, feature indices, and gate expansion indices
    for efficient feature extraction and gating operations.
    
    Attributes:
        num_l0: Number of L=0 (scalar) channels
        num_l1: Number of L=1 (vector) channels
        num_l2: Number of L=2 (tensor) channels
        l0_indices: Tuple of indices for L=0 features, or None
        l1_indices: Tuple of indices for L=1 features, or None
        l2_indices: Tuple of indices for L=2 features, or None
        gate_indices: Tuple mapping each feature dim to its instance index
        num_instances: Total irrep instances (num_l0 + num_l1 + num_l2)
        total_dim: Total feature dimension
    """
    __slots__ = ('num_l0', 'num_l1', 'num_l2', 'l0_indices', 'l1_indices',
                 'l2_indices', 'gate_indices', 'num_instances', 'total_dim')
    
    def __init__(self, num_l0, num_l1, num_l2, l0_indices, l1_indices,
                 l2_indices, gate_indices, num_instances, total_dim):
        self.num_l0 = num_l0
        self.num_l1 = num_l1
        self.num_l2 = num_l2
        self.l0_indices = l0_indices
        self.l1_indices = l1_indices
        self.l2_indices = l2_indices
        self.gate_indices = gate_indices
        self.num_instances = num_instances
        self.total_dim = total_dim
    
    def __repr__(self):
        return (f"IrrepsInfo(L0={self.num_l0}, L1={self.num_l1}, L2={self.num_l2}, "
                f"dim={self.total_dim}, instances={self.num_instances})")


def build_irreps_info(irreps) -> IrrepsInfo:
    """Build precomputed irreps metadata from an e3nn Irreps object.
    
    Consolidates channel counting, index building, and gate index construction
    into a single call.
    
    Args:
        irreps: e3nn.Irreps object (parse strings first via e3nn.Irreps(str))
        
    Returns:
        IrrepsInfo with precomputed channel counts, indices, and gate mapping
    """
    num_l0 = sum(mul for mul, ir in irreps if ir.l == 0)
    num_l1 = sum(mul for mul, ir in irreps if ir.l == 1)
    num_l2 = sum(mul for mul, ir in irreps if ir.l == 2)
    
    l0_indices = build_l_indices(irreps, 0)
    l1_indices = build_l_indices(irreps, 1)
    l2_indices = build_l_indices(irreps, 2)
    
    gate_indices = []
    gate_counter = 0
    total_dim = 0
    for mul, ir in irreps:
        for _m in range(mul):
            gate_indices.extend([gate_counter] * ir.dim)
            gate_counter += 1
            total_dim += ir.dim
    
    return IrrepsInfo(
        num_l0=num_l0, num_l1=num_l1, num_l2=num_l2,
        l0_indices=l0_indices, l1_indices=l1_indices, l2_indices=l2_indices,
        gate_indices=tuple(gate_indices),
        num_instances=gate_counter,
        total_dim=total_dim,
    )


def extract_invariants(
    x: jnp.ndarray,
    info: IrrepsInfo,
) -> jnp.ndarray:
    """Extract rotation-invariant features: [L=0 scalars, ||L=1||, ||L=2||].
    
    Concatenates L=0 scalar values, L=1 vector norms, and L=2 tensor norms
    into a single invariant representation.
    
    Args:
        x: Feature array, shape (batch, total_dim)
        info: IrrepsInfo for the feature space
        
    Returns:
        Invariant features, shape (batch, num_instances)
    """
    parts = []
    if info.num_l0 > 0 and info.l0_indices is not None:
        parts.append(x[:, info.l0_indices])
    if info.num_l1 > 0 and info.l1_indices is not None:
        parts.append(compute_l1_norms(x[:, info.l1_indices], info.num_l1))
    if info.num_l2 > 0 and info.l2_indices is not None:
        parts.append(compute_l2_norms(x[:, info.l2_indices], info.num_l2))
    return jnp.concatenate(parts, axis=-1)


def expand_gates(
    gates: jnp.ndarray,
    gate_indices: Tuple[int, ...],
) -> jnp.ndarray:
    """Expand per-instance gates to full feature dimension.
    
    Each irrep instance gate value is repeated for all its feature components:
    1 for L=0, 3 for L=1, 5 for L=2.
    
    Args:
        gates: Per-instance gates, shape (batch, num_instances)
        gate_indices: Tuple mapping each feature dim to its instance index
        
    Returns:
        Expanded gates, shape (batch, total_dim)
    """
    return gates[:, gate_indices]


# =============================================================================
# Constants
# =============================================================================

# Frobenius weights for 5-component L=2 tensor: [xy, xz, yz, (xx-yy)/2, zz]
# 
# For a traceless symmetric 3x3 tensor with 5 components:
#   ||T||_F^2 = xx^2 + yy^2 + zz^2 + 2*xy^2 + 2*xz^2 + 2*yz^2
#
# With our representation: aniso = (xx-yy)/2, and tracelessness xx+yy+zz=0:
#   xx = aniso - zz/2
#   yy = -aniso - zz/2
#
# Substituting:
#   ||T||_F^2 = (aniso - zz/2)^2 + (-aniso - zz/2)^2 + zz^2 + 2*xy^2 + 2*xz^2 + 2*yz^2
#             = 2*aniso^2 + zz^2/2 + zz^2 + 2*xy^2 + 2*xz^2 + 2*yz^2  
#             = 2*xy^2 + 2*xz^2 + 2*yz^2 + 2*aniso^2 + 1.5*zz^2
#
# Therefore the correct weights are [2, 2, 2, 2, 1.5]
FROBENIUS_WEIGHTS_L2 = jnp.array([2.0, 2.0, 2.0, 2.0, 1.5])

# Small epsilon for numerical stability in divisions/sqrt
EPS = 1e-8


# =============================================================================
# Feature Extraction
# =============================================================================

def extract_by_indices(
    x: jnp.ndarray,
    indices: Optional[jnp.ndarray],
) -> jnp.ndarray:
    """
    Extract features at specified indices.
    
    Args:
        x: Feature array of shape (batch, feature_dim)
        indices: Indices to extract, or None
        
    Returns:
        Extracted features of shape (batch, len(indices)), or zeros if indices is None
    """
    if indices is None:
        return jnp.zeros((x.shape[0], 0))
    return x[:, indices]



# =============================================================================
# Norm Computations
# =============================================================================

def compute_l1_norms(
    l1_features: jnp.ndarray,
    num_channels: int,
) -> jnp.ndarray:
    """
    Compute norms of each L=1 (vector) channel.
    
    Args:
        l1_features: L=1 features of shape (batch, num_channels * 3)
        num_channels: Number of L=1 channels
        
    Returns:
        Norms of shape (batch, num_channels)
    """
    if num_channels == 0:
        return jnp.zeros((l1_features.shape[0], 0))
    l1_reshaped = l1_features.reshape(-1, num_channels, 3)
    norms = jnp.sqrt(jnp.sum(l1_reshaped ** 2, axis=-1) + EPS)
    return norms


def compute_l2_norms(
    l2_features: jnp.ndarray,
    num_channels: int,
) -> jnp.ndarray:
    """
    Compute Frobenius norms of each L=2 (tensor) channel.
    
    Uses proper Frobenius weights for rotation-invariant norm.
    
    Args:
        l2_features: L=2 features of shape (batch, num_channels * 5)
        num_channels: Number of L=2 channels
        
    Returns:
        Norms of shape (batch, num_channels)
    """
    if num_channels == 0:
        return jnp.zeros((l2_features.shape[0], 0))
    l2_reshaped = l2_features.reshape(-1, num_channels, 5)
    # Use proper Frobenius weights for rotation-invariant norm
    norms = jnp.sqrt(jnp.sum(FROBENIUS_WEIGHTS_L2 * l2_reshaped ** 2, axis=-1) + EPS)
    return norms


def e3norm_l2(
    tensor: jnp.ndarray,
    scale: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """
    E3-equivariant normalization for L=2 tensor in 5-component representation.
    
    Normalizes by the proper Frobenius norm, optionally applying a learnable
    scale factor:  G_normed = G / ||G||_F [* scale]
    
    Args:
        tensor: Tensor of shape (..., 5) in [xy, xz, yz, (xx-yy)/2, zz] format
        scale: Optional learnable scale factor, broadcastable
        
    Returns:
        Normalized tensor of same shape as input
    """
    norm = compute_frobenius_norm(tensor)
    normed = tensor / (norm[..., None] + EPS)
    if scale is not None:
        normed = normed * scale
    return normed


def compute_traceless_outer_product(
    r_hat: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute the traceless symmetric outer product of a unit vector.
    
    For a unit vector r̂ = (x, y, z), computes the 5-component L=2 representation
    of the traceless symmetric tensor S = r̂⊗r̂ - (1/3)I in the basis
    [xy, xz, yz, (xx-yy)/2, zz - 1/3].
    
    This uses the same convention as ``compute_gyration_tensor`` so that
    ``FROBENIUS_WEIGHTS_L2 = [2, 2, 2, 2, 1.5]`` gives the correct
    Frobenius norm.
    
    Args:
        r_hat: Unit vectors of shape (..., 3)
        
    Returns:
        Traceless outer product of shape (..., 5)
    """
    x = r_hat[..., 0]
    y = r_hat[..., 1]
    z = r_hat[..., 2]
    
    r_sq = x**2 + y**2 + z**2
    xy = x * y
    xz = x * z
    yz = y * z
    aniso = (x**2 - y**2) / 2.0
    zz = z**2 - r_sq / 3.0
    scale = jnp.sqrt(3/2) #unitary vector maps to unitary Frobenius norm
    return scale*jnp.stack([xy, xz, yz, aniso, zz], axis=-1)


# =============================================================================
# Scalar/Inner Products
# =============================================================================

def compute_l1_cosine_similarity(
    u: jnp.ndarray,
    v: jnp.ndarray,
    num_channels: int,
) -> jnp.ndarray:
    """
    Compute channel-wise cosine similarity between L=1 features.
    
    Uses unit vectors to capture angular information only.
    
    Args:
        u: L=1 features of shape (batch, num_channels * 3)
        v: L=1 features of shape (batch, num_channels * 3)
        num_channels: Number of L=1 channels
        
    Returns:
        Cosine similarities of shape (batch, num_channels)
    """
    if num_channels == 0:
        return jnp.zeros((u.shape[0], 0))
    
    u_reshaped = u.reshape(-1, num_channels, 3)
    v_reshaped = v.reshape(-1, num_channels, 3)
    
    # Clamp norms to prevent numerical instability for near-zero vectors
    u_norms = jnp.maximum(
        jnp.sqrt(jnp.sum(u_reshaped ** 2, axis=-1, keepdims=True) + EPS), 1e-3)
    v_norms = jnp.maximum(
        jnp.sqrt(jnp.sum(v_reshaped ** 2, axis=-1, keepdims=True) + EPS), 1e-3)
    
    u_unit = u_reshaped / u_norms
    v_unit = v_reshaped / v_norms
    
    cos_sim = jnp.sum(u_unit * v_unit, axis=-1)
    return cos_sim


def compute_l2_frobenius_product(
    u: jnp.ndarray,
    v: jnp.ndarray,
    num_channels: int,
    normalize: bool = False,
) -> jnp.ndarray:
    """
    Compute channel-wise Frobenius inner product between L=2 tensors.
    
    For symmetric traceless tensors in 5-component representation:
    Tr[U^T V] = sum_ij U_ij V_ij with proper weights for off-diagonal terms.
    
    Args:
        u: L=2 features of shape (batch, num_channels * 5)
        v: L=2 features of shape (batch, num_channels * 5)
        num_channels: Number of L=2 channels
        normalize: If True, normalize to cosine-like alignment in [-1, 1]
        
    Returns:
        Inner products of shape (batch, num_channels)
    """
    if num_channels == 0:
        return jnp.zeros((u.shape[0], 0))
    
    u_reshaped = u.reshape(-1, num_channels, 5)
    v_reshaped = v.reshape(-1, num_channels, 5)
    
    if normalize:
        # Use proper Frobenius weights for normalization
        # Clamp norms to prevent numerical instability for near-zero tensors
        u_norms = jnp.maximum(
            jnp.sqrt(jnp.sum(FROBENIUS_WEIGHTS_L2 * u_reshaped ** 2, axis=-1, keepdims=True) + EPS), 1e-3)
        v_norms = jnp.maximum(
            jnp.sqrt(jnp.sum(FROBENIUS_WEIGHTS_L2 * v_reshaped ** 2, axis=-1, keepdims=True) + EPS), 1e-3)
        u_reshaped = u_reshaped / u_norms
        v_reshaped = v_reshaped / v_norms
    
    # Weighted Frobenius inner product
    product = jnp.sum(FROBENIUS_WEIGHTS_L2 * u_reshaped * v_reshaped, axis=-1)
    return product


def compute_tensor_alignment(
    u: jnp.ndarray,
    v: jnp.ndarray,
    num_channels: int,
) -> jnp.ndarray:
    """
    Compute per-channel alignment between normalized L=2 tensors.
    
    This is the Frobenius inner product of unit tensors, measuring
    angular alignment between tensor orientations (like cosine similarity
    for vectors, but for rank-2 tensors).
    
    Args:
        u: L=2 features of shape (batch, num_channels * 5)
        v: L=2 features of shape (batch, num_channels * 5)
        num_channels: Number of L=2 channels
        
    Returns:
        Alignment scores in [-1, 1] of shape (batch, num_channels)
    """
    return compute_l2_frobenius_product(u, v, num_channels, normalize=True)


def compute_gyration_alignment(
    edge_tensor: jnp.ndarray,
    ring_tensor: jnp.ndarray,
    ring_norm: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute alignment between edge gyration tensor and ring gyration tensor.
    
    Computes: Tr[r G] / ||G||_F
    
    This measures how aligned the edge orientation is with the overall ring geometry.
    
    Args:
        edge_tensor: Edge gyration tensor, shape (batch, 5)
        ring_tensor: Ring gyration tensor, shape (batch, 5)
        ring_norm: Ring gyration norm, shape (batch,)
        
    Returns:
        Alignment score of shape (batch, 1)
    """
    trace = jnp.sum(FROBENIUS_WEIGHTS_L2 * edge_tensor * ring_tensor, axis=-1)
    alignment = trace / (ring_norm + EPS)
    return alignment[:, None]


# =============================================================================
# Radial Basis Functions
# =============================================================================

def compute_radial_basis_gaussian(
    distances: jnp.ndarray,
    num_basis: int,
    cutoff: float,
) -> jnp.ndarray:
    """
    Compute Gaussian radial basis functions for distance encoding.
    
    Args:
        distances: Distance values, shape (batch,) or (batch, 1)
        num_basis: Number of radial basis functions
        cutoff: Cutoff distance
        
    Returns:
        RBF values of shape (batch, num_basis)
    """
    distances = distances.reshape(-1, 1)
    centers = jnp.linspace(0, cutoff, num_basis)
    sigma = cutoff / num_basis
    rbf = jnp.exp(-0.5 * ((distances - centers) / sigma) ** 2)
    return rbf


def compute_radial_basis_bessel(
    distances: jnp.ndarray,
    num_basis: int,
    cutoff: float,
) -> jnp.ndarray:
    """
    Compute Bessel radial basis functions (as in DimeNet).
    
    Args:
        distances: Distance values, shape (batch,) or (batch, 1)
        num_basis: Number of radial basis functions
        cutoff: Cutoff distance
        
    Returns:
        RBF values of shape (batch, num_basis)
    """
    distances = distances.reshape(-1, 1)
    n = jnp.arange(1, num_basis + 1)
    d_scaled = distances / cutoff
    rbf = jnp.sqrt(2.0 / cutoff) * jnp.sin(n * jnp.pi * d_scaled) / (distances + EPS)
    return rbf


def compute_smooth_cutoff(
    distances: jnp.ndarray,
    cutoff: float,
) -> jnp.ndarray:
    """
    Apply smooth cosine cutoff envelope to distances.
    
    Returns 1 at d=0, smoothly transitions to 0 at d=cutoff.
    
    Args:
        distances: Distance values, shape (batch,)
        cutoff: Cutoff distance
        
    Returns:
        Cutoff envelope values in [0, 1], shape (batch,)
    """
    x = distances / cutoff
    envelope = 0.5 * (1.0 + jnp.cos(jnp.pi * x))
    return jnp.where(distances < cutoff, envelope, 0.0)


# =============================================================================
# Angular Basis Functions (Legendre Polynomials)
# =============================================================================

def compute_legendre_basis(
    x: jnp.ndarray,
    num_basis: int,
) -> jnp.ndarray:
    """Legendre polynomial expansion P_0(x), ..., P_{K-1}(x).

    Uses Bonnet's recurrence:
        P_0 = 1
        P_1 = x
        P_{n+1} = ((2n+1)*x*P_n - n*P_{n-1}) / (n+1)

    Input *x* should be in [-1, 1] (e.g. alignment scores between
    normalised gyration tensors).

    Args:
        x: Values in [-1, 1], shape ``(batch,)`` or ``(batch, C)``.
        num_basis: Number of Legendre polynomials K (P_0 .. P_{K-1}).

    Returns:
        Shape ``(*x.shape, num_basis)``.
    """
    if num_basis == 0:
        return jnp.zeros((*x.shape, 0))

    basis = [jnp.ones_like(x)]
    if num_basis > 1:
        basis.append(x)

    for n in range(1, num_basis - 1):
        p_next = ((2 * n + 1) * x * basis[-1] - n * basis[-2]) / (n + 1)
        basis.append(p_next)

    return jnp.stack(basis, axis=-1)


# =============================================================================
# Index Building Utilities
# =============================================================================

def build_l_indices(
    irreps: e3nn.Irreps,
    target_l: int,
) -> Optional[Tuple[int, ...]]:
    """
    Build indices for extracting features of a specific L value.
    
    Args:
        irreps: e3nn Irreps object
        target_l: Target angular momentum value (0, 1, 2, ...)
        
    Returns:
        Tuple of indices, or None if no features of that L exist.
        Returns tuple (not array) to avoid issues with nnx.state().
    """
    indices = []
    idx = 0
    for mul, ir in irreps:
        for m in range(mul):
            if ir.l == target_l:
                indices.extend(range(idx, idx + ir.dim))
            idx += ir.dim
    
    if not indices:
        return None
    return tuple(indices)


def build_scalar_indices(irreps: e3nn.Irreps) -> Optional[Tuple[int, ...]]:
    """Build indices for L=0 (scalar) features."""
    return build_l_indices(irreps, 0)


def build_l1_indices(irreps: e3nn.Irreps) -> Optional[Tuple[int, ...]]:
    """Build indices for L=1 (vector) features."""
    return build_l_indices(irreps, 1)


def build_l2_indices(irreps: e3nn.Irreps) -> Optional[Tuple[int, ...]]:
    """Build indices for L=2 (tensor) features."""
    return build_l_indices(irreps, 2)



def build_instance_slices(
    irreps: e3nn.Irreps,
    target_l: int,
) -> List[Tuple[int, int]]:
    """
    Build (start, end) slices for each instance of a specific L value.
    
    Useful for iterating over individual L instances in a JIT-compatible way.
    
    Args:
        irreps: e3nn Irreps object
        target_l: Target angular momentum value
        
    Returns:
        List of (start_idx, end_idx) tuples for each instance
    """
    slices = []
    idx = 0
    for mul, ir in irreps:
        for m in range(mul):
            if ir.l == target_l:
                slices.append((idx, idx + ir.dim))
            idx += ir.dim
    return slices


# =============================================================================
# Channel-Wise Linear Operations
# =============================================================================
# 
# IMPORTANT: For equivariant networks, linear layers must be applied CHANNEL-WISE
# (across different instances of the same L value), NOT component-wise.
#
# For L=1 features: shape (batch, num_channels * 3) represents num_channels vectors
# For L=2 features: shape (batch, num_channels * 5) represents num_channels tensors
#
# A channel-wise linear maps: num_input_channels -> num_output_channels
# Applied identically to each spatial component.

def apply_channel_wise_linear_l1(
    linear_fn,
    l1_features: jnp.ndarray,
    num_input_channels: int,
) -> jnp.ndarray:
    """
    Apply a linear layer channel-wise to L=1 (vector) features.
    
    The linear layer maps num_input_channels -> num_output_channels and is
    applied identically to each of the 3 spatial components.
    
    Args:
        linear_fn: Callable linear layer (num_input_channels -> num_output_channels)
        l1_features: L=1 features of shape (batch, num_input_channels * 3)
        num_input_channels: Number of input L=1 channels
        
    Returns:
        Transformed features of shape (batch, num_output_channels * 3)
    """
    batch_size = l1_features.shape[0]
    # Reshape: (batch, num_channels * 3) -> (batch, num_channels, 3) -> (batch, 3, num_channels)
    reshaped = l1_features.reshape(batch_size, num_input_channels, 3).transpose(0, 2, 1)
    # Apply linear to last dimension: (batch, 3, num_input) -> (batch, 3, num_output)
    transformed = linear_fn(reshaped)
    num_output_channels = transformed.shape[-1]
    # Transpose back and flatten: (batch, 3, num_output) -> (batch, num_output * 3)
    return transformed.transpose(0, 2, 1).reshape(batch_size, num_output_channels * 3)


def apply_channel_wise_linear_l2(
    linear_fn,
    l2_features: jnp.ndarray,
    num_input_channels: int,
) -> jnp.ndarray:
    """
    Apply a linear layer channel-wise to L=2 (tensor) features.
    
    The linear layer maps num_input_channels -> num_output_channels and is
    applied identically to each of the 5 tensor components.
    
    Args:
        linear_fn: Callable linear layer (num_input_channels -> num_output_channels)
        l2_features: L=2 features of shape (batch, num_input_channels * 5)
        num_input_channels: Number of input L=2 channels
        
    Returns:
        Transformed features of shape (batch, num_output_channels * 5)
    """
    batch_size = l2_features.shape[0]
    # Reshape: (batch, num_channels * 5) -> (batch, num_channels, 5) -> (batch, 5, num_channels)
    reshaped = l2_features.reshape(batch_size, num_input_channels, 5).transpose(0, 2, 1)
    # Apply linear to last dimension: (batch, 5, num_input) -> (batch, 5, num_output)
    transformed = linear_fn(reshaped)
    num_output_channels = transformed.shape[-1]
    # Transpose back and flatten: (batch, 5, num_output) -> (batch, num_output * 5)
    return transformed.transpose(0, 2, 1).reshape(batch_size, num_output_channels * 5)


# =============================================================================
# Cartesian Tensor Product Operations
# =============================================================================

def l2_to_matrix(T5: jnp.ndarray) -> jnp.ndarray:
    """Reconstruct 3×3 symmetric traceless matrix from 5-component representation.

    Convention: T5 = [T_xy, T_xz, T_yz, (T_xx-T_yy)/2, T_zz]
    with tracelessness T_xx + T_yy + T_zz = 0:
        T_xx = T5[..., 3] - T5[..., 4] / 2
        T_yy = -T5[..., 3] - T5[..., 4] / 2
        T_zz = T5[..., 4]

    Args:
        T5: Tensor components, shape (..., 5)

    Returns:
        3×3 symmetric matrix, shape (..., 3, 3)
    """
    t0 = T5[..., 0]  # T_xy
    t1 = T5[..., 1]  # T_xz
    t2 = T5[..., 2]  # T_yz
    t3 = T5[..., 3]  # (T_xx - T_yy) / 2
    t4 = T5[..., 4]  # T_zz

    xx = t3 - t4 / 2
    yy = -t3 - t4 / 2
    zz = t4

    row0 = jnp.stack([xx, t0, t1], axis=-1)
    row1 = jnp.stack([t0, yy, t2], axis=-1)
    row2 = jnp.stack([t1, t2, zz], axis=-1)
    return jnp.stack([row0, row1, row2], axis=-2)


def contract_l2_l1(T5: jnp.ndarray, v3: jnp.ndarray) -> jnp.ndarray:
    """Matrix-vector contraction: L=2 × L=1 → L=1.

    Equivariant under SO(3): R·(T·v) = (R·T·Rᵀ)·(R·v).

    Args:
        T5: Tensor components, shape (..., 5)
        v3: Vector components, shape (..., 3)

    Returns:
        Contracted vector, shape (..., 3)
    """
    T_mat = l2_to_matrix(T5)  # (..., 3, 3)
    return jnp.einsum('...ij,...j->...i', T_mat, v3)


def outer_l1_l1_to_l2(u3: jnp.ndarray, v3: jnp.ndarray) -> jnp.ndarray:
    """Traceless symmetric outer product: L=1 × L=1 → L=2.

    Computes the traceless part of (u⊗v + v⊗u)/2 in the 5-component
    [T_xy, T_xz, T_yz, (T_xx-T_yy)/2, T_zz] convention.

    Args:
        u3: First vector, shape (..., 3)
        v3: Second vector, shape (..., 3)

    Returns:
        Traceless tensor, shape (..., 5)
    """
    dot = jnp.sum(u3 * v3, axis=-1)  # (...,)
    t0 = (u3[..., 0] * v3[..., 1] + u3[..., 1] * v3[..., 0]) / 2  # T_xy
    t1 = (u3[..., 0] * v3[..., 2] + u3[..., 2] * v3[..., 0]) / 2  # T_xz
    t2 = (u3[..., 1] * v3[..., 2] + u3[..., 2] * v3[..., 1]) / 2  # T_yz
    t3 = (u3[..., 0] * v3[..., 0] - u3[..., 1] * v3[..., 1]) / 2  # (T_xx-T_yy)/2
    t4 = u3[..., 2] * v3[..., 2] - dot / 3                          # T_zz (traceless)
    return jnp.stack([t0, t1, t2, t3, t4], axis=-1)


def compute_l1_dot_products(
    u: jnp.ndarray,
    v: jnp.ndarray,
    num_channels: int,
) -> jnp.ndarray:
    """
    Compute channel-wise dot products between L=1 (vector) features.
    
    This is rotation-invariant: <Ru, Rv> = <u, v>
    
    Args:
        u: L=1 features of shape (batch, num_channels * 3)
        v: L=1 features of shape (batch, num_channels * 3)
        num_channels: Number of L=1 channels
        
    Returns:
        Dot products of shape (batch, num_channels)
    """
    if num_channels == 0:
        return jnp.zeros((u.shape[0], 0))
    u_reshaped = u.reshape(-1, num_channels, 3)
    v_reshaped = v.reshape(-1, num_channels, 3)
    return jnp.sum(u_reshaped * v_reshaped, axis=-1)
