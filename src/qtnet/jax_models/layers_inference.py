"""
V2 layers for the inference experiment (SGNN_v2, EGNN_v2).

Changes vs. original layers:
- Distance encoders: BesselCutoffEncoder (cutoff models) and GaussianRBF (FC).
- Scalar gates are normalised: [r̂_ij, rbf(d)] instead of raw r_ij;
  [Ĝ, rbf(d)] instead of raw G; [ΔĜ, Legendre(alignment)] instead of raw ΔG.
- EdgeGeometryReminder: equivariant geometry injection at EdgeDown (L=1 + L=2).
"""

from typing import Optional, Dict

import jax
import jax.numpy as jnp
import flax.nnx as nnx

from qtnet.jax_models.layer_utils import (
    EPS,
    compute_radial_basis_bessel,
    compute_smooth_cutoff,
    compute_legendre_basis,
    compute_tensor_alignment,
    compute_frobenius_norm,
    compute_traceless_outer_product,
    FROBENIUS_WEIGHTS_L2,
    extract_by_indices,
)
from qtnet.jax_models.equivariant_layers import (
    EquivariantEdgeDownMessages,
)


# =============================================================================
# Distance Encoders
# =============================================================================

class BesselCutoffEncoder(nnx.Module):
    """Bessel RBF × smooth cosine cutoff envelope.

    No learnable parameters. Output shape: ``(n, num_basis)``.
    """

    def __init__(self, num_basis: int = 16, cutoff: float = 8.0):
        super().__init__()
        self.num_basis = num_basis
        self.cutoff = cutoff

    def __call__(self, distances: jnp.ndarray) -> jnp.ndarray:
        rbf = compute_radial_basis_bessel(distances, self.num_basis, self.cutoff)
        env = compute_smooth_cutoff(distances, self.cutoff)
        return rbf * env[:, None]


class GaussianRBF(nnx.Module):
    """Learnable Gaussian radial basis functions (no cutoff envelope).

    Parameters mu_k and sigma_k are initialised uniformly in
    ``[d_min, d_max]`` and optionally made learnable.
    Output shape: ``(n, num_basis)``, values in (0, 1].
    """

    def __init__(
        self,
        num_basis: int = 16,
        d_min: float = 0.0,
        d_max: float = 25.9,
        learnable: bool = True,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        self.num_basis = num_basis
        centers = jnp.linspace(d_min, d_max, num_basis)
        spacing = (d_max - d_min) / max(num_basis - 1, 1)
        widths = jnp.full(num_basis, spacing)

        if learnable:
            self.centers = nnx.Param(centers)
            self.widths = nnx.Param(widths)
        else:
            self.centers = centers
            self.widths = widths

    def __call__(self, distances: jnp.ndarray) -> jnp.ndarray:
        d = distances.reshape(-1, 1)
        mu = self.centers if not isinstance(self.centers, nnx.Param) else self.centers.value
        sigma = self.widths if not isinstance(self.widths, nnx.Param) else self.widths.value
        # Ensure widths stay positive
        sigma = jnp.abs(sigma) + EPS
        return jnp.exp(-0.5 * ((d - mu) / sigma) ** 2)


# =============================================================================
# V2 Scalar Layers — Normalised geometric gates
# =============================================================================

class ScalarNodeEncoder_v2(nnx.Module):
    """ScalarNodeEncoder with normalised gate: [r̂_ij, rbf(d)] → tanh gate."""

    def __init__(
        self,
        num_scalar_out: int,
        embedding_dim: int,
        distance_encoder: nnx.Module,
        hidden_dim: int = 64,
        geometric_filter_dim: int = 64,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        if rngs is None:
            rngs = nnx.Rngs(0)

        self.num_scalar_out = num_scalar_out
        self.distance_encoder = distance_encoder
        gate_input_dim = 3 + distance_encoder.num_basis

        self.mlp = nnx.Sequential(
            nnx.Linear(2 * embedding_dim, hidden_dim, rngs=rngs),
            nnx.silu,
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
            nnx.silu,
            nnx.Linear(hidden_dim, num_scalar_out, rngs=rngs),
        )
        self.r_gate = nnx.Sequential(
            nnx.Linear(gate_input_dim, geometric_filter_dim, rngs=rngs),
            nnx.silu,
            nnx.Linear(geometric_filter_dim, num_scalar_out, rngs=rngs),
            jnp.tanh,
        )

    def __call__(
        self,
        chem_embeddings: jnp.ndarray,
        x_mask: jnp.ndarray,
        up_senders: jnp.ndarray,
        up_receivers: jnp.ndarray,
        up_mask: Optional[jnp.ndarray] = None,
        static: Optional[Dict] = None,
    ) -> jnp.ndarray:
        num_nodes = chem_embeddings.shape[0]
        pos = static['pos']

        r_ij = pos[up_receivers] - pos[up_senders]
        d = jnp.sqrt(jnp.sum(r_ij ** 2, axis=-1) + EPS)
        r_hat = r_ij / (d[:, None] + EPS)
        rbf = self.distance_encoder(d)

        emb_s = chem_embeddings[up_senders]
        emb_r = chem_embeddings[up_receivers]
        messages = self.mlp(jnp.concatenate([emb_r, emb_s], axis=-1))

        gate = self.r_gate(jnp.concatenate([r_hat, rbf], axis=-1))
        messages = messages * gate

        if up_mask is not None:
            messages = jnp.where(up_mask[:, None], messages, 0.0)

        node_features = jax.ops.segment_sum(messages, up_receivers, num_nodes)
        if x_mask is not None:
            node_features = jnp.where(x_mask[:, None], node_features, 0.0)
        return node_features


class ScalarNodeUpMessages_v2(nnx.Module):
    """ScalarNodeUpMessages with normalised gate: [r̂_ij, rbf(d)] → tanh gate."""

    def __init__(
        self,
        num_scalars: int,
        distance_encoder: nnx.Module,
        hidden_dim: int = 64,
        num_edge_scalars: int = 0,
        geometric_filter_dim: int = 64,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        if rngs is None:
            rngs = nnx.Rngs(0)

        self.num_scalars = num_scalars
        self.num_edge_scalars = num_edge_scalars
        self.distance_encoder = distance_encoder
        gate_input_dim = 3 + distance_encoder.num_basis

        mlp_input_dim = 2 * num_scalars + num_edge_scalars
        self.message_mlp = nnx.Sequential(
            nnx.Linear(mlp_input_dim, hidden_dim, rngs=rngs),
            nnx.silu,
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
            nnx.silu,
            nnx.Linear(hidden_dim, num_scalars, rngs=rngs),
        )
        self.r_gate = nnx.Sequential(
            nnx.Linear(gate_input_dim, geometric_filter_dim, rngs=rngs),
            nnx.silu,
            nnx.Linear(geometric_filter_dim, num_scalars, rngs=rngs),
            jnp.tanh,
        )

    def __call__(
        self,
        x: jnp.ndarray,
        x_mask: jnp.ndarray,
        up_senders: jnp.ndarray,
        up_receivers: jnp.ndarray,
        up_mask: Optional[jnp.ndarray] = None,
        static: Optional[Dict] = None,
        edge_features: Optional[jnp.ndarray] = None,
        up_intermediaries: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        num_nodes = x.shape[0]
        pos = static['pos']

        h_s = x[up_senders]
        h_r = x[up_receivers]

        r_ij = pos[up_receivers]-pos[up_senders]
        d = jnp.sqrt(jnp.sum(r_ij ** 2, axis=-1) + EPS)
        r_hat = r_ij / (d[:, None] + EPS)
        rbf = self.distance_encoder(d)

        if self.num_edge_scalars > 0 and edge_features is not None and up_intermediaries is not None:
            e_ij = edge_features[up_intermediaries]
            mlp_input = jnp.concatenate([h_r, h_s, e_ij], axis=-1)
        else:
            mlp_input = jnp.concatenate([h_r, h_s], axis=-1)
        messages = self.message_mlp(mlp_input)

        gate = self.r_gate(jnp.concatenate([r_hat, rbf], axis=-1))
        messages = messages * gate

        if up_mask is not None:
            messages = jnp.where(up_mask[:, None], messages, 0.0)

        update = jax.ops.segment_sum(messages, up_receivers, num_nodes)
        if x_mask is not None:
            update = jnp.where(x_mask[:, None], update, 0.0)
        return update


class ScalarEdgeEncoder_v2(nnx.Module):
    """ScalarEdgeEncoder with normalised gate: [Ĝ, rbf(d)] → tanh gate."""

    def __init__(
        self,
        num_node_scalars: int,
        num_scalar_out: int,
        distance_encoder: nnx.Module,
        hidden_dim: int = 64,
        geometric_filter_dim: int = 64,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        if rngs is None:
            rngs = nnx.Rngs(0)

        self.distance_encoder = distance_encoder
        gate_input_dim = 5 + distance_encoder.num_basis

        self.inner_mlp = nnx.Sequential(
            nnx.Linear(num_node_scalars, hidden_dim, rngs=rngs),
            nnx.silu,
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
        )
        self.agg_norm = nnx.LayerNorm(num_features=hidden_dim, rngs=rngs)
        self.outer_mlp = nnx.Sequential(
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
            nnx.silu,
            nnx.Linear(hidden_dim, num_scalar_out, rngs=rngs),
        )
        self.geo_gate = nnx.Sequential(
            nnx.Linear(gate_input_dim, geometric_filter_dim, rngs=rngs),
            nnx.silu,
            nnx.Linear(geometric_filter_dim, num_scalar_out, rngs=rngs),
            jnp.tanh,
        )

    def __call__(
        self,
        boundary_x: jnp.ndarray,
        boundary_senders: jnp.ndarray,
        boundary_receivers: jnp.ndarray,
        boundary_mask: Optional[jnp.ndarray] = None,
        x_mask: Optional[jnp.ndarray] = None,
        static: Optional[Dict] = None,
    ) -> jnp.ndarray:
        G = static['G']
        G_norm = static['G_norm']
        d = static['distance']
        num_edges = G.shape[0]

        G_hat = G / (G_norm[:, None] + EPS)
        rbf = self.distance_encoder(d)

        node_features = boundary_x[boundary_senders]
        node_contrib = self.inner_mlp(node_features)
        if boundary_mask is not None:
            node_contrib = jnp.where(boundary_mask[:, None], node_contrib, 0.0)

        edge_hidden = jax.ops.segment_sum(node_contrib, boundary_receivers, num_edges)
        edge_hidden = self.agg_norm(edge_hidden)
        edge_features = self.outer_mlp(edge_hidden)

        geo_gate = self.geo_gate(jnp.concatenate([G_hat, rbf], axis=-1))
        edge_features = edge_features * geo_gate

        if x_mask is not None:
            edge_features = jnp.where(x_mask[:, None], edge_features, 0.0)
        return edge_features


class ScalarEdgeBoundaryMessages_v2(nnx.Module):
    """ScalarEdgeBoundaryMessages with normalised gate: [Ĝ, rbf(d)] → tanh gate."""

    def __init__(
        self,
        num_node_scalars: int,
        num_edge_scalars: int,
        distance_encoder: nnx.Module,
        hidden_dim: int = 64,
        geometric_filter_dim: int = 64,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        if rngs is None:
            rngs = nnx.Rngs(0)

        self.distance_encoder = distance_encoder
        gate_input_dim = 5 + distance_encoder.num_basis

        self.message_mlp = nnx.Sequential(
            nnx.Linear(num_node_scalars, hidden_dim, rngs=rngs),
            nnx.silu,
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
        )
        self.agg_norm = nnx.LayerNorm(num_features=hidden_dim, rngs=rngs)
        self.output_mlp = nnx.Sequential(
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
            nnx.silu,
            nnx.Linear(hidden_dim, num_edge_scalars, rngs=rngs),
        )
        self.geo_gate = nnx.Sequential(
            nnx.Linear(gate_input_dim, geometric_filter_dim, rngs=rngs),
            nnx.silu,
            nnx.Linear(geometric_filter_dim, num_edge_scalars, rngs=rngs),
            jnp.tanh,
        )

    def __call__(
        self,
        x: jnp.ndarray,
        x_mask: jnp.ndarray,
        boundary_x: jnp.ndarray,
        boundary_senders: jnp.ndarray,
        boundary_receivers: jnp.ndarray,
        boundary_mask: Optional[jnp.ndarray] = None,
        static: Optional[Dict] = None,
    ) -> jnp.ndarray:
        G = static['G']
        G_norm = static['G_norm']
        d = static['distance']
        num_edges = G.shape[0]

        G_hat = G / (G_norm[:, None] + EPS)
        rbf = self.distance_encoder(d)

        node_features = boundary_x[boundary_senders]
        node_contrib = self.message_mlp(node_features)
        if boundary_mask is not None:
            node_contrib = jnp.where(boundary_mask[:, None], node_contrib, 0.0)

        edge_hidden = jax.ops.segment_sum(node_contrib, boundary_receivers, num_edges)
        edge_hidden = self.agg_norm(edge_hidden)
        update = self.output_mlp(edge_hidden)

        geo_gate = self.geo_gate(jnp.concatenate([G_hat, rbf], axis=-1))
        update = update * geo_gate

        if x_mask is not None:
            update = jnp.where(x_mask[:, None], update, 0.0)
        return update


class ScalarEdgeDownMessages_v2(nnx.Module):
    """ScalarEdgeDownMessages with normalised gate: [ΔĜ, Legendre(alignment)] → tanh gate.

    ΔĜ = Ĝ_sender − Ĝ_receiver (purely angular, bounded).
    Alignment is the Frobenius inner product of unit gyration tensors,
    expanded into a Legendre polynomial basis.
    """

    def __init__(
        self,
        num_edge_scalars: int,
        num_node_scalars: int,
        hidden_dim: int = 64,
        geometric_filter_dim: int = 64,
        legendre_basis_dim: int = 8,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        if rngs is None:
            rngs = nnx.Rngs(0)

        self.legendre_basis_dim = legendre_basis_dim
        gate_input_dim = 5 + legendre_basis_dim  # ΔĜ (5) + Legendre(alignment) (K)

        self.message_mlp = nnx.Sequential(
            nnx.Linear(2 * num_edge_scalars + num_node_scalars, hidden_dim, rngs=rngs),
            nnx.silu,
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
            nnx.silu,
            nnx.Linear(hidden_dim, num_edge_scalars, rngs=rngs),
        )
        self.geo_gate = nnx.Sequential(
            nnx.Linear(gate_input_dim, geometric_filter_dim, rngs=rngs),
            nnx.silu,
            nnx.Linear(geometric_filter_dim, num_edge_scalars, rngs=rngs),
            jnp.tanh,
        )

    def __call__(
        self,
        x: jnp.ndarray,
        x_mask: jnp.ndarray,
        node_features: jnp.ndarray,
        down_senders: jnp.ndarray,
        down_receivers: jnp.ndarray,
        down_intermediaries: jnp.ndarray,
        down_mask: Optional[jnp.ndarray] = None,
        static: Optional[Dict] = None,
    ) -> jnp.ndarray:
        G = static['G']
        num_edges = x.shape[0]

        #Calculate relative gyration
        delta_G = G[down_receivers] - G[down_senders]
        delta_G_norm = compute_frobenius_norm(delta_G)

        # Normalise gyration tensors
        delta_G_hat = delta_G / (delta_G_norm[:, None] +EPS)

        # Tensor alignment in [-1, 1] → Legendre polynomials
        alignment = compute_tensor_alignment(
            G[down_senders], G[down_receivers], num_channels=1
        ).squeeze(-1)
        legendre = compute_legendre_basis(alignment, self.legendre_basis_dim)

        # Gather features
        e_s = x[down_senders]
        e_r = x[down_receivers]
        n_k = node_features[down_intermediaries]

        messages = self.message_mlp(jnp.concatenate([e_r, e_s, n_k], axis=-1))

        gate = self.geo_gate(jnp.concatenate([delta_G_hat, legendre], axis=-1))
        messages = messages * gate

        if down_mask is not None:
            messages = jnp.where(down_mask[:, None], messages, 0.0)

        update = jax.ops.segment_sum(messages, down_receivers, num_edges)
        if x_mask is not None:
            update = jnp.where(x_mask[:, None], update, 0.0)
        return update


# =============================================================================
# EdgeGeometryReminder (equivariant, for EGNN_v2)
# =============================================================================

class EdgeGeometryReminder(EquivariantEdgeDownMessages):
    """Inject geometric primitives into edge L=1 and L=2 channels.

    Operates on the edge-edge down-adjacency (shared-node messages):

    - **L=0**: zeros (no scalar geometric primitive).
    - **L=1**: normalised cross product
      ``(sender_far − shared) × (receiver_far − shared)``.
    - **L=2**: normalised ΔĜ = (G_sender − G_receiver) / ||ΔG||_F.

    The gate combines feature invariants with Legendre polynomials on
    gyration-tensor alignment (inherited from EquivariantEdgeDownMessages).
    """

    def __call__(
        self,
        x: jnp.ndarray,
        x_mask: jnp.ndarray,
        node_features: jnp.ndarray,
        down_senders: jnp.ndarray,
        down_receivers: jnp.ndarray,
        down_intermediaries: jnp.ndarray,
        down_mask: Optional[jnp.ndarray] = None,
        static: Optional[Dict] = None,
        node_static: Optional[Dict] = None,
        **kwargs,
    ) -> Dict:
        if static is None or 'G' not in static:
            raise ValueError("static dict must contain 'G' key for gyration tensors")

        G = static['G']
        atoms = static['atoms']       # (n_edges, 2): atom indices per edge
        pos = node_static['pos'] if node_static is not None else static['pos']

        # --- Compute gate (feature + geometric) ---
        alignment = compute_tensor_alignment(
            G[down_senders], G[down_receivers], num_channels=1
        ).squeeze(-1)

        s = x[down_senders]
        r = x[down_receivers]
        inter = node_features[down_intermediaries]

        invariants = self._compute_invariants(s, r, inter)
        feature_gate = self.feature_gate_mlp(invariants)
        geo_basis = self._compute_geo_basis(alignment)
        geo_gate = self.geo_gate_mlp(geo_basis)
        combined_gate = feature_gate * geo_gate

        # --- L=1: normalised cross product ---
        sender_a0, sender_a1 = atoms[down_senders, 0], atoms[down_senders, 1]
        sender_far = jnp.where(sender_a0 == down_intermediaries, sender_a1, sender_a0)

        receiver_a0, receiver_a1 = atoms[down_receivers, 0], atoms[down_receivers, 1]
        receiver_far = jnp.where(receiver_a0 == down_intermediaries, receiver_a1, receiver_a0)

        v_s = pos[sender_far] - pos[down_intermediaries]
        v_r = pos[receiver_far] - pos[down_intermediaries]
        cross = jnp.cross(v_s, v_r)
        cross_norm = jnp.sqrt(jnp.sum(cross ** 2, axis=-1, keepdims=True) + EPS)
        cross_hat = cross / (cross_norm + EPS)          # (num_msg, 3)

        # --- L=2: normalised ΔĜ ---
        delta_G = G[down_senders] - G[down_receivers]
        delta_G_norm = compute_frobenius_norm(delta_G)
        delta_G_hat = delta_G / (delta_G_norm[:, None] + EPS)  # (num_msg, 5)

        # --- Build mapped output ---
        num_msg = s.shape[0]
        mapped = jnp.zeros((num_msg, self.receiver_info.total_dim))

        if self.sender_info.num_l1 > 0 and self.receiver_info.num_l1 > 0:
            num_l1 = self.receiver_info.num_l1
            s_l1 = jnp.repeat(cross_hat[:, None, :], repeats=num_l1, axis=1).reshape(
                num_msg, num_l1 * 3)
            mapped = mapped.at[:, self.receiver_info.l1_indices].set(s_l1)

        if self.sender_info.num_l2 > 0 and self.receiver_info.num_l2 > 0:
            num_l2 = self.receiver_info.num_l2
            s_l2 = jnp.repeat(delta_G_hat[:, None, :], repeats=num_l2, axis=1).reshape(
                num_msg, num_l2 * 5)
            mapped = mapped.at[:, self.receiver_info.l2_indices].set(s_l2)

        # Apply gate and aggregate
        expanded_gate = combined_gate[:, self.receiver_info.gate_indices]
        messages = expanded_gate * mapped

        output = self.aggregate(messages, down_receivers, x.shape[0], mask=down_mask)

        if x_mask is not None:
            output = jnp.where(x_mask[:, None], output, 0.0)

        return {'x': output}
