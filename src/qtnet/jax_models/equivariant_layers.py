from typing import Optional, Dict, List, Tuple

import jax.numpy as jnp
import jax
import flax.nnx as nnx
from qtnet.jax_models.dynamic_activations import activation

from qtnet.jax_models.base_cochain import BaseCochainTensorProduct, EquivariantMessageLayer
from qtnet.jax_models.layer_utils import (
    # Constants
    FROBENIUS_WEIGHTS_L2,
    EPS,
    # Irreps metadata
    IrrepsInfo,
    build_irreps_info,
    extract_invariants,
    expand_gates,
    # Feature extraction
    extract_by_indices,
    # Norm computations
    compute_l1_norms,
    compute_l2_norms,
    # Inner products
    compute_tensor_alignment,
    # Radial basis functions
    compute_radial_basis_bessel,
    compute_smooth_cutoff,
    # Traceless outer product
    compute_traceless_outer_product,
    # Index utilities
    build_l_indices,
    # Channel-wise linear application
    apply_channel_wise_linear_l1,
    apply_channel_wise_linear_l2,
)


class EquivariantNorm(BaseCochainTensorProduct):
    """
    Equivariant normalization layer.
    
    Applies:
    - LayerNorm to scalar (L=0) features
    - E3Norm to L>0 features: x_normalized = w * x / (eps + sqrt(mean(||x||^2)))
      with learnable scale w per channel
    
    This is a standalone normalization layer without residual connection.
    For normalization with residual, use EquivariantGatedResidual.
    
    All operations are JIT-compatible.
    
    Args:
        irreps: Irreps specification (e.g., "8x0e + 4x1o + 2x2e")
        eps: Small constant for numerical stability (default: 1e-5)
        rngs: Flax random number generator
    """
    
    def __init__(
        self,
        irreps: str,
        eps: float = 1e-5,
        rngs: nnx.Rngs = None,
    ):
        super().__init__(irreps_in=irreps, aggr="add", rngs=rngs)
        
        self.eps = eps
        
        # Get irreps info
        info = self._irreps_info['x']
        irreps_obj = info['irreps']
        
        # Precompute irreps metadata
        self.x_info = self.register_space('x', irreps_obj)
        self.scalar_mul = self.x_info.num_l0

        # Use Flax's LayerNorm for scalars
        if self.scalar_mul > 0:
            self.scalar_norm = nnx.LayerNorm(
                num_features=self.scalar_mul,
                epsilon=eps,
                rngs=self.rngs
            )
        else:
            self.scalar_norm = None

        # Learnable scale per L-value group (one w per channel within each L).
        # Separated by L so _apply_e3norm can process all channels of a given L
        # in a single vectorised op instead of one scatter per channel.
        self.w_l1 = nnx.Param(jnp.ones(self.x_info.num_l1)) if self.x_info.num_l1 > 0 else None
        self.w_l2 = nnx.Param(jnp.ones(self.x_info.num_l2)) if self.x_info.num_l2 > 0 else None

        # Precompute indices
        self.scalar_indices = self.x_info.l0_indices
        self.total_dim = self.x_info.total_dim
        # Backwards-compatible: also provide nonscalar_slices list used by older tests
        #self.nonscalar_slices = []
        #offset = 0
        #for mul, ir in irreps_obj:
        #    for _ in range(mul):
        #        if ir.l > 0:
        #            self.nonscalar_slices.append((offset, offset + ir.dim, ir.l))
        #        offset += ir.dim
    
    def __call__(
        self,
        x: jnp.ndarray,
        x_mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Apply equivariant normalization.
        
        Args:
            x: Features of shape (num_cells, feature_dim)
            x_mask: Optional mask for valid cells (num_cells,). Masked cells get zero output.
            
        Returns:
            Normalized features of shape (num_cells, feature_dim)
        """
        output = x
        
        # === LayerNorm for scalars ===
        if self.scalar_indices is not None:
            scalars = output[:, self.scalar_indices]
            scalars_norm = self.scalar_norm(scalars)
            output = output.at[:, self.scalar_indices].set(scalars_norm)
        
        # === E3Norm for non-scalars ===
        output = self._apply_e3norm(output, x_mask)
        
        # Apply mask to output if provided
        if x_mask is not None:
            output = jnp.where(x_mask[:, None], output, 0.0)
        
        return output
    
    def _apply_e3norm(
        self,
        output: jnp.ndarray,
        x_mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Apply E3Norm to all non-scalar channels (per-entity, per-channel).
        
        E3Norm(x_{i,c}) = w_c * x_{i,c} / (eps + ||x_{i,c}||)
        
        Each irrep instance of each entity is independently normalized,
        following the standard used by MACE / e3nn EquivariantLayerNormV2.
        
        Uses proper Frobenius weights for L=2 tensor norm computation.
        """
        # --- L=1 (vectors): (N, num_l1*3) → (N, num_l1, 3) → normalise → flatten ---
        if self.x_info.num_l1 > 0 and self.x_info.l1_indices is not None:
            l1 = output[:, self.x_info.l1_indices]                          # (N, num_l1*3)
            l1_r = l1.reshape(-1, self.x_info.num_l1, 3)                   # (N, num_l1, 3)
            norms = jnp.sqrt(
                jnp.sum(l1_r ** 2, axis=-1, keepdims=True) + self.eps)     # (N, num_l1, 1)
            # w_l1: (num_l1,) → broadcast over N and xyz
            normalised = self.w_l1.value[None, :, None] * l1_r / norms     # (N, num_l1, 3)
            output = output.at[:, self.x_info.l1_indices].set(
                normalised.reshape(-1, self.x_info.num_l1 * 3))

        # --- L=2 (tensors): (N, num_l2*5) → (N, num_l2, 5) → normalise → flatten ---
        if self.x_info.num_l2 > 0 and self.x_info.l2_indices is not None:
            l2 = output[:, self.x_info.l2_indices]                          # (N, num_l2*5)
            l2_r = l2.reshape(-1, self.x_info.num_l2, 5)                   # (N, num_l2, 5)
            norms = jnp.sqrt(
                jnp.sum(
                    FROBENIUS_WEIGHTS_L2 * l2_r ** 2, axis=-1, keepdims=True)
                + self.eps)                                                  # (N, num_l2, 1)
            normalised = self.w_l2.value[None, :, None] * l2_r / norms     # (N, num_l2, 5)
            output = output.at[:, self.x_info.l2_indices].set(
                normalised.reshape(-1, self.x_info.num_l2 * 5))

        return output


class EquivariantNodeEncoder(BaseCochainTensorProduct):
    """
    Initial node feature encoder that generates L=0 (scalar) and L=1 (vector) features
    from chemical embeddings and geometric information.
    
    This layer computes initial node features via message passing over the neighbor graph:
    
        h_i^{(0)} = sum_j m^s_{ij} * Γ_0^s(||r_{ij}||)
        x_i^{(0)} = sum_j m^v_{ij} * Γ_0^v(||r_{ij}||) ⊗ r̂_{ij}
    
    where:
        - m_{ij} = Ω_0(EMBED(Z_i), EMBED(Z_j)) is a message from chemical embeddings
        - Γ_0^s, Γ_0^v are distance‑dependent learnable MLP filters (previously radial basis)
        - r̂_{ij} is the unit direction vector from node i to node j
        - m^s_{ij} and m^v_{ij} are scalar and vector message components
    
    The chemical embedding is provided externally (from a shared ChemicalEmbedding instance).
    
    Args:
        num_scalar_out: Number of output scalar (L=0) channels
        num_vector_out: Number of output vector (L=1) channels
        embedding_dim: Dimension of the chemical embeddings
        hidden_dim: Hidden dimension for the message MLP Ω_0 (default: 64)
        geometric_filter_dim: Hidden dimension for the geometry MLPs (default: 32)
        aggr: Aggregation type ('add', 'mean'). Default: 'add'
        rngs: Flax random number generator
    """
    
    def __init__(
        self,
        num_scalar_out: int,
        num_vector_out: int,
        num_tensor_out: int,
        embedding_dim: int,
        hidden_dim: int = 64,
        geometric_filter_dim: int = 32,
        geo_basis_dim: int = 16,
        cutoff: float = 5.0,
        aggr: str = "add",
        rngs: nnx.Rngs = None,
    ):
        # Build output irreps string (now includes L=2)
        irreps_out = f"{num_scalar_out}x0e + {num_vector_out}x1o + {num_tensor_out}x2e"
        super().__init__(irreps_in=irreps_out, aggr=aggr, rngs=rngs)
        
        self.num_scalar_out = num_scalar_out
        self.num_vector_out = num_vector_out
        self.num_tensor_out = num_tensor_out
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.geometric_filter_dim = geometric_filter_dim
        self.geo_basis_dim = geo_basis_dim
        self.cutoff = cutoff
        
        # Total message dimension: scalars + vectors + tensors (one scalar gate per channel)
        message_dim = num_scalar_out + num_vector_out + num_tensor_out
        
        # === Message MLP: Ω_0(EMBED(Z_i), EMBED(Z_j)) ===
        omega_input_dim = 2 * embedding_dim
        self.omega_mlp = nnx.Sequential(
            nnx.Linear(omega_input_dim, hidden_dim, rngs=self.rngs),
            activation(),
            nnx.Linear(hidden_dim, hidden_dim, rngs=self.rngs),
            activation(),
            nnx.Linear(hidden_dim, message_dim, rngs=self.rngs),
        )
        
        # === Distance-based geometric filters (RBF input) ===
        self.gamma_scalar = nnx.Sequential(
            nnx.Linear(geo_basis_dim, geometric_filter_dim, rngs=self.rngs),
            activation(),
            nnx.Linear(geometric_filter_dim, num_scalar_out, rngs=self.rngs),
            nnx.sigmoid,
        )
        
        self.gamma_vector = nnx.Sequential(
            nnx.Linear(geo_basis_dim, geometric_filter_dim, rngs=self.rngs),
            activation(),
            nnx.Linear(geometric_filter_dim, num_vector_out, rngs=self.rngs),
            nnx.sigmoid,
        )
        
        self.gamma_tensor = nnx.Sequential(
            nnx.Linear(geo_basis_dim, geometric_filter_dim, rngs=self.rngs),
            activation(),
            nnx.Linear(geometric_filter_dim, num_tensor_out, rngs=self.rngs),
            nnx.sigmoid,
        )
        
        # Output dimension
        self.output_dim = num_scalar_out + num_vector_out * 3 + num_tensor_out * 5
    
    def __call__(
        self,
        chem_embeddings: jnp.ndarray,
        x_mask: jnp.ndarray,
        up_senders: jnp.ndarray,
        up_receivers: jnp.ndarray,
        up_mask: Optional[jnp.ndarray] = None,
        static: Optional[Dict[str, jnp.ndarray]] = None,
        **kwargs
    ) -> jnp.ndarray:
        """
        Generate initial node features from chemical embeddings and geometry.
        
        Args:
            chem_embeddings: Chemical embeddings of shape (num_nodes, embedding_dim)
            x_mask: Mask for valid nodes of shape (num_nodes,)
            up_senders: Sender node indices for neighbor edges (num_edges,)
            up_receivers: Receiver node indices for neighbor edges (num_edges,)
            up_mask: Optional mask for valid edges (num_edges,)
            static: Dict containing 'pos' with node positions (num_nodes, 3)
            
        Returns:
            Initial node features of shape (num_nodes, output_dim) with L=0+L=1+L=2
        """
        num_nodes = chem_embeddings.shape[0]
        num_edges = up_senders.shape[0]
        
        # Get positions
        if static is None or 'pos' not in static:
            raise ValueError("static dict must contain 'pos' key with node positions")
        pos = static['pos']
        
        # === Gather chemical embeddings for sender/receiver ===
        emb_s = chem_embeddings[up_senders]   # (num_edges, embedding_dim)
        emb_r = chem_embeddings[up_receivers]  # (num_edges, embedding_dim)
        
        # === Compute geometric quantities ===
        pos_s = pos[up_senders]
        pos_r = pos[up_receivers]
        r_ij = pos_r - pos_s  # Vector from receiver to sender (direction of message)
        distances = jnp.sqrt(jnp.sum(r_ij**2, axis=-1) + EPS)
        r_ij_unit = r_ij / (distances[:, None] + EPS)
        
        # === RBF expansion of distances with smooth cutoff ===
        rbf = compute_radial_basis_bessel(distances, self.geo_basis_dim, self.cutoff)  # (num_edges, geo_basis_dim)
        cutoff_env = compute_smooth_cutoff(distances, self.cutoff)  # (num_edges,)
        rbf = rbf * cutoff_env[:, None]
        
        # === Message from chemical embeddings: Ω_0(EMBED(Z_i), EMBED(Z_j)) ===
        omega_input = jnp.concatenate([emb_r, emb_s], axis=-1)  # (num_edges, 2*embedding_dim)
        m_ij = self.omega_mlp(omega_input)  # (num_edges, message_dim)
        
        # Split into scalar, vector, and tensor message components
        m_scalar = m_ij[:, :self.num_scalar_out]
        m_vector = m_ij[:, self.num_scalar_out:self.num_scalar_out + self.num_vector_out]
        m_tensor = m_ij[:, self.num_scalar_out + self.num_vector_out:]
        
        # === Distance-based filters (RBF input) ===
        gamma_s = self.gamma_scalar(rbf)   # (num_edges, num_scalar_out)
        gamma_v = self.gamma_vector(rbf)   # (num_edges, num_vector_out)
        gamma_t = self.gamma_tensor(rbf)   # (num_edges, num_tensor_out)
        
        # === Compute scalar messages: m^s * Γ^s(RBF(||r||)) ===
        scalar_messages = m_scalar * gamma_s  # (num_edges, num_scalar_out)
        
        # === Compute vector messages: m^v * Γ^v(RBF(||r||)) ⊗ r̂ ===
        vector_coeffs = m_vector * gamma_v  # (num_edges, num_vector_out)
        vector_messages = vector_coeffs[:, :, None] * r_ij_unit[:, None, :]  # (num_edges, num_vector_out, 3)
        vector_messages = vector_messages.reshape(num_edges, -1)  # (num_edges, num_vector_out * 3)
        
        # === Compute tensor messages: m^t * Γ^t(RBF(||r||)) ⊗ (r̂⊗r̂ - I/3) ===
        T_ij = compute_traceless_outer_product(r_ij_unit)  # (num_edges, 5)
        tensor_coeffs = m_tensor * gamma_t  # (num_edges, num_tensor_out)
        tensor_messages = tensor_coeffs[:, :, None] * T_ij[:, None, :]  # (num_edges, num_tensor_out, 5)
        tensor_messages = tensor_messages.reshape(num_edges, -1)  # (num_edges, num_tensor_out * 5)
        
        # === Concatenate all messages ===
        messages = jnp.concatenate([scalar_messages, vector_messages, tensor_messages], axis=-1)
        
        # === Apply edge mask ===
        if up_mask is not None:
            messages = jnp.where(up_mask[:, None], messages, 0.0)
        
        # === Aggregate to nodes ===
        node_features = self.aggregate(
            messages=messages,
            index=up_receivers,
            num_segments=num_nodes,
            mask=up_mask
        )
        
        # === Apply node mask ===
        if x_mask is not None:
            node_features = jnp.where(x_mask[:, None], node_features, 0.0)
        
        return node_features


class EquivariantEdgeEncoder(BaseCochainTensorProduct):
    """
    Generates INITIAL edge features from boundary node features.
    
    Creates L=0 (scalar), L=1 (vector), and L=2 (tensor) edge features directly
    from node features that already carry all three angular momenta. No tensor
    product with the gyration tensor is used — L=1 and L=2 are aggregated from
    the corresponding node features via gated channel-wise linear maps.
    
    L=0: inner MLP(invariants) → aggregate → LayerNorm → outer MLP → Γ(RBF(d))
    L=1: φ_l1(invariants) ⊙ W_l1(node_l1) → aggregate → Γ_l1(RBF(d))
    L=2: φ_l2(invariants) ⊙ W_l2(node_l2) → aggregate → Γ_l2(RBF(d))
    
    Geometric gate uses Bessel RBF of edge interatomic distance.
    
    Args:
        node_irreps_in: Node input irreps (e.g., "8x0e + 4x1o + 2x2e")
        num_scalar_out: Number of output scalar channels
        num_l1_out: Number of output L=1 channels
        num_l2_out: Number of output L=2 channels
        hidden_dim: Hidden dimension for MLPs
        geometric_filter_dim: Hidden dimension for geometric gate MLPs
        geo_basis_dim: Number of Bessel RBF basis functions
        cutoff: Distance cutoff for RBF
        rngs: Flax random number generator
    """
    
    def __init__(
        self,
        node_irreps_in: str,
        num_scalar_out: int = 8,
        num_l1_out: int = 4,
        num_l2_out: int = 4,
        hidden_dim: int = 32,
        geometric_filter_dim: int = 32,
        geo_basis_dim: int = 16,
        cutoff: float = 5.0,
        rngs: nnx.Rngs = None,
    ):
        self.num_scalar_out = num_scalar_out
        self.num_l1_out = num_l1_out
        self.num_l2_out = num_l2_out
        output_irreps = f"{num_scalar_out}x0e + {num_l1_out}x1o + {num_l2_out}x2e"
        
        super().__init__(irreps_in=output_irreps, aggr="add", rngs=rngs)
        
        self.node_irreps_in = node_irreps_in
        self.hidden_dim = hidden_dim
        self.geometric_filter_dim = geometric_filter_dim
        self.geo_basis_dim = geo_basis_dim
        self.cutoff = cutoff
        
        # Parse node input irreps
        node_irreps_obj = self.e3nn.Irreps(node_irreps_in)
        self.node_info = self.register_space('node', node_irreps_obj)
        self.num_node_scalars = self.node_info.num_l0
        self.num_node_l1_channels = self.node_info.num_l1
        self.num_node_l2_channels = self.node_info.num_l2
        
        if self.num_node_l1_channels == 0:
            raise ValueError("node_irreps_in must include L=1 features")
        if self.num_node_l2_channels == 0:
            raise ValueError("node_irreps_in must include L=2 features")
        
        # Invariant MLP input: [scalars, ||L=1||, ||L=2||]
        mlp_input_dim = self.num_node_scalars + self.num_node_l1_channels + self.num_node_l2_channels
        
        # === L=0 path: inner MLP → aggregate → outer MLP ===
        self.scalar_node_mlp = nnx.Sequential(
            nnx.Linear(mlp_input_dim, hidden_dim, rngs=self.rngs),
            activation(),
            nnx.Linear(hidden_dim, hidden_dim, rngs=self.rngs),
        )
        self.scalar_edge_mlp = nnx.Sequential(
            nnx.Linear(hidden_dim, hidden_dim, rngs=self.rngs),
            activation(),
            nnx.Linear(hidden_dim, num_scalar_out, rngs=self.rngs),
        )
        self.agg_norm = nnx.LayerNorm(num_features=hidden_dim, rngs = self.rngs)
        
        # === L=1 path: per-node gate × channel-wise linear ===
        self.phi_l1 = nnx.Sequential(
            nnx.Linear(mlp_input_dim, hidden_dim, rngs=self.rngs),
            activation(),
            nnx.Linear(hidden_dim, num_l1_out, rngs=self.rngs),
        )
        self.W_l1 = nnx.Linear(self.num_node_l1_channels, num_l1_out, rngs=self.rngs)
        
        # === L=2 path: per-node gate × channel-wise linear ===
        self.phi_l2 = nnx.Sequential(
            nnx.Linear(mlp_input_dim, hidden_dim, rngs=self.rngs),
            activation(),
            nnx.Linear(hidden_dim, num_l2_out, rngs=self.rngs),
        )
        self.W_l2 = nnx.Linear(self.num_node_l2_channels, num_l2_out, rngs=self.rngs)
        
        # === Geometric gates (RBF of edge distance) ===
        self.scalar_geo_gate = nnx.Sequential(
            nnx.Linear(geo_basis_dim, geometric_filter_dim, rngs=self.rngs),
            activation(),
            nnx.Linear(geometric_filter_dim, num_scalar_out, rngs=self.rngs),
            nnx.sigmoid,
        )
        self.l1_geo_gate = nnx.Sequential(
            nnx.Linear(geo_basis_dim, geometric_filter_dim, rngs=self.rngs),
            activation(),
            nnx.Linear(geometric_filter_dim, num_l1_out, rngs=self.rngs),
            nnx.sigmoid,
        )
        self.l2_geo_gate = nnx.Sequential(
            nnx.Linear(geo_basis_dim, geometric_filter_dim, rngs=self.rngs),
            activation(),
            nnx.Linear(geometric_filter_dim, num_l2_out, rngs=self.rngs),
            nnx.sigmoid,
        )
        
        # Derive indices from IrrepsInfo
        self.node_scalar_indices = self.node_info.l0_indices
        self.node_l1_indices = self.node_info.l1_indices
        self.node_l2_indices = self.node_info.l2_indices
        
        self.output_irreps = self.e3nn.Irreps(output_irreps)
    
    
    def __call__(
        self,
        boundary_x: jnp.ndarray,
        boundary_senders: jnp.ndarray,
        boundary_receivers: jnp.ndarray,
        boundary_mask: Optional[jnp.ndarray] = None,
        x_mask: Optional[jnp.ndarray] = None,
        static: Optional[Dict] = None,
        **kwargs,
    ) -> Dict:
        if static is None or 'distance' not in static:
            raise ValueError("static dict must contain 'distance' key")
        
        distance = static['distance']
        num_edges = distance.shape[0]
        
        # Gather node features for each boundary message
        x_boundary = boundary_x[boundary_senders]
        
        # Extract components
        node_scalars = extract_by_indices(x_boundary, self.node_scalar_indices)
        node_l1 = extract_by_indices(x_boundary, self.node_l1_indices)
        node_l2 = extract_by_indices(x_boundary, self.node_l2_indices)
        
        # Compute invariants
        l1_norms = compute_l1_norms(node_l1, self.num_node_l1_channels)
        l2_norms = compute_l2_norms(node_l2, self.num_node_l2_channels)
        mlp_input = jnp.concatenate([node_scalars, l1_norms, l2_norms], axis=-1)
        
        # === L=0 path ===
        scalar_contrib = self.scalar_node_mlp(mlp_input)
        scalar_agg = self.aggregate(
            messages=scalar_contrib, index=boundary_receivers,
            num_segments=num_edges, mask=boundary_mask,
        )
        scalar_agg = self.agg_norm(scalar_agg)
        scalar_features = self.scalar_edge_mlp(scalar_agg)
        
        # === L=1 path: gate × channel-wise linear → aggregate ===
        phi_l1_out = self.phi_l1(mlp_input)
        node_l1_mapped = apply_channel_wise_linear_l1(self.W_l1, node_l1, self.num_node_l1_channels)
        l1_mapped_reshaped = node_l1_mapped.reshape(-1, self.num_l1_out, 3)
        l1_messages = (phi_l1_out[:, :, None] * l1_mapped_reshaped).reshape(-1, self.num_l1_out * 3)
        l1_agg = self.aggregate(
            messages=l1_messages, index=boundary_receivers,
            num_segments=num_edges, mask=boundary_mask,
        )
        
        # === L=2 path: gate × channel-wise linear → aggregate ===
        phi_l2_out = self.phi_l2(mlp_input)
        node_l2_mapped = apply_channel_wise_linear_l2(self.W_l2, node_l2, self.num_node_l2_channels)
        l2_mapped_reshaped = node_l2_mapped.reshape(-1, self.num_l2_out, 5)
        l2_messages = (phi_l2_out[:, :, None] * l2_mapped_reshaped).reshape(-1, self.num_l2_out * 5)
        l2_agg = self.aggregate(
            messages=l2_messages, index=boundary_receivers,
            num_segments=num_edges, mask=boundary_mask,
        )
        
        # === Geometric gates (RBF of edge distance with smooth cutoff) ===
        rbf = compute_radial_basis_bessel(distance, self.geo_basis_dim, self.cutoff)
        cutoff_env = compute_smooth_cutoff(distance, self.cutoff)
        rbf = rbf * cutoff_env[:, None]
        gamma_scalar = self.scalar_geo_gate(rbf)
        gamma_l1 = self.l1_geo_gate(rbf)
        gamma_l2 = self.l2_geo_gate(rbf)
        
        gated_scalars = scalar_features * gamma_scalar
        
        l1_agg_reshaped = l1_agg.reshape(-1, self.num_l1_out, 3)
        gated_l1 = (gamma_l1[:, :, None] * l1_agg_reshaped).reshape(-1, self.num_l1_out * 3)
        
        l2_agg_reshaped = l2_agg.reshape(-1, self.num_l2_out, 5)
        gated_l2 = (gamma_l2[:, :, None] * l2_agg_reshaped).reshape(-1, self.num_l2_out * 5)
        
        # Concatenate L=0 + L=1 + L=2
        output = jnp.concatenate([gated_scalars, gated_l1, gated_l2], axis=-1)
        
        if x_mask is not None:
            output = jnp.where(x_mask[:, None], output, 0.0)
        
        return {'x': output}


class EquivariantBagEncoder(BaseCochainTensorProduct):
    """
    Generates bag features from boundary edge features using boundary aggregation.
    
    Simple structure matching EdgeEncoder: extract invariants for feature gate,
    apply gate to edge features, aggregate to bags.
    
    No geometric conditioning — edge features already encode geometry from
    the EdgeEncoder and EdgeDown steps.
    
    Bag features are re-encoded fresh each layer (bags are a communication
    device, not a persistent representation).
    
    Args:
        edge_irreps_in: Edge input irreps specification
        num_scalar_out: Number of output scalar channels
        num_l1_out: Number of output L=1 channels
        num_l2_out: Number of output L=2 channels
        hidden_dim: Hidden dimension for MLPs
        rngs: Flax random number generator
    """
    
    def __init__(
        self,
        edge_irreps_in: str,
        num_scalar_out: int = 8,
        num_l1_out: int = 4,
        num_l2_out: int = 4,
        hidden_dim: int = 32,
        rngs: nnx.Rngs = None,
    ):
        self.num_scalar_out = num_scalar_out
        self.num_l1_out = num_l1_out
        self.num_l2_out = num_l2_out
        output_irreps = f"{num_scalar_out}x0e + {num_l1_out}x1o + {num_l2_out}x2e"
        
        super().__init__(irreps_in=output_irreps, aggr="add", rngs=rngs)
        
        self.edge_irreps_in = edge_irreps_in
        self.hidden_dim = hidden_dim
        
        # Parse edge input irreps
        edge_irreps_obj = self.e3nn.Irreps(edge_irreps_in)
        self.edge_info = self.register_space('edge', edge_irreps_obj)
        
        # Invariant MLP input: [scalars, ||L=1||, ||L=2||]
        mlp_input_dim = self.edge_info.num_instances
        
        # === L=0 path: inner MLP → aggregate → outer MLP ===
        self.scalar_inner_mlp = nnx.Sequential(
            nnx.Linear(mlp_input_dim, hidden_dim, rngs=self.rngs),
            activation(),
            nnx.Linear(hidden_dim, hidden_dim, rngs=self.rngs),
        )
        self.scalar_outer_mlp = nnx.Sequential(
            nnx.Linear(hidden_dim, hidden_dim, rngs=self.rngs),
            activation(),
            nnx.Linear(hidden_dim, num_scalar_out, rngs=self.rngs),
        )
        self.agg_norm = nnx.LayerNorm(num_features=hidden_dim, rngs = self.rngs)
        
        # === L=1 path: per-edge gate × channel-wise linear → aggregate ===
        if self.edge_info.num_l1 > 0 and num_l1_out > 0:
            self.phi_l1 = nnx.Sequential(
                nnx.Linear(mlp_input_dim, hidden_dim, rngs=self.rngs),
                activation(),
                nnx.Linear(hidden_dim, num_l1_out, rngs=self.rngs),
            )
            self.W_l1 = nnx.Linear(self.edge_info.num_l1, num_l1_out, rngs=self.rngs)
        
        # === L=2 path: per-edge gate × channel-wise linear → aggregate ===
        if self.edge_info.num_l2 > 0 and num_l2_out > 0:
            self.phi_l2 = nnx.Sequential(
                nnx.Linear(mlp_input_dim, hidden_dim, rngs=self.rngs),
                activation(),
                nnx.Linear(hidden_dim, num_l2_out, rngs=self.rngs),
            )
            self.W_l2 = nnx.Linear(self.edge_info.num_l2, num_l2_out, rngs=self.rngs)
        
        # Normalize full output (LayerNorm for L=0, E3Norm for L>0)
        self.output_norm = EquivariantNorm(output_irreps, rngs=self.rngs)
        self.output_irreps = self.e3nn.Irreps(output_irreps)
    
    def __call__(
        self,
        boundary_x: jnp.ndarray,
        boundary_senders: jnp.ndarray,
        boundary_receivers: jnp.ndarray,
        boundary_mask: Optional[jnp.ndarray] = None,
        x_mask: Optional[jnp.ndarray] = None,
        **kwargs,
    ) -> Dict:
        num_bags = x_mask.shape[0] if x_mask is not None else int(boundary_receivers.max()) + 1
        
        # Gather edge features for each boundary message
        edge_features = boundary_x[boundary_senders]
        
        # Extract invariants from edge features
        invariants = extract_invariants(edge_features, self.edge_info)
        
        # === L=0 path ===
        scalar_contrib = self.scalar_inner_mlp(invariants)
        scalar_agg = self.aggregate(
            messages=scalar_contrib, index=boundary_receivers,
            num_segments=num_bags, mask=boundary_mask,
        )
        scalar_agg = self.agg_norm(scalar_agg)
        scalar_features = self.scalar_outer_mlp(scalar_agg)
        
        output_parts = [scalar_features]
        
        # === L=1 path ===
        if self.edge_info.num_l1 > 0 and self.num_l1_out > 0:
            edge_l1 = edge_features[:, self.edge_info.l1_indices]
            phi_l1_out = self.phi_l1(invariants)
            l1_mapped = apply_channel_wise_linear_l1(self.W_l1, edge_l1, self.edge_info.num_l1)
            l1_reshaped = l1_mapped.reshape(-1, self.num_l1_out, 3)
            l1_messages = (phi_l1_out[:, :, None] * l1_reshaped).reshape(-1, self.num_l1_out * 3)
            l1_agg = self.aggregate(
                messages=l1_messages, index=boundary_receivers,
                num_segments=num_bags, mask=boundary_mask,
            )
            output_parts.append(l1_agg)
        
        # === L=2 path ===
        if self.edge_info.num_l2 > 0 and self.num_l2_out > 0:
            edge_l2 = edge_features[:, self.edge_info.l2_indices]
            phi_l2_out = self.phi_l2(invariants)
            l2_mapped = apply_channel_wise_linear_l2(self.W_l2, edge_l2, self.edge_info.num_l2)
            l2_reshaped = l2_mapped.reshape(-1, self.num_l2_out, 5)
            l2_messages = (phi_l2_out[:, :, None] * l2_reshaped).reshape(-1, self.num_l2_out * 5)
            l2_agg = self.aggregate(
                messages=l2_messages, index=boundary_receivers,
                num_segments=num_bags, mask=boundary_mask,
            )
            output_parts.append(l2_agg)
        
        output = jnp.concatenate(output_parts, axis=-1)
        output = self.output_norm(output, x_mask)
        
        if x_mask is not None:
            output = jnp.where(x_mask[:, None], output, 0.0)
        
        return {'x': output}


class EquivariantEdgeCoboundaryMessages(EquivariantMessageLayer):
    """Bag → edge coboundary messages. No geometric filter.

    Thin wrapper around :class:`EquivariantMessageLayer`.
    """

    def __init__(
        self,
        edge_irreps: str,
        bag_irreps: str,
        hidden_dim: int = 32,
        hidden_l1_channels: int = 4,
        hidden_l2_channels: int = 4,
        rngs: nnx.Rngs = None,
    ):
        super().__init__(
            sender_irreps=bag_irreps,
            receiver_irreps=edge_irreps,
            hidden_dim=hidden_dim,
            hidden_l1_channels=hidden_l1_channels,
            hidden_l2_channels=hidden_l2_channels,
            geo_basis_type=None,
            aggr='add',
            rngs=rngs,
        )
        self.edge_irreps = edge_irreps
        self.bag_irreps = bag_irreps

    def __call__(
        self,
        x: jnp.ndarray,
        x_mask: Optional[jnp.ndarray],
        coboundary_x: jnp.ndarray,
        coboundary_senders: jnp.ndarray,
        coboundary_receivers: jnp.ndarray,
        coboundary_mask: Optional[jnp.ndarray] = None,
        **kwargs,
    ) -> Dict:
        return super().__call__(
            sender_features=coboundary_x,
            receiver_features=x,
            senders=coboundary_senders,
            receivers=coboundary_receivers,
            num_receivers=x.shape[0],
            mask=coboundary_mask,
            receiver_mask=x_mask,
        )


class EquivariantGatedResidual(BaseCochainTensorProduct):
    """
    Provides two methods:
        pre_norm(x, x_mask): EquivariantNorm before a sub-layer
        residual(x, update, x_mask): x + sigmoid_gate(x) * update
    
    The gate sees invariants extracted from the UN-NORMED residual stream x:
        [L=0 scalars, ||L=1 channels||, ||L=2 channels||_F]
    These are fed through a sigmoid MLP producing one gate value per instance,
    then expanded to the full feature dimension.
    
    Args:
        irreps: Irreps specification (e.g., "8x0e + 4x1o + 2x2e")
        gate_hidden_dim: Hidden dimension for the gate MLP (default: 64)
        eps: Small constant for numerical stability (default: 1e-5)
        rngs: Flax random number generator
    """
    
    def __init__(
        self,
        irreps: str,
        gate_hidden_dim: int = 64,
        eps: float = 1e-5,
        rngs: nnx.Rngs = None,
    ):
        super().__init__(irreps_in=irreps, aggr="add", rngs=rngs)
        
        self.norm = EquivariantNorm(irreps, eps=eps, rngs=rngs)
        
        # Precompute irreps metadata (channel counts, indices, gate mapping)
        self.x_info = self.register_space('x', irreps)
        
        self.gate_mlp = nnx.Sequential(
            nnx.Linear(self.x_info.num_instances, gate_hidden_dim, rngs=self.rngs),
            activation(),
            nnx.Linear(gate_hidden_dim, self.x_info.num_instances, rngs=self.rngs),
            nnx.sigmoid,
        )
    
    def pre_norm(
        self,
        x: jnp.ndarray,
        x_mask: jnp.ndarray = None,
    ) -> jnp.ndarray:
        """Apply EquivariantNorm before a sub-layer."""
        return self.norm(x, x_mask)
    
    def residual(
        self,
        x: jnp.ndarray,
        update: jnp.ndarray,
        x_mask: jnp.ndarray = None,
    ) -> jnp.ndarray:
        """Apply sigmoid-gated residual: x + gate(x) * update."""
        gates = self.gate_mlp(extract_invariants(x, self.x_info))
        output = x + expand_gates(gates, self.x_info.gate_indices) * update
        if x_mask is not None:
            output = jnp.where(x_mask[:, None], output, 0.0)
        return output


class EquivariantFFN(BaseCochainTensorProduct):
    """
    Equivariant feed-forward network (position-wise nonlinear transform).
    
    Expects pre-normalized features (via EquivariantNorm or EquivariantGatedResidual).
    
    Extracts invariants [L=0 scalars, ||L=1||, ||L=2||_F], feeds them through
    an MLP that outputs scalar updates + L>0 gates, then applies:
        - L=0: direct scalar update (added to features)
        - L>0: gate * input features
    
    Args:
        irreps: Irreps specification (e.g., "8x0e + 4x1o + 2x2e")
        hidden_dim: Hidden dimension for MLP (default: 64)
        rngs: Flax random number generator
    """
    
    def __init__(
        self,
        irreps: str,
        hidden_dim: int = 64,
        rngs: nnx.Rngs = None,
    ):
        super().__init__(irreps_in=irreps, aggr="add", rngs=rngs)
        
        self.hidden_dim = hidden_dim

        # Precompute irreps metadata
        info = self._irreps_info['x']
        irreps_obj = info['irreps']
        self.x_info = self.register_space('x', irreps_obj)
        
        # MLP: invariants → scalar update + L>0 gates
        # Input: num_instances (scalars + L=1 norms + L=2 norms)
        # Output: num_l0 (scalar update) + num_l1 + num_l2 (gates)
        mlp_output_dim = self.x_info.num_l0 + self.x_info.num_l1 + self.x_info.num_l2
        
        self.mlp = nnx.Sequential(
            nnx.Linear(self.x_info.num_instances, hidden_dim, rngs=self.rngs),
            activation(),
            nnx.Linear(hidden_dim, hidden_dim, rngs=self.rngs),
            activation(),
            nnx.Linear(hidden_dim, mlp_output_dim, rngs=self.rngs),
        )
    
    def __call__(
        self,
        x: jnp.ndarray,
        x_mask: Optional[jnp.ndarray] = None,
        **kwargs,
    ) -> Dict:
        invariants = extract_invariants(x, self.x_info)
        mlp_out = self.mlp(invariants)
        
        # Split into scalar update + L>0 gates
        num_l0 = self.x_info.num_l0
        num_l1 = self.x_info.num_l1
        scalar_update = mlp_out[:, :num_l0]
        l_gt0_gates = mlp_out[:, num_l0:]
        
        # Build output: scalar update placed directly, L>0 gated
        output = jnp.zeros_like(x)
        if num_l0 > 0 and self.x_info.l0_indices is not None:
            output = output.at[:, self.x_info.l0_indices].set(scalar_update)
        
        if num_l1 + self.x_info.num_l2 > 0:
            # Build full gate vector: 1.0 for scalars (unused), actual gates for L>0
            full_gates = jnp.concatenate(
                [jnp.ones((x.shape[0], num_l0)), l_gt0_gates], axis=-1)
            expanded = full_gates[:, self.x_info.gate_indices]
            # For L>0: gate * x. For L=0: we already set scalar_update above.
            l_gt0_output = expanded * x
            # Only write L>0 portions
            if self.x_info.l1_indices is not None:
                output = output.at[:, self.x_info.l1_indices].set(
                    l_gt0_output[:, self.x_info.l1_indices])
            if self.x_info.l2_indices is not None:
                output = output.at[:, self.x_info.l2_indices].set(
                    l_gt0_output[:, self.x_info.l2_indices])
        
        if x_mask is not None:
            output = jnp.where(x_mask[:, None], output, 0.0)
        
        return {'x': output}


class EquivariantNodeCoboundaryMessages(EquivariantMessageLayer):
    """Edge → node messages. No geometric filter.

    Thin wrapper around :class:`EquivariantMessageLayer` that preserves the
    original call signature used by *models_equivariant.py*.
    """

    def __init__(
        self,
        node_irreps: str,
        edge_irreps: str,
        hidden_dim: int = 32,
        hidden_l1_channels: int = 4,
        hidden_l2_channels: int = 4,
        use_tensor_products: bool = False,
        hidden_tp_dim: int = 8,
        rngs: nnx.Rngs = None,
    ):
        super().__init__(
            sender_irreps=edge_irreps,
            receiver_irreps=node_irreps,
            hidden_dim=hidden_dim,
            hidden_l1_channels=hidden_l1_channels,
            hidden_l2_channels=hidden_l2_channels,
            geo_basis_type=None,
            aggr='add',
            use_tensor_products=use_tensor_products,
            hidden_tp_dim=hidden_tp_dim,
            rngs=rngs,
        )
        self.node_irreps = node_irreps
        self.edge_irreps = edge_irreps

    def __call__(
        self,
        x: jnp.ndarray,
        x_mask: Optional[jnp.ndarray],
        coboundary_x: jnp.ndarray,
        coboundary_senders: jnp.ndarray,
        coboundary_receivers: jnp.ndarray,
        coboundary_mask: Optional[jnp.ndarray] = None,
        **kwargs,
    ) -> Dict:
        return super().__call__(
            sender_features=coboundary_x,
            receiver_features=x,
            senders=coboundary_senders,
            receivers=coboundary_receivers,
            num_receivers=x.shape[0],
            mask=coboundary_mask,
            receiver_mask=x_mask,
        )


class EquivariantNodeBoundaryMessages(EquivariantMessageLayer):
    """Bag → node messages. No geometric filter.

    Thin wrapper around :class:`EquivariantMessageLayer`.
    """

    def __init__(
        self,
        node_irreps: str,
        bag_irreps: str,
        hidden_dim: int = 32,
        hidden_l1_channels: int = 4,
        hidden_l2_channels: int = 4,
        rngs: nnx.Rngs = None,
    ):
        super().__init__(
            sender_irreps=bag_irreps,
            receiver_irreps=node_irreps,
            hidden_dim=hidden_dim,
            hidden_l1_channels=hidden_l1_channels,
            hidden_l2_channels=hidden_l2_channels,
            geo_basis_type=None,
            aggr='add',
            rngs=rngs,
        )
        self.node_irreps = node_irreps
        self.bag_irreps = bag_irreps

    def __call__(
        self,
        x: jnp.ndarray,
        x_mask: Optional[jnp.ndarray],
        boundary_x: jnp.ndarray,
        boundary_senders: jnp.ndarray,
        boundary_receivers: jnp.ndarray,
        boundary_mask: Optional[jnp.ndarray] = None,
        **kwargs,
    ) -> Dict:
        return super().__call__(
            sender_features=boundary_x,
            receiver_features=x,
            senders=boundary_senders,
            receivers=boundary_receivers,
            num_receivers=x.shape[0],
            mask=boundary_mask,
            receiver_mask=x_mask,
        )


class EquivariantDualChannelUpdate(BaseCochainTensorProduct):
    """
    Combines two update channels with learned per-instance gating.
    
    Used wherever two parallel update streams feed the same feature vector:
    message-passing update + geometry-injection update. A shared gate
    backbone produces per-instance weights for each channel. Gates are
    independent sigmoids (not softmax), so both channels can contribute fully.
    
    Gate backbone takes invariants from the UN-NORMED residual stream:
        [L=0 scalars, ||L=1 channels||, ||L=2 channels||_F]
    
    These are fed through a shared MLP backbone, then two sigmoid heads
    produce per-instance gates (one per irrep instance) that are expanded
    to the full feature dimension via _gate_indices.
    
    Output: x + expanded_g_msg * message_update + expanded_g_geo * geometry_update
    
    Args:
        irreps: Irreps specification (e.g., "8x0e + 4x1o + 2x2e")
        gate_hidden_dim: Hidden dimension for the gate MLP (default: 64)
        eps: Small constant for numerical stability (default: 1e-5)
        rngs: Flax random number generator
    """
    
    def __init__(
        self,
        irreps: str,
        gate_hidden_dim: int = 64,
        eps: float = 1e-5,
        rngs: nnx.Rngs = None,
    ):
        super().__init__(irreps_in=irreps, aggr="add", rngs=rngs)
        
        self.norm = EquivariantNorm(irreps, eps=eps, rngs=rngs)
        
        # Precompute irreps metadata (channel counts, indices, gate mapping)
        self.x_info = self.register_space('x', irreps)
        
        # Shared gate backbone
        self.gate_backbone = nnx.Sequential(
            nnx.Linear(self.x_info.num_instances, gate_hidden_dim, rngs=self.rngs),
            activation(),
        )
        
        # Separate sigmoid heads for message and geometry channels
        self.gate_message_head = nnx.Sequential(
            nnx.Linear(gate_hidden_dim, self.x_info.num_instances, rngs=self.rngs),
            nnx.sigmoid,
        )
        self.gate_geometry_head = nnx.Sequential(
            nnx.Linear(gate_hidden_dim, self.x_info.num_instances, rngs=self.rngs),
            nnx.sigmoid,
        )
    
    def pre_norm(
        self,
        x: jnp.ndarray,
        x_mask: jnp.ndarray = None,
    ) -> jnp.ndarray:
        """Apply EquivariantNorm before computing both update channels."""
        return self.norm(x, x_mask)
    
    def combine(
        self,
        x: jnp.ndarray,
        message_update: jnp.ndarray,
        geometry_update: jnp.ndarray,
        x_mask: jnp.ndarray = None,
    ) -> jnp.ndarray:
        """Combine two update channels with learned gating.
        
        Args:
            x: Un-normed features (residual stream), shape (num_cells, feature_dim)
            message_update: Message-passing update, shape (num_cells, feature_dim)
            geometry_update: Geometry-injection update, shape (num_cells, feature_dim)
            x_mask: Optional valid cell mask
            
        Returns:
            Updated features: x + g_msg * message_update + g_geo * geometry_update
        """
        # Shared backbone → two sigmoid heads
        h = self.gate_backbone(extract_invariants(x, self.x_info))
        g_msg = self.gate_message_head(h)
        g_geo = self.gate_geometry_head(h)
        
        output = x + (expand_gates(g_msg, self.x_info.gate_indices) * message_update
                      + expand_gates(g_geo, self.x_info.gate_indices) * geometry_update)
        
        if x_mask is not None:
            output = jnp.where(x_mask[:, None], output, 0.0)
        return output


class ChemicalReminder(BaseCochainTensorProduct):
    """
    Chemical reminder for equivariant models.
    
    Pre-norms the features via EquivariantNorm, extracts invariants,
    concatenates with projected chemical embeddings, applies MLP to produce
    scalar updates + L>0 gates, then uses gated residual.
    
    Matches the ScalarChemicalReminder pattern: self-contained norm + gate.
    
    Args:
        irreps: Irreps specification for node features
        embedding_dim: Dimension of the chemical embeddings
        hidden_dim: Hidden dimension for MLPs (default: 64)
        eps: Small constant for numerical stability (default: 1e-5)
        rngs: Flax random number generator
    """
    
    def __init__(
        self,
        irreps: str,
        embedding_dim: int,
        hidden_dim: int = 64,
        eps: float = 1e-5,
        rngs: nnx.Rngs = None,
    ):
        super().__init__(irreps_in=irreps, aggr="add", rngs=rngs)
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Precompute irreps metadata
        info = self._irreps_info['x']
        irreps_obj = info['irreps']
        self.x_info = self.register_space('x', irreps_obj)
        
        # Internal pre-norm
        self.norm = EquivariantNorm(irreps, eps=eps, rngs=rngs)
        
        # Embedding projection
        self.emb_projection = nnx.Linear(embedding_dim, embedding_dim, rngs=self.rngs)
        
        # MLP: [invariants, projected_emb] → scalar update + L>0 gates
        mlp_input_dim = self.x_info.num_instances + embedding_dim
        mlp_output_dim = self.x_info.num_l0 + self.x_info.num_l1 + self.x_info.num_l2
        
        self.mlp = nnx.Sequential(
            nnx.Linear(mlp_input_dim, hidden_dim, rngs=self.rngs),
            activation(),
            nnx.Linear(hidden_dim, hidden_dim, rngs=self.rngs),
            activation(),
            nnx.Linear(hidden_dim, mlp_output_dim, rngs=self.rngs),
        )
        
        # Sigmoid gate on un-normed features (for gated residual)
        self.gate_mlp = nnx.Sequential(
            nnx.Linear(self.x_info.num_instances, hidden_dim, rngs=self.rngs),
            activation(),
            nnx.Linear(hidden_dim, self.x_info.num_instances, rngs=self.rngs),
            nnx.sigmoid,
        )
    
    def __call__(
        self,
        x: jnp.ndarray,
        chem_embeddings: jnp.ndarray,
        x_mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        # Pre-norm
        normed = self.norm(x, x_mask)
        
        # Extract invariants from normed features
        invariants = extract_invariants(normed, self.x_info)
        
        # Project embeddings and concatenate
        projected_emb = self.emb_projection(chem_embeddings)
        mlp_input = jnp.concatenate([invariants, projected_emb], axis=-1)
        
        # MLP produces scalar update + L>0 gates
        mlp_out = self.mlp(mlp_input)
        num_l0 = self.x_info.num_l0
        scalar_update = mlp_out[:, :num_l0]
        l_gt0_gates = mlp_out[:, num_l0:]
        
        # Build update: scalar update placed, L>0 gated from normed input
        update = jnp.zeros_like(x)
        if num_l0 > 0 and self.x_info.l0_indices is not None:
            update = update.at[:, self.x_info.l0_indices].set(scalar_update)
        
        num_l_gt0 = self.x_info.num_l1 + self.x_info.num_l2
        if num_l_gt0 > 0:
            full_gates = jnp.concatenate(
                [jnp.ones((x.shape[0], num_l0)), l_gt0_gates], axis=-1)
            expanded = full_gates[:, self.x_info.gate_indices]
            gated = expanded * normed
            if self.x_info.l1_indices is not None:
                update = update.at[:, self.x_info.l1_indices].set(
                    gated[:, self.x_info.l1_indices])
            if self.x_info.l2_indices is not None:
                update = update.at[:, self.x_info.l2_indices].set(
                    gated[:, self.x_info.l2_indices])
        
        # Gated residual using un-normed features
        g = self.gate_mlp(extract_invariants(x, self.x_info))
        output = x + expand_gates(g, self.x_info.gate_indices) * update
        
        if x_mask is not None:
            output = jnp.where(x_mask[:, None], output, 0.0)
        
        return output


class EquivariantNodeUpMessages(EquivariantMessageLayer):
    """Node → node up-adjacency messages. Bessel RBF geo filter on inter-node distance.

    Supports optional edge intermediary enrichment.
    Thin wrapper around :class:`EquivariantMessageLayer`.
    """

    def __init__(
        self,
        irreps_in: str,
        hidden_dim: int = 32,
        hidden_l1_channels: int = 4,
        hidden_l2_channels: int = 4,
        geometric_filter_dim: int = 32,
        geo_basis_dim: int = 16,
        cutoff: float = 5.0,
        edge_irreps: str = None,
        aggr: str = "add",
        use_tensor_products: bool = False,
        hidden_tp_dim: int = 8,
        rngs: nnx.Rngs = None,
    ):
        super().__init__(
            sender_irreps=irreps_in,
            receiver_irreps=irreps_in,
            hidden_dim=hidden_dim,
            hidden_l1_channels=hidden_l1_channels,
            hidden_l2_channels=hidden_l2_channels,
            geo_basis_type='rbf_bessel',
            geo_basis_dim=geo_basis_dim,
            geo_filter_dim=geometric_filter_dim,
            cutoff=cutoff,
            intermediary_irreps=edge_irreps,
            aggr=aggr,
            use_tensor_products=use_tensor_products,
            hidden_tp_dim=hidden_tp_dim,
            rngs=rngs,
        )
        self.cutoff = cutoff

    def __call__(
        self,
        x: jnp.ndarray,
        x_mask: jnp.ndarray,
        up_senders: jnp.ndarray,
        up_receivers: jnp.ndarray,
        up_mask: Optional[jnp.ndarray] = None,
        static: Optional[Dict[str, jnp.ndarray]] = None,
        edge_features: Optional[jnp.ndarray] = None,
        up_intermediaries: Optional[jnp.ndarray] = None,
        **kwargs,
    ) -> Dict[str, jnp.ndarray]:
        if static is None or 'pos' not in static:
            raise ValueError("static dict must contain 'pos' key with node positions")

        pos = static['pos']
        pos_s = pos[up_senders]
        pos_r = pos[up_receivers]
        distances = jnp.sqrt(jnp.sum((pos_r - pos_s) ** 2, axis=-1) + EPS)

        return super().__call__(
            sender_features=x,
            receiver_features=x,
            senders=up_senders,
            receivers=up_receivers,
            num_receivers=x.shape[0],
            mask=up_mask,
            receiver_mask=x_mask,
            geo_quantity=distances,
            intermediary_features=edge_features,
            intermediary_indices=up_intermediaries,
            static=static,
        )


class EquivariantEdgeBoundaryMessages(EquivariantMessageLayer):
    """Node → edge boundary messages. Bessel RBF geo filter on edge distance.

    Thin wrapper around :class:`EquivariantMessageLayer`.
    """

    def __init__(
        self,
        node_irreps: str,
        edge_irreps: str,
        hidden_dim: int = 32,
        hidden_l1_channels: int = 4,
        hidden_l2_channels: int = 4,
        geometric_filter_dim: int = 32,
        geo_basis_dim: int = 16,
        cutoff: float = 5.0,
        rngs: nnx.Rngs = None,
    ):
        super().__init__(
            sender_irreps=node_irreps,
            receiver_irreps=edge_irreps,
            hidden_dim=hidden_dim,
            hidden_l1_channels=hidden_l1_channels,
            hidden_l2_channels=hidden_l2_channels,
            geo_basis_type='rbf_bessel',
            geo_basis_dim=geo_basis_dim,
            geo_filter_dim=geometric_filter_dim,
            cutoff=cutoff,
            aggr='add',
            rngs=rngs,
        )
        self.node_irreps = node_irreps
        self.edge_irreps = edge_irreps

    def __call__(
        self,
        x: jnp.ndarray,
        x_mask: Optional[jnp.ndarray],
        boundary_x: jnp.ndarray,
        boundary_senders: jnp.ndarray,
        boundary_receivers: jnp.ndarray,
        boundary_mask: Optional[jnp.ndarray] = None,
        static: Optional[Dict] = None,
        **kwargs,
    ) -> Dict:
        if static is None or 'distance' not in static:
            raise ValueError("static dict must contain 'distance' key")

        # Each message's geo quantity is the receiving edge's interatomic distance
        geo_quantity = static['distance'][boundary_receivers]

        return super().__call__(
            sender_features=boundary_x,
            receiver_features=x,
            senders=boundary_senders,
            receivers=boundary_receivers,
            num_receivers=x.shape[0],
            mask=boundary_mask,
            receiver_mask=x_mask,
            geo_quantity=geo_quantity,
        )


class EquivariantEdgeDownMessages(EquivariantMessageLayer):
    """Edge → edge down-adjacency messages via shared nodes. Legendre geo filter
    on gyration tensor alignment only (no norms).

    Thin wrapper around :class:`EquivariantMessageLayer`.
    """

    def __init__(
        self,
        edge_irreps: str,
        node_irreps: str,
        hidden_dim: int = 32,
        hidden_l1_channels: int = 4,
        hidden_l2_channels: int = 4,
        geometric_filter_dim: int = 32,
        geo_basis_dim: int = 16,
        rngs: nnx.Rngs = None,
    ):
        super().__init__(
            sender_irreps=edge_irreps,
            receiver_irreps=edge_irreps,
            hidden_dim=hidden_dim,
            hidden_l1_channels=hidden_l1_channels,
            hidden_l2_channels=hidden_l2_channels,
            geo_basis_type='legendre',
            geo_basis_dim=geo_basis_dim,
            geo_filter_dim=geometric_filter_dim,
            intermediary_irreps=node_irreps,
            aggr='add',
            rngs=rngs,
        )
        self.edge_irreps = edge_irreps
        self.node_irreps = node_irreps

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
        **kwargs,
    ) -> Dict:
        if static is None or 'G' not in static:
            raise ValueError("static dict must contain 'G' key for gyration tensors")

        G = static['G']
        alignment = compute_tensor_alignment(
            G[down_senders], G[down_receivers], num_channels=1
        ).squeeze(-1)  # (num_messages,)

        return super().__call__(
            sender_features=x,
            receiver_features=x,
            senders=down_senders,
            receivers=down_receivers,
            num_receivers=x.shape[0],
            mask=down_mask,
            receiver_mask=x_mask,
            geo_quantity=alignment,
            intermediary_features=node_features,
            intermediary_indices=down_intermediaries,
        )


class NodeHead(BaseCochainTensorProduct):
    """Node head that projects internal features to output predictions with appropriate
    transformations for each irrep type.
    
    For molecular property prediction, the output might be:
    - 2 scalars: N (electron count), LI (localization index)  
    - 1 vector: dipole moment (μ_x, μ_y, μ_z)
    - 1 tensor: quadrupole moment (Q_xy, Q_xz, Q_yz, Q_aniso, Q_zz) - traceless
    
    Transformation:
        h_out = MLP(h_in)    (L=0 scalars - MLP is allowed, scalars are invariant)
        v_out = W_v @ v_in   (L=1 vectors - channel-wise linear, same weight for all 3 components)
        T_out = W_t @ T_in   (L=2 tensors - channel-wise linear, same weight for all 5 components)
    
    Scalars can use MLP (they're invariant). Vectors and tensors use channel-wise
    linear projections (via apply_channel_wise_linear_l1/l2) to preserve equivariance.
    
    Args:
        irreps_in: Input irreps specification (e.g., "8x0e + 4x1o + 2x2e")
        num_scalar_out: Number of output scalar channels (default: 2)
        num_vector_out: Number of output vector channels (default: 1)
        num_tensor_out: Number of output tensor channels (default: 1)
        hidden_dim: Hidden dimension for scalar MLP (default: 64)
        use_mlp: Whether to use MLP for scalars (default: True)
        rngs: Flax random number generator
    """
    
    def __init__(
        self,
        irreps_in: str,
        num_scalar_out: int = 2,
        num_vector_out: int = 1,
        num_tensor_out: int = 1,
        hidden_dim: int = 64,
        use_mlp: bool = True,
        rngs: nnx.Rngs = None,
    ):
        # Output irreps: scalars + vectors + tensors
        self.num_scalar_out = num_scalar_out
        self.num_vector_out = num_vector_out
        self.num_tensor_out = num_tensor_out
        output_irreps = f"{num_scalar_out}x0e + {num_vector_out}x1o + {num_tensor_out}x2e"
        
        super().__init__(irreps_in=irreps_in, aggr="add", rngs=rngs)
        
        self.output_irreps = output_irreps
        self.hidden_dim = hidden_dim
        self.use_mlp = use_mlp
        
        # Parse input irreps
        input_irreps_obj = self.e3nn.Irreps(irreps_in)
        output_irreps_obj = self.e3nn.Irreps(output_irreps)
        self.in_info = self.register_space('in', input_irreps_obj)
        self.out_info = self.register_space('out', output_irreps_obj)
        
        # Derive channel counts and indices from IrrepsInfo
        self.num_l0_in = self.in_info.num_l0
        self.num_l1_in = self.in_info.num_l1
        self.num_l2_in = self.in_info.num_l2
        self.l0_indices_in = self.in_info.l0_indices
        self.l1_indices_in = self.in_info.l1_indices
        self.l2_indices_in = self.in_info.l2_indices
        self.l0_indices_out = self.out_info.l0_indices
        self.l1_indices_out = self.out_info.l1_indices
        self.l2_indices_out = self.out_info.l2_indices
        
        # Total output dimension
        self.output_dim = num_scalar_out + num_vector_out * 3 + num_tensor_out * 5
        
        # === Scalar projection (MLP allowed - scalars are invariant) ===
        if self.num_l0_in > 0 and num_scalar_out > 0:
            if use_mlp:
                self.scalar_mlp = nnx.Sequential(
                    nnx.Linear(self.num_l0_in, hidden_dim, rngs=self.rngs),
                    activation(),
                    nnx.Linear(hidden_dim, hidden_dim, rngs=self.rngs),
                    activation(),
                    nnx.Linear(hidden_dim, num_scalar_out, rngs=self.rngs),
                )
            else:
                self.scalar_linear = nnx.Linear(
                    self.num_l0_in, num_scalar_out, rngs=self.rngs
                )
        
        # === Vector projection (L=1) - channel-wise linear for equivariance ===
        # Maps num_l1_in -> num_vector_out channels, applied identically to all 3 components.
        if self.num_l1_in > 0 and num_vector_out > 0:
            self.vector_linear = nnx.Linear(self.num_l1_in, num_vector_out, rngs=self.rngs)
        else:
            self.vector_linear = None
        
        # === Tensor projection (L=2) - channel-wise linear for equivariance ===
        # Maps num_l2_in -> num_tensor_out channels, applied identically to all 5 components.
        if self.num_l2_in > 0 and num_tensor_out > 0:
            self.tensor_linear = nnx.Linear(self.num_l2_in, num_tensor_out, rngs=self.rngs)
        else:
            self.tensor_linear = None
    
    def __call__(
        self,
        x: jnp.ndarray,
        x_mask: Optional[jnp.ndarray] = None,
        **kwargs,
    ) -> Dict:
        """
        Project node features to output representation.
        
        Args:
            x: Node features of shape (num_nodes, input_dim)
            x_mask: Optional mask for valid nodes (num_nodes,)
            
        Returns:
            Dictionary with 'x' containing projected features (num_nodes, output_dim)
            and individual predictions for convenience:
            - 'scalars': (num_nodes, num_scalar_out)
            - 'vectors': (num_nodes, num_vector_out * 3)
            - 'tensors': (num_nodes, num_tensor_out * 5)
        """
        num_nodes = x.shape[0]
        
        # === Extract input features by irrep type using precomputed indices ===
        scalars_in = x[:, self.l0_indices_in] if self.num_l0_in > 0 and self.l0_indices_in is not None else None
        vectors_in = x[:, self.l1_indices_in] if self.num_l1_in > 0 and self.l1_indices_in is not None else None
        tensors_in = x[:, self.l2_indices_in] if self.num_l2_in > 0 and self.l2_indices_in is not None else None
        
        # === Project scalars (MLP allowed - scalars are invariant) ===
        if self.num_l0_in > 0 and self.num_scalar_out > 0 and scalars_in is not None:
            if self.use_mlp:
                scalars_out = self.scalar_mlp(scalars_in)
            else:
                scalars_out = self.scalar_linear(scalars_in)
        else:
            scalars_out = jnp.zeros((num_nodes, self.num_scalar_out))
        
        # === Project vectors (channel-wise linear - equivariant) ===
        if self.num_l1_in > 0 and self.num_vector_out > 0 and vectors_in is not None and self.vector_linear is not None:
            vectors_out = apply_channel_wise_linear_l1(self.vector_linear, vectors_in, self.num_l1_in)
        else:
            vectors_out = jnp.zeros((num_nodes, self.num_vector_out * 3))
        
        # === Project tensors (channel-wise linear - equivariant) ===
        if self.num_l2_in > 0 and self.num_tensor_out > 0 and tensors_in is not None and self.tensor_linear is not None:
            tensors_out = apply_channel_wise_linear_l2(self.tensor_linear, tensors_in, self.num_l2_in)
        else:
            tensors_out = jnp.zeros((num_nodes, self.num_tensor_out * 5))
        
        # === Assemble output ===
        output = jnp.concatenate([scalars_out, vectors_out, tensors_out], axis=-1)
        
        # === Apply mask ===
        if x_mask is not None:
            output = jnp.where(x_mask[:, None], output, 0.0)
            scalars_out = jnp.where(x_mask[:, None], scalars_out, 0.0)
            vectors_out = jnp.where(x_mask[:, None], vectors_out, 0.0)
            tensors_out = jnp.where(x_mask[:, None], tensors_out, 0.0)
        
        return {
            'x': output,
            'scalars': scalars_out,
            'vectors': vectors_out,
            'tensors': tensors_out,
        }


class EquivariantPerLayerReadout(nnx.Module):
    """
    Per-layer readout with softmax-weighted combination.
    
    Wraps a NodeHead and applies it to node features from each layer,
    then combines outputs using learnable softmax weights (one set each
    for scalars, vectors, and tensors).
    
    Args:
        node_irreps: Node irreps specification
        hidden_dim: Hidden dimension for the internal NodeHead
        num_layers: Total number of layers (determines weight vector size)
        num_scalar_out: Number of output scalars (default: 2)
        num_vector_out: Number of output vectors (default: 1)
        num_tensor_out: Number of output tensors (default: 1)
        rngs: Flax random number generator
    """
    
    def __init__(
        self,
        node_irreps: str,
        hidden_dim: int,
        num_layers: int,
        num_scalar_out: int = 2,
        num_vector_out: int = 1,
        num_tensor_out: int = 1,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        if rngs is None:
            rngs = nnx.Rngs(0)
        
        self.head = NodeHead(
            irreps_in=node_irreps,
            num_scalar_out=num_scalar_out,
            num_vector_out=num_vector_out,
            num_tensor_out=num_tensor_out,
            hidden_dim=hidden_dim,
            use_mlp=True,
            rngs=rngs,
        )
        
        self.scalar_weights = nnx.Param(jnp.zeros(num_layers))
        self.vector_weights = nnx.Param(jnp.zeros(num_layers))
        self.tensor_weights = nnx.Param(jnp.zeros(num_layers))
    
    def __call__(
        self,
        node_features_per_layer: List[jnp.ndarray],
        x_mask: jnp.ndarray,
    ) -> Dict[str, jnp.ndarray]:
        """
        Apply head to each layer's features and combine with softmax weights.
        
        Args:
            node_features_per_layer: List of node features, one per layer
            x_mask: Valid node mask
            
        Returns:
            Dict with 'scalars', 'vectors', 'tensors'
        """
        scalar_w = jax.nn.softmax(self.scalar_weights.value)  # (num_layers,)
        vector_w = jax.nn.softmax(self.vector_weights.value)
        tensor_w = jax.nn.softmax(self.tensor_weights.value)

        # Stack into a single leading-axis array for lax.scan
        stacked = jnp.stack(node_features_per_layer, axis=0)  # (L, N, feat_dim)

        num_nodes = node_features_per_layer[0].shape[0]
        init = (
            jnp.zeros((num_nodes, self.head.num_scalar_out)),
            jnp.zeros((num_nodes, self.head.num_vector_out * 3)),
            jnp.zeros((num_nodes, self.head.num_tensor_out * 5)),
        )

        def body(carry, inputs):
            nf, ws, wv, wt = inputs
            out = self.head(x=nf, x_mask=x_mask)
            s, v, t = carry
            return (
                s + ws * out['scalars'],
                v + wv * out['vectors'],
                t + wt * out['tensors'],
            ), None

        (scalars_sum, vectors_sum, tensors_sum), _ = jax.lax.scan(
            body, init, (stacked, scalar_w, vector_w, tensor_w))

        return {
            'scalars': scalars_sum,
            'vectors': vectors_sum,
            'tensors': tensors_sum,
        }
    

class GeometryReminder(EquivariantNodeUpMessages):
    """
    Same gate computation as NodeUp, but multiplies r̂_ij / T_ij
    instead of sender features.

    _compute_invariants  → inherited (sender+receiver scalars,
                           compressed L1/L2 norms, cross-alignment,
                           intermediary, RBF geo gate) — unchanged
    _map_gate_aggregate  → overridden: gate * geometry instead of
                           gate * mapped_sender_features
    __call__             → inherited unchanged
    """

    def _map_gate_aggregate(
        self,
        sender_gathered: jnp.ndarray,  # (E, sender_dim) — not used
        gate: jnp.ndarray,             # (E, num_receiver_instances)
        senders: jnp.ndarray,
        receivers: jnp.ndarray,
        num_receivers: int,
        mask: Optional[jnp.ndarray],
        static: Optional[Dict[str, jnp.ndarray]] = None,
    ) -> jnp.ndarray:
        
        '''
        Adapt _map_gate_aggregate to gate geometry instead of features
        '''

        if static is None or 'pos' not in static:
            raise ValueError("static dict must contain 'pos' key with node positions")

        pos = static['pos']
        pos_s = pos[senders]
        pos_r = pos[receivers]
        r_ij = pos_r - pos_s  # Vector from receiver to sender (direction of message)
        distances = jnp.sqrt(jnp.sum(r_ij**2, axis=-1) + EPS)
        r_hat = r_ij / (distances[:, None] + EPS)

        T_ij = compute_traceless_outer_product(r_hat)  # (num_edges, 5)
        


        num_msg = sender_gathered.shape[0]
        mapped = jnp.zeros((num_msg, self.receiver_info.total_dim))

        # L=0: no geometric primitive to inject, leave zeros
        # (NodeUp handles the scalar channel; we only renew direction)

        # L=1 (channel-wise)
        if self.sender_info.num_l1 > 0 and self.receiver_info.num_l1 > 0:
            # r_hat: (num_msg, 3) -> expand to (num_msg, num_l1 * 3)
            num_l1 = self.receiver_info.num_l1
            s_l1 = jnp.repeat(r_hat[:, None, :], repeats=num_l1, axis=1).reshape(
                num_msg, num_l1 * 3
            )
            mapped = mapped.at[:, self.receiver_info.l1_indices].set(s_l1)

        # L=2 (channel-wise)
        if self.sender_info.num_l2 > 0 and self.receiver_info.num_l2 > 0:
            # T_ij: (num_msg, 5) -> expand to (num_msg, num_l2 * 5)
            num_l2 = self.receiver_info.num_l2
            s_l2 = jnp.repeat(T_ij[:, None, :], repeats=num_l2, axis=1).reshape(
                num_msg, num_l2 * 5
            )
            mapped = mapped.at[:, self.receiver_info.l2_indices].set(s_l2)

        # Expand gate: (num_msg, num_instances) → (num_msg, total_dim)
        expanded_gate = gate[:, self.receiver_info.gate_indices]
        messages = expanded_gate * mapped

        # Aggregate
        return self.aggregate(messages, receivers, num_receivers, mask=mask)