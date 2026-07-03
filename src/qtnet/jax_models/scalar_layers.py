from typing import Optional, Dict

import jax.numpy as jnp
import jax
import flax.nnx as nnx
from qtnet.jax_models.dynamic_activations import activation


class ChemicalEmbedding(nnx.Module):
    """
    Learnable embedding layer for chemical species with optional atomic features.
    
    Maps discrete chemical species indices to continuous embedding vectors,
    optionally concatenated with additional per-atom features (N, LI, Mu, Q).
    
    When use_atom_features=True:
        - Input: one-hot species (num_species) + 10 atomic features = num_species + 10
        - A linear layer projects this combined input to embedding_dim
    
    When use_atom_features=False:
        - Input: species indices only
        - Embedding lookup from learned embedding matrix
    
    Args:
        num_species: Number of distinct chemical species (vocabulary size)
        embedding_dim: Dimension of the output embedding vectors
        use_atom_features: If True, expect 10 additional atomic features per atom
        num_atom_features: Number of atomic features (default 10: N, LI, Mu_xyz, Q_5comp)
        rngs: Flax random number generator
    """
    
    def __init__(
        self,
        num_species: int,
        embedding_dim: int,
        use_atom_features: bool = False,
        num_atom_features: int = 10,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        if rngs is None:
            rngs = nnx.Rngs(0)
        self.rngs = rngs
        
        self.num_species = num_species
        self.embedding_dim = embedding_dim
        self.use_atom_features = use_atom_features
        self.num_atom_features = num_atom_features
        
        if use_atom_features:
            # Combined input: one-hot species + atomic features
            input_dim = num_species + num_atom_features
            self.projection = nnx.Linear(input_dim, embedding_dim, rngs=rngs)
        else:
            # Initialize embedding matrix with scaled normal initialization
            # Shape: (num_species, embedding_dim)
            key = rngs.params()
            scale = 1.0 / jnp.sqrt(embedding_dim)
            self.embedding_matrix = nnx.Param(
                jax.random.normal(key, (num_species, embedding_dim)) * scale
            )
    
    def __call__(
        self,
        species_indices: jnp.ndarray,
        atom_features: jnp.ndarray = None,
    ) -> jnp.ndarray:
        """
        Look up embeddings for chemical species.
        
        Args:
            species_indices: Integer indices of chemical species, shape (num_nodes,)
                             Values should be in range [0, num_species - 1]
            atom_features: Optional atomic features, shape (num_nodes, num_atom_features)
                           Required when use_atom_features=True
            
        Returns:
            Embedding vectors of shape (num_nodes, embedding_dim)
        """
        if self.use_atom_features:
            # Create one-hot encoding for species
            one_hot = jax.nn.one_hot(species_indices, self.num_species)
            # Concatenate with atomic features
            if atom_features is None:
                raise ValueError("atom_features required when use_atom_features=True")
            combined = jnp.concatenate([one_hot, atom_features], axis=-1)
            return self.projection(combined)
        else:
            return self.embedding_matrix.value[species_indices]


# =============================================================================
# Pre-norm + gated residual building blocks
# =============================================================================

class ScalarGatedResidual(nnx.Module):
    """Pre-norm + gated residual block for scalar features.
    
    Provides two methods:
    - pre_norm(x): LayerNorm normalization of features before a sub-layer
    - residual(x, update, x_mask): x + sigmoid_gate(x) * update
    
    The gate sees un-normed features (the raw residual stream) so it can
    make receptivity decisions based on current feature magnitudes.
    """
    
    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 64,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        if rngs is None:
            rngs = nnx.Rngs(0)
        self.norm = nnx.LayerNorm(num_features=num_features, rngs=rngs)
        self.gate_mlp = nnx.Sequential(
            nnx.Linear(num_features, hidden_dim, rngs=rngs),
            activation(),
            nnx.Linear(hidden_dim, num_features, rngs=rngs),
            nnx.sigmoid,
        )
    
    def pre_norm(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.norm(x)
    
    def residual(
        self,
        x: jnp.ndarray,
        update: jnp.ndarray,
        x_mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        g = self.gate_mlp(x)
        output = x + g * update
        if x_mask is not None:
            output = jnp.where(x_mask[:, None], output, 0.0)
        return output


class ScalarDualChannelNodeUpdate(nnx.Module):
    """Combines edge→node and bag→node updates with learned per-feature gating.
    
    Both channels are computed from the same pre-normed node state.
    A shared gate backbone produces per-feature weights for each channel.
    Gates are independent sigmoids (not softmax), so both channels can
    contribute fully — they carry complementary, not redundant, information.
    """
    
    def __init__(
        self,
        num_node_scalars: int,
        hidden_dim: int = 64,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        if rngs is None:
            rngs = nnx.Rngs(0)
        
        # Pre-norm (shared for both channels)
        self.norm = nnx.LayerNorm(num_features=num_node_scalars, rngs=rngs)
        
        # Shared gate backbone
        self.gate_backbone = nnx.Sequential(
            nnx.Linear(num_node_scalars, hidden_dim, rngs=rngs),
            activation(),
        )
        # Separate heads → per-feature sigmoid gates
        self.gate_edge_head = nnx.Sequential(
            nnx.Linear(hidden_dim, num_node_scalars, rngs=rngs),
            nnx.sigmoid,
        )
        self.gate_bag_head = nnx.Sequential(
            nnx.Linear(hidden_dim, num_node_scalars, rngs=rngs),
            nnx.sigmoid,
        )
    
    def pre_norm(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.norm(x)
    
    def combine(
        self,
        x: jnp.ndarray,
        edge_update: jnp.ndarray,
        bag_update: jnp.ndarray,
        x_mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Combine two update channels with learned gating."""
        h = self.gate_backbone(x)  # gate sees un-normed features
        g_edge = self.gate_edge_head(h)
        g_bag = self.gate_bag_head(h)
        
        combined = g_edge * edge_update + g_bag * bag_update
        output = x + combined
        
        if x_mask is not None:
            output = jnp.where(x_mask[:, None], output, 0.0)
        return output


# =============================================================================
# Encoders
# =============================================================================

class ScalarNodeEncoder(nnx.Module):
    """
    Scalar node encoder using displacement vector components for geometric filtering.
    
    Geometric information is encoded via the 3 components of the relative
    position vector r_ij as filter inputs in a tanh-bounded gate.
    
    h_i = Σ_j MLP(emb_i, emb_j) * tanh_gate(r_ij)
    """
    
    def __init__(
        self,
        num_scalar_out: int,
        embedding_dim: int,
        hidden_dim: int = 64,
        geometric_filter_dim: Optional[int] = None,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        if rngs is None:
            rngs = nnx.Rngs(0)
        self.rngs = rngs
        
        self.num_scalar_out = num_scalar_out
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.geometric_filter_dim = (
            hidden_dim if geometric_filter_dim is None else geometric_filter_dim
        )
        
        # MLP for chemical information
        mlp_input_dim = 2 * embedding_dim
        self.mlp = nnx.Sequential(
            nnx.Linear(mlp_input_dim, hidden_dim, rngs=rngs),
            activation(),
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
            activation(),
            nnx.Linear(hidden_dim, num_scalar_out, rngs=rngs),
        )
        
        # Geometric gate: tanh-bounded, sign-sensitive to r_ij
        self.r_gate = nnx.Sequential(
            nnx.Linear(3, self.geometric_filter_dim, rngs=rngs),
            activation(),
            nnx.Linear(self.geometric_filter_dim, num_scalar_out, rngs=rngs),
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
        """Generate initial scalar node features."""
        num_nodes = chem_embeddings.shape[0]
        
        if static is None or 'pos' not in static:
            raise ValueError("static dict must contain 'pos'")
        pos = static['pos']
        
        # Gather embeddings
        emb_s = chem_embeddings[up_senders]
        emb_r = chem_embeddings[up_receivers]
        
        # Compute displacement vectors
        r_ij = pos[up_receivers] - pos[up_senders]
        
        # MLP for chemical features
        mlp_input = jnp.concatenate([emb_r, emb_s], axis=-1)
        messages = self.mlp(mlp_input)
        
        # Apply geometric gate
        gate = self.r_gate(r_ij)
        messages = messages * gate
        
        # Apply edge mask
        if up_mask is not None:
            messages = jnp.where(up_mask[:, None], messages, 0.0)
        
        # Aggregate to nodes
        node_features = jax.ops.segment_sum(messages, up_receivers, num_nodes)
        
        if x_mask is not None:
            node_features = jnp.where(x_mask[:, None], node_features, 0.0)
        
        return node_features


class ScalarEdgeEncoder(nnx.Module):
    """
    Scalar edge encoder using gyration tensor components for filtering.

    Uses the 5 components of the traceless gyration tensor as filter inputs
    in a tanh-bounded gate.

    h_edge = MLP_outer(Σ_k∈∂(edge) MLP_inner(h_k)) * tanh_gate(G_components)
    """
    
    def __init__(
        self,
        num_node_scalars: int,
        num_scalar_out: int,
        hidden_dim: int = 64,
        geometric_filter_dim: Optional[int] = None,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        if rngs is None:
            rngs = nnx.Rngs(0)
        self.rngs = rngs
        
        self.num_node_scalars = num_node_scalars
        self.num_scalar_out = num_scalar_out
        self.hidden_dim = hidden_dim
        self.geometric_filter_dim = (
            hidden_dim if geometric_filter_dim is None else geometric_filter_dim
        )
        
        # Inner MLP (per-node)
        self.inner_mlp = nnx.Sequential(
            nnx.Linear(num_node_scalars, hidden_dim, rngs=rngs),
            activation(),
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
        )
        
        # Normalize after aggregation (decouples outer MLP from degree)
        self.agg_norm = nnx.LayerNorm(num_features=hidden_dim, rngs=rngs)
        
        # Outer MLP (after aggregation)
        self.outer_mlp = nnx.Sequential(
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
            activation(),
            nnx.Linear(hidden_dim, num_scalar_out, rngs=rngs),
        )
        
        # Geometric gate: tanh-bounded
        self.geo_gate = nnx.Sequential(
            nnx.Linear(5, self.geometric_filter_dim, rngs=rngs),
            activation(),
            nnx.Linear(self.geometric_filter_dim, num_scalar_out, rngs=rngs),
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
        """Generate scalar edge features."""
        if static is None or 'G' not in static:
            raise ValueError("static dict must contain 'G'")
        
        G = static['G']
        num_edges = G.shape[0]
        
        # Gather node features
        node_features = boundary_x[boundary_senders]
        
        # Inner MLP
        node_contrib = self.inner_mlp(node_features)
        
        if boundary_mask is not None:
            node_contrib = jnp.where(boundary_mask[:, None], node_contrib, 0.0)
        
        # Aggregate to edges
        edge_hidden = jax.ops.segment_sum(node_contrib, boundary_receivers, num_edges)
        edge_hidden = self.agg_norm(edge_hidden)
        
        # Outer MLP
        edge_features = self.outer_mlp(edge_hidden)
        
        # Geometric gate
        geo_gate = self.geo_gate(G)
        edge_features = edge_features * geo_gate
        
        if x_mask is not None:
            edge_features = jnp.where(x_mask[:, None], edge_features, 0.0)
        
        return edge_features


# =============================================================================
# Bag-of-bonds layers (replacing ring layers)
# =============================================================================

class ScalarBagEncoder(nnx.Module):
    """
    Scalar bag-of-bonds encoder.
    
    Encodes bag features from member edge features using boundary aggregation.
    No geometric conditioning — edge features already encode geometry from
    the EdgeEncoder and EdgeDown steps.
    
    Bag features are re-encoded fresh each layer (bags are a communication
    device, not a persistent representation).
    """
    
    def __init__(
        self,
        num_edge_scalars: int,
        num_bag_scalars: int,
        hidden_dim: int = 64,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        if rngs is None:
            rngs = nnx.Rngs(0)
        
        self.num_edge_scalars = num_edge_scalars
        self.num_bag_scalars = num_bag_scalars
        
        # Inner MLP (per edge)
        self.inner_mlp = nnx.Sequential(
            nnx.Linear(num_edge_scalars, hidden_dim, rngs=rngs),
            activation(),
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
        )
        
        # Normalize after aggregation (decouples outer MLP from degree)
        self.agg_norm = nnx.LayerNorm(num_features=hidden_dim, rngs=rngs)
        
        # Outer MLP (after aggregation)
        self.outer_mlp = nnx.Sequential(
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
            activation(),
            nnx.Linear(hidden_dim, num_bag_scalars, rngs=rngs),
        )
    
    def __call__(
        self,
        boundary_x: jnp.ndarray,  # edge features
        boundary_senders: jnp.ndarray,
        boundary_receivers: jnp.ndarray,
        boundary_mask: Optional[jnp.ndarray] = None,
        x_mask: Optional[jnp.ndarray] = None,
        **kwargs,
    ) -> jnp.ndarray:
        num_bags = x_mask.shape[0] if x_mask is not None else int(boundary_receivers.max()) + 1
        
        edge_features = boundary_x[boundary_senders]
        edge_contrib = self.inner_mlp(edge_features)
        
        if boundary_mask is not None:
            edge_contrib = jnp.where(boundary_mask[:, None], edge_contrib, 0.0)
        
        bag_hidden = jax.ops.segment_sum(edge_contrib, boundary_receivers, num_bags)
        bag_hidden = self.agg_norm(bag_hidden)
        bag_features = self.outer_mlp(bag_hidden)
        
        if x_mask is not None:
            bag_features = jnp.where(x_mask[:, None], bag_features, 0.0)
        
        return bag_features


class ScalarEdgeCoboundaryMessages(nnx.Module):
    """Scalar edge coboundary messages (bag-to-edge).
    
    Each edge receives messages from its parent bag(s). No geometric filter —
    bag features already encode geometry from EdgeEncoder and EdgeDown steps.
    Uses a message MLP on [edge, bag] features.
    """
    
    def __init__(
        self,
        num_edge_scalars: int,
        num_bag_scalars: int,
        hidden_dim: int = 64,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        if rngs is None:
            rngs = nnx.Rngs(0)
        
        # Message MLP on [receiver_edge, sender_bag]
        self.message_mlp = nnx.Sequential(
            nnx.Linear(num_edge_scalars + num_bag_scalars, hidden_dim, rngs=rngs),
            activation(),
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
            activation(),
            nnx.Linear(hidden_dim, num_edge_scalars, rngs=rngs),
        )
    
    def __call__(
        self,
        x: jnp.ndarray,  # edge features (pre-normed by caller)
        x_mask: jnp.ndarray,
        coboundary_x: jnp.ndarray,  # bag features
        coboundary_senders: jnp.ndarray,
        coboundary_receivers: jnp.ndarray,
        coboundary_mask: Optional[jnp.ndarray] = None,
        **kwargs,
    ) -> jnp.ndarray:
        num_edges = x.shape[0]
        
        edge_features = x[coboundary_receivers]
        bag_features = coboundary_x[coboundary_senders]
        
        # Message MLP
        mlp_input = jnp.concatenate([edge_features, bag_features], axis=-1)
        messages = self.message_mlp(mlp_input)
        
        if coboundary_mask is not None:
            messages = jnp.where(coboundary_mask[:, None], messages, 0.0)
        
        update = jax.ops.segment_sum(messages, coboundary_receivers, num_edges)
        
        if x_mask is not None:
            update = jnp.where(x_mask[:, None], update, 0.0)
        
        return update


# =============================================================================
# Edge feed-forward network
# =============================================================================

class ScalarEdgeFFN(nnx.Module):
    """Per-edge feed-forward network (position-wise nonlinear transform).
    
    Edges only receive two aggregation-based updates (boundary messages and
    bag decoder), so the FFN provides a per-entity nonlinear transform between
    message-passing steps.
    """
    
    def __init__(
        self,
        num_scalars: int,
        hidden_dim: int = 64,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        if rngs is None:
            rngs = nnx.Rngs(0)
        
        self.mlp = nnx.Sequential(
            nnx.Linear(num_scalars, hidden_dim, rngs=rngs),
            activation(),
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
            activation(),
            nnx.Linear(hidden_dim, num_scalars, rngs=rngs),
        )
    
    def __call__(
        self,
        x: jnp.ndarray,
        x_mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        output = self.mlp(x)
        if x_mask is not None:
            output = jnp.where(x_mask[:, None], output, 0.0)
        return output


# =============================================================================
# Message-passing layers
# =============================================================================

class ScalarNodeCoboundaryMessages(nnx.Module):
    """Scalar node coboundary messages (edge-to-node).
    
    Uses a message MLP on [receiver_node, sender_edge] features.
    No internal gating — the residual gate handles update receptivity.
    """
    
    def __init__(
        self,
        num_node_scalars: int,
        num_edge_scalars: int,
        hidden_dim: int = 64,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        if rngs is None:
            rngs = nnx.Rngs(0)
        
        # Message MLP on [receiver_node, sender_edge]
        self.message_mlp = nnx.Sequential(
            nnx.Linear(num_node_scalars + num_edge_scalars, hidden_dim, rngs=rngs),
            activation(),
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
            activation(),
            nnx.Linear(hidden_dim, num_node_scalars, rngs=rngs),
        )
    
    def __call__(
        self,
        x: jnp.ndarray,  # node features (pre-normed by caller)
        x_mask: jnp.ndarray,
        coboundary_x: jnp.ndarray,  # edge features
        coboundary_senders: jnp.ndarray,
        coboundary_receivers: jnp.ndarray,
        coboundary_mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        num_nodes = x.shape[0]
        
        node_features = x[coboundary_receivers]
        edge_features = coboundary_x[coboundary_senders]
        
        # Message MLP
        mlp_input = jnp.concatenate([node_features, edge_features], axis=-1)
        messages = self.message_mlp(mlp_input)
        
        if coboundary_mask is not None:
            messages = jnp.where(coboundary_mask[:, None], messages, 0.0)
        
        update = jax.ops.segment_sum(messages, coboundary_receivers, num_nodes)
        
        if x_mask is not None:
            update = jnp.where(x_mask[:, None], update, 0.0)
        
        return update


class ScalarNodeBoundaryMessages(nnx.Module):
    """Scalar node boundary messages (bag-to-node, long-range communication).
    
    Despite bags being higher-dimensional, this is a boundary relation
    in the wrapped/periodic structure: bags are in the boundary of nodes.
    
    For bag type (X, Y), all atoms of element X or Y receive messages.
    This enables one-step communication between distant atoms that share
    a bond type.
    
    Uses a message MLP on [receiver_node, sender_bag] features.
    Gating is handled externally by ScalarDualChannelNodeUpdate.
    """
    
    def __init__(
        self,
        num_node_scalars: int,
        num_bag_scalars: int,
        hidden_dim: int = 64,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        if rngs is None:
            rngs = nnx.Rngs(0)
        
        # Message MLP on [receiver_node, sender_bag]
        self.message_mlp = nnx.Sequential(
            nnx.Linear(num_node_scalars + num_bag_scalars, hidden_dim, rngs=rngs),
            activation(),
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
            activation(),
            nnx.Linear(hidden_dim, num_node_scalars, rngs=rngs),
        )
    
    def __call__(
        self,
        x: jnp.ndarray,  # node features (pre-normed by caller)
        x_mask: jnp.ndarray,
        boundary_x: jnp.ndarray,  # bag features
        boundary_senders: jnp.ndarray,  # bag indices
        boundary_receivers: jnp.ndarray,  # node indices
        boundary_mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        num_nodes = x.shape[0]
        
        node_features = x[boundary_receivers]
        bag_features = boundary_x[boundary_senders]
        
        # Message MLP
        mlp_input = jnp.concatenate([node_features, bag_features], axis=-1)
        messages = self.message_mlp(mlp_input)
        
        if boundary_mask is not None:
            messages = jnp.where(boundary_mask[:, None], messages, 0.0)
        
        update = jax.ops.segment_sum(messages, boundary_receivers, num_nodes)
        
        if x_mask is not None:
            update = jnp.where(x_mask[:, None], update, 0.0)
        
        return update


class ScalarChemicalReminder(nnx.Module):
    """Chemical reminder for scalar models.
    
    Pre-norms the features, concatenates with chemical embeddings, applies MLP,
    then uses gated residual. Placed at the end of each message-passing cycle
    so the readout head receives chemically grounded features.
    """
    
    def __init__(
        self,
        num_scalars: int,
        embedding_dim: int,
        hidden_dim: int = 64,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        if rngs is None:
            rngs = nnx.Rngs(0)
        
        self.norm = nnx.LayerNorm(num_features=num_scalars, rngs=rngs)
        self.mlp = nnx.Sequential(
            nnx.Linear(num_scalars + embedding_dim, hidden_dim, rngs=rngs),
            activation(),
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
            activation(),
            nnx.Linear(hidden_dim, num_scalars, rngs=rngs),
        )
        self.gate_mlp = nnx.Sequential(
            nnx.Linear(num_scalars, hidden_dim, rngs=rngs),
            activation(),
            nnx.Linear(hidden_dim, num_scalars, rngs=rngs),
            nnx.sigmoid,
        )
        self.emb_projection = nnx.Linear(embedding_dim, embedding_dim, rngs=rngs)
    
    def __call__(
        self,
        x: jnp.ndarray,
        chem_embeddings: jnp.ndarray,
        x_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        normed = self.norm(x)
        projected_emb = self.emb_projection(chem_embeddings)
        mlp_input = jnp.concatenate([normed, projected_emb], axis=-1)
        update = self.mlp(mlp_input)
        g = self.gate_mlp(x)
        output = x + g * update
        if x_mask is not None:
            output = jnp.where(x_mask[:, None], output, 0.0)
        return output


class ScalarNodeUpMessages(nnx.Module):
    """Scalar node up messages (node-to-node via neighbors).

    Uses tanh-bounded geometric gate for sign-sensitive directional filtering.
    Optionally conditions on intermediary edge features (edge-enriched messages).
    """
    
    def __init__(
        self,
        num_scalars: int,
        hidden_dim: int = 64,
        num_edge_scalars: int = 0,
        geometric_filter_dim: Optional[int] = None,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        if rngs is None:
            rngs = nnx.Rngs(0)
        self.rngs = rngs
        self.num_scalars = num_scalars
        self.hidden_dim = hidden_dim
        self.num_edge_scalars = num_edge_scalars
        self.geometric_filter_dim = (
            hidden_dim if geometric_filter_dim is None else geometric_filter_dim
        )
        
        # Message MLP: takes [h_r, h_s] or [h_r, h_s, e_ij]
        mlp_input_dim = 2 * num_scalars + num_edge_scalars
        self.message_mlp = nnx.Sequential(
            nnx.Linear(mlp_input_dim, hidden_dim, rngs=rngs),
            activation(),
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
            activation(),
            nnx.Linear(hidden_dim, num_scalars, rngs=rngs),
        )
        
        # Geometric gate: tanh-bounded, sign-sensitive
        self.r_gate = nnx.Sequential(
            nnx.Linear(3, self.geometric_filter_dim, rngs=rngs),
            activation(),
            nnx.Linear(self.geometric_filter_dim, num_scalars, rngs=rngs),
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
        
        if static is None or 'pos' not in static:
            raise ValueError("static must contain 'pos'")
        pos = static['pos']
        
        # Gather features
        h_s = x[up_senders]
        h_r = x[up_receivers]
        
        # Displacement vector components
        r_ij = pos[up_senders] - pos[up_receivers]
        
        # Message MLP: include edge features if available
        if self.num_edge_scalars > 0 and edge_features is not None and up_intermediaries is not None:
            e_ij = edge_features[up_intermediaries]
            mlp_input = jnp.concatenate([h_r, h_s, e_ij], axis=-1)
        else:
            mlp_input = jnp.concatenate([h_r, h_s], axis=-1)
        messages = self.message_mlp(mlp_input)
        
        # Apply r_ij gate
        gate = self.r_gate(r_ij)
        messages = messages * gate
        
        # Mask and aggregate
        if up_mask is not None:
            messages = jnp.where(up_mask[:, None], messages, 0.0)
        
        update = jax.ops.segment_sum(messages, up_receivers, num_nodes)
        
        if x_mask is not None:
            update = jnp.where(x_mask[:, None], update, 0.0)
        
        return update


class ScalarEdgeBoundaryMessages(nnx.Module):
    """Scalar edge boundary messages (node-to-edge).

    Uses tanh-bounded geometric gate for the gyration tensor components.
    """
    
    def __init__(
        self,
        num_node_scalars: int,
        num_edge_scalars: int,
        hidden_dim: int = 64,
        geometric_filter_dim: Optional[int] = None,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        if rngs is None:
            rngs = nnx.Rngs(0)
        self.rngs = rngs
        self.hidden_dim = hidden_dim
        self.geometric_filter_dim = (
            hidden_dim if geometric_filter_dim is None else geometric_filter_dim
        )
        
        self.message_mlp = nnx.Sequential(
            nnx.Linear(num_node_scalars, hidden_dim, rngs=rngs),
            activation(),
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
        )
        
        # Normalize after aggregation (decouples output MLP from degree)
        self.agg_norm = nnx.LayerNorm(num_features=hidden_dim, rngs=rngs)
        
        self.output_mlp = nnx.Sequential(
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
            activation(),
            nnx.Linear(hidden_dim, num_edge_scalars, rngs=rngs),
        )
        
        # Geometric gate: tanh-bounded
        self.geo_gate = nnx.Sequential(
            nnx.Linear(5, self.geometric_filter_dim, rngs=rngs),
            activation(),
            nnx.Linear(self.geometric_filter_dim, num_edge_scalars, rngs=rngs),
            jnp.tanh,
        )
    
    def __call__(
        self,
        x: jnp.ndarray,  # edge features (not used in computation)
        x_mask: jnp.ndarray,
        boundary_x: jnp.ndarray,  # node features
        boundary_senders: jnp.ndarray,
        boundary_receivers: jnp.ndarray,
        boundary_mask: Optional[jnp.ndarray] = None,
        static: Optional[Dict] = None,
    ) -> jnp.ndarray:
        if static is None or 'G' not in static:
            raise ValueError("static must contain 'G'")
        
        G = static['G']
        num_edges = G.shape[0]
        
        node_features = boundary_x[boundary_senders]
        node_contrib = self.message_mlp(node_features)
        
        if boundary_mask is not None:
            node_contrib = jnp.where(boundary_mask[:, None], node_contrib, 0.0)
        
        edge_hidden = jax.ops.segment_sum(node_contrib, boundary_receivers, num_edges)
        edge_hidden = self.agg_norm(edge_hidden)
        update = self.output_mlp(edge_hidden)
        
        geo_gate = self.geo_gate(G)
        update = update * geo_gate
        
        if x_mask is not None:
            update = jnp.where(x_mask[:, None], update, 0.0)
        
        return update


class ScalarEdgeDownMessages(nnx.Module):
    """Scalar edge down messages (edge-to-edge via shared nodes).

    Messages between edges that share a node (angular/bond-angle interactions).
    Uses tanh-bounded geometric gate on ΔG = G_sender - G_receiver (relative
    gyration tensor, 5-dim), encoding angular relationships between bonds.

    Intermediary node features provide chemical context for the shared atom.
    """
    
    def __init__(
        self,
        num_edge_scalars: int,
        num_node_scalars: int,
        hidden_dim: int = 64,
        geometric_filter_dim: Optional[int] = None,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        if rngs is None:
            rngs = nnx.Rngs(0)
        self.hidden_dim = hidden_dim
        self.geometric_filter_dim = (
            hidden_dim if geometric_filter_dim is None else geometric_filter_dim
        )
        
        # Message MLP: takes [edge_receiver, edge_sender, node_intermediary]
        self.message_mlp = nnx.Sequential(
            nnx.Linear(2 * num_edge_scalars + num_node_scalars, hidden_dim, rngs=rngs),
            activation(),
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
            activation(),
            nnx.Linear(hidden_dim, num_edge_scalars, rngs=rngs),
        )
        
        # Geometric gate: tanh-bounded, on relative gyration tensor ΔG
        self.geo_gate = nnx.Sequential(
            nnx.Linear(5, self.geometric_filter_dim, rngs=rngs),
            activation(),
            nnx.Linear(self.geometric_filter_dim, num_edge_scalars, rngs=rngs),
            jnp.tanh,
        )
    
    def __call__(
        self,
        x: jnp.ndarray,  # edge features (pre-normed by caller)
        x_mask: jnp.ndarray,
        node_features: jnp.ndarray,  # node features (intermediary)
        down_senders: jnp.ndarray,
        down_receivers: jnp.ndarray,
        down_intermediaries: jnp.ndarray,
        down_mask: Optional[jnp.ndarray] = None,
        static: Optional[Dict] = None,
    ) -> jnp.ndarray:
        if static is None or 'G' not in static:
            raise ValueError("static must contain 'G'")
        
        G = static['G']
        num_edges = x.shape[0]
        
        # Gather features
        e_s = x[down_senders]
        e_r = x[down_receivers]
        n_k = node_features[down_intermediaries]
        
        # Relative gyration tensor (angular information)
        delta_G = G[down_senders] - G[down_receivers]
        
        # Message MLP
        mlp_input = jnp.concatenate([e_r, e_s, n_k], axis=-1)
        messages = self.message_mlp(mlp_input)
        
        # Apply ΔG gate
        gate = self.geo_gate(delta_G)
        messages = messages * gate
        
        # Mask and aggregate
        if down_mask is not None:
            messages = jnp.where(down_mask[:, None], messages, 0.0)
        
        update = jax.ops.segment_sum(messages, down_receivers, num_edges)
        
        if x_mask is not None:
            update = jnp.where(x_mask[:, None], update, 0.0)
        
        return update


# =============================================================================
# Output head
# =============================================================================

class ScalarAtomicHead(nnx.Module):
    """
    Output head for scalar models.
    
    Includes a LayerNorm before the backbone to normalize the residual stream
    features before prediction.
    
    Outputs:
    - 2 scalars: N, LI
    - 3 scalars: dipole components (treated as scalars, not equivariant)
    - 5 scalars: quadrupole components (treated as scalars, not equivariant)
    """
    
    def __init__(
        self,
        num_scalars_in: int,
        hidden_dim: int = 64,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        if rngs is None:
            rngs = nnx.Rngs(0)
        
        self.norm = nnx.LayerNorm(num_features=num_scalars_in, rngs=rngs)
        
        # Shared backbone
        self.backbone = nnx.Sequential(
            nnx.Linear(num_scalars_in, hidden_dim, rngs=rngs),
            activation(),
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
            activation(),
        )
        
        # Output heads
        self.head_scalars = nnx.Linear(hidden_dim, 2, rngs=rngs)  # N, LI
        self.head_vectors = nnx.Linear(hidden_dim, 3, rngs=rngs)  # dipole components
        self.head_tensors = nnx.Linear(hidden_dim, 5, rngs=rngs)  # quadrupole components
    
    def __call__(
        self,
        x: jnp.ndarray,
        x_mask: Optional[jnp.ndarray] = None,
    ) -> Dict[str, jnp.ndarray]:
        h = self.norm(x)
        h = self.backbone(h)
        
        scalars = self.head_scalars(h)
        vectors = self.head_vectors(h)
        tensors = self.head_tensors(h)
        
        if x_mask is not None:
            scalars = jnp.where(x_mask[:, None], scalars, 0.0)
            vectors = jnp.where(x_mask[:, None], vectors, 0.0)
            tensors = jnp.where(x_mask[:, None], tensors, 0.0)
        
        return {
            'scalars': scalars,
            'vectors': vectors,
            'tensors': tensors,
        }


class PerLayerReadout(nnx.Module):
    """Per-layer readout with learned weighted combination.
    
    Wraps a shared ScalarAtomicHead and maintains per-layer softmax weights
    for each property type (scalars, vectors, tensors). Each layer's
    predictions are combined via a learned weighted sum.
    
    Usage:
        readout = PerLayerReadout(num_scalars_in, hidden_dim, num_layers, rngs)
        
        # After each layer (including layer 0):
        readout.read_layer(node_features, x_mask, layer_idx=i)
        
        # After all layers:
        output = readout.combine(x_mask)
    """
    
    def __init__(
        self,
        num_scalars_in: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        if rngs is None:
            rngs = nnx.Rngs(0)
        
        self.num_layers = num_layers
        
        self.head = ScalarAtomicHead(
            num_scalars_in=num_scalars_in,
            hidden_dim=hidden_dim,
            rngs=rngs,
        )
        
        # Learned weights (initialized to 0 → uniform softmax)
        self.layer_weights_scalars = nnx.Param(jnp.zeros(num_layers))
        self.layer_weights_vectors = nnx.Param(jnp.zeros(num_layers))
        self.layer_weights_tensors = nnx.Param(jnp.zeros(num_layers))
    
    def read_and_combine(
        self,
        node_features_per_layer: list,
        x_mask: Optional[jnp.ndarray] = None,
    ) -> Dict[str, jnp.ndarray]:
        """Compute per-layer predictions and combine with learned weights.
        
        Args:
            node_features_per_layer: list of node feature arrays, one per layer
            x_mask: node mask for zeroing padded positions
        
        Returns:
            Dict with 'scalars', 'vectors', 'tensors' (weighted combination)
        """
        all_scalars = []
        all_vectors = []
        all_tensors = []
        
        for features in node_features_per_layer:
            out = self.head(features, x_mask)
            all_scalars.append(out['scalars'])
            all_vectors.append(out['vectors'])
            all_tensors.append(out['tensors'])
        
        w_s = jax.nn.softmax(self.layer_weights_scalars.value)
        w_v = jax.nn.softmax(self.layer_weights_vectors.value)
        w_t = jax.nn.softmax(self.layer_weights_tensors.value)
        
        scalars = sum(w * s for w, s in zip(w_s, all_scalars))
        vectors = sum(w * v for w, v in zip(w_v, all_vectors))
        tensors = sum(w * t for w, t in zip(w_t, all_tensors))
        
        return {
            'scalars': scalars,
            'vectors': vectors,
            'tensors': tensors,
        }
