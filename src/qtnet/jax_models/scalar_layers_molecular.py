"""
Molecular (graph-level) output head and per-layer readout for scalar models.

Drop these two classes into scalar_layers.py alongside ScalarAtomicHead
and PerLayerReadout.

Predicted property
------------------
A single scalar per graph (e.g. total energy, HOMO-LUMO gap, ionisation
potential). This is intentionally different from the per-atom head, which
outputs N, LI, dipole components and quadrupole components.

Pooling strategy: learned attention pooling with segment-wise softmax.
  - An attention gate maps each node's normalised features to a scalar logit.
  - Logits are normalised per graph (segment softmax) so the weights across
    each molecule sum to 1, regardless of molecule size or batch padding.
  - The weighted node-feature sum is passed through a backbone MLP → single
    linear projection → one scalar per graph.

Batching: pass graph_idx (shape N_total,) and num_graphs so segment
operations assign each node to the correct molecule. For single-molecule
inference leave graph_idx=None; all nodes are pooled together.
"""

from typing import Optional

import jax.numpy as jnp
import jax
import flax.nnx as nnx


class ScalarMolecularHead(nnx.Module):
    """
    Output head for molecular (graph-level) scalar models.

    Replaces the per-atom ScalarAtomicHead with an attention-pooling step
    that collapses N node features into a single graph-level feature vector,
    then projects it to **one scalar** per graph.

    Attention weights are computed from normalised node features and are
    normalised per graph via a segment-wise softmax; padding nodes
    (x_mask == False) are excluded from the pool.

    Output per graph:
        scalar : (G, 1) – single molecular property

    Args:
        num_scalars_in : Width of the incoming node feature vectors.
        hidden_dim     : Hidden width used in the attention gate and backbone.
        rngs           : Flax RNG container.
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

        # Pre-norm (mirrors ScalarAtomicHead)
        self.norm = nnx.LayerNorm(num_features=num_scalars_in, rngs=rngs)

        # Attention gate: node features → scalar logit.
        # Two-layer MLP; no final activation so logits can be freely shifted
        # before the segment softmax.
        self.attention_gate = nnx.Sequential(
            nnx.Linear(num_scalars_in, hidden_dim, rngs=rngs),
            nnx.silu,
            nnx.Linear(hidden_dim, 1, rngs=rngs),
        )

        # Backbone operating on the pooled graph-level feature vector
        self.backbone = nnx.Sequential(
            nnx.Linear(num_scalars_in, hidden_dim, rngs=rngs),
            nnx.silu,
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
            nnx.silu,
        )

        # Single output: one scalar per graph
        self.head = nnx.Linear(hidden_dim, 1, rngs=rngs)

    def __call__(
        self,
        x: jnp.ndarray,
        x_mask: Optional[jnp.ndarray] = None,
        graph_idx: Optional[jnp.ndarray] = None,
        num_graphs: int = 1,
    ) -> jnp.ndarray:
        """
        Args:
            x          : Node features, shape (N_total, num_scalars_in).
            x_mask     : Boolean validity mask, shape (N_total,).
                         Padding nodes should be False; excluded from pool.
            graph_idx  : Integer graph index for every node, shape (N_total,).
                         If None all nodes are treated as one graph.
            num_graphs : Number of graphs in the batch (ignored when
                         graph_idx is None).

        Returns:
            Scalar predictions, shape (G, 1) where G = num_graphs (or 1).
        """
        h = self.norm(x)                                    # (N, d)

        # ---- Attention logits ------------------------------------------------
        attn_logits = self.attention_gate(h).squeeze(-1)    # (N,)

        # ---- Pooling ---------------------------------------------------------
        if graph_idx is None:
            # Single-graph path: plain masked softmax then weighted sum.
            if x_mask is not None:
                attn_logits = jnp.where(x_mask, attn_logits, -1e9)
            attn_weights = jax.nn.softmax(attn_logits)             # (N,)
            graph_features = jnp.sum(
                attn_weights[:, None] * h, axis=0, keepdims=True   # (1, d)
            )
        else:
            # Batched path: numerically stable segment-wise softmax.
            if x_mask is not None:
                attn_logits = jnp.where(x_mask, attn_logits, -1e9)

            # Subtract per-graph max for stability
            max_per_graph = jax.ops.segment_max(
                attn_logits, graph_idx, num_graphs
            )                                                        # (G,)
            shifted = attn_logits - max_per_graph[graph_idx]         # (N,)

            exp_shifted = jnp.exp(shifted)
            if x_mask is not None:
                exp_shifted = jnp.where(x_mask, exp_shifted, 0.0)

            sum_exp = jax.ops.segment_sum(
                exp_shifted, graph_idx, num_graphs
            )                                                        # (G,)
            attn_weights = exp_shifted / (sum_exp[graph_idx] + 1e-8)  # (N,)

            graph_features = jax.ops.segment_sum(
                attn_weights[:, None] * h, graph_idx, num_graphs    # (G, d)
            )

        # ---- Backbone + single output head ----------------------------------
        g = self.backbone(graph_features)   # (G, hidden_dim)
        return self.head(g)                 # (G, 1)


class MolecularPerLayerReadout(nnx.Module):
    """Per-layer molecular readout with learned weighted combination.

    Mirrors PerLayerReadout, but wraps ScalarMolecularHead. The shared head
    is applied to each layer's node features to produce a graph-level scalar;
    those scalars are then combined via a single learned softmax-weighted sum.

    Usage::

        readout = MolecularPerLayerReadout(num_scalars_in, hidden_dim,
                                           num_layers, rngs)
        output = readout.read_and_combine(
            layer_features, x_mask, graph_idx, num_graphs
        )
        # output shape: (G, 1)

    Args:
        num_scalars_in : Width of incoming node feature vectors.
        hidden_dim     : Hidden width passed through to ScalarMolecularHead.
        num_layers     : Total number of layers (including layer 0).
        rngs           : Flax RNG container.
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

        self.head = ScalarMolecularHead(
            num_scalars_in=num_scalars_in,
            hidden_dim=hidden_dim,
            rngs=rngs,
        )

        # Single set of layer weights (one scalar output → one weight vector).
        # Initialised to 0 so softmax starts uniform.
        self.layer_weights = nnx.Param(jnp.zeros(num_layers))

    def read_and_combine(
        self,
        node_features_per_layer: list,
        x_mask: Optional[jnp.ndarray] = None,
        graph_idx: Optional[jnp.ndarray] = None,
        num_graphs: int = 1,
    ) -> jnp.ndarray:
        """Compute per-layer graph-level scalar predictions and combine.

        Args:
            node_features_per_layer : List of (N, d) arrays, one per layer
                                      (including layer 0).
            x_mask                  : (N,) boolean node validity mask.
            graph_idx               : (N,) integer graph index per node.
                                      Pass None for single-graph inference.
            num_graphs              : Number of graphs in the batch.

        Returns:
            (G, 1) weighted combination of per-layer scalar predictions.
        """
        w = jax.nn.softmax(self.layer_weights.value)   # (num_layers,)

        return sum(
            w[i] * self.head(features, x_mask, graph_idx, num_graphs)
            for i, features in enumerate(node_features_per_layer)
        )  # (G, 1)


# =============================================================================
# Multi-property readout (num_outputs predictions per graph)
# =============================================================================

class ScalarMolecularMultiHead(nnx.Module):
    """
    Output head for molecular models predicting multiple properties.

    Same attention-pooling backbone as ScalarMolecularHead, but projects
    to `num_outputs` scalars per graph instead of 1. The attention gate
    (shared weights) produces a single pooled graph-feature vector;
    `num_outputs` independent linear heads project it to each property.

    Args:
        num_scalars_in : Width of incoming node feature vectors.
        num_outputs    : Number of molecular properties to predict.
        hidden_dim     : Hidden width for attention gate and backbone.
        rngs           : Flax RNG container.
    """

    def __init__(
        self,
        num_scalars_in: int,
        num_outputs: int = 4,
        hidden_dim: int = 64,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        if rngs is None:
            rngs = nnx.Rngs(0)

        self.num_outputs = num_outputs

        self.norm = nnx.LayerNorm(num_features=num_scalars_in, rngs=rngs)

        self.attention_gate = nnx.Sequential(
            nnx.Linear(num_scalars_in, hidden_dim, rngs=rngs),
            nnx.silu,
            nnx.Linear(hidden_dim, 1, rngs=rngs),
        )

        self.backbone = nnx.Sequential(
            nnx.Linear(num_scalars_in, hidden_dim, rngs=rngs),
            nnx.silu,
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
            nnx.silu,
        )

        self.head = nnx.Linear(hidden_dim, num_outputs, rngs=rngs)

    def __call__(
        self,
        x: jnp.ndarray,
        x_mask: Optional[jnp.ndarray] = None,
        graph_idx: Optional[jnp.ndarray] = None,
        num_graphs: int = 1,
    ) -> jnp.ndarray:
        """
        Returns:
            (G, num_outputs) predictions.
        """
        h = self.norm(x)
        attn_logits = self.attention_gate(h).squeeze(-1)    # (N,)

        if graph_idx is None:
            if x_mask is not None:
                attn_logits = jnp.where(x_mask, attn_logits, -1e9)
            attn_weights = jax.nn.softmax(attn_logits)
            graph_features = jnp.sum(attn_weights[:, None] * h, axis=0, keepdims=True)
        else:
            if x_mask is not None:
                attn_logits = jnp.where(x_mask, attn_logits, -1e9)
            max_per_graph = jax.ops.segment_max(attn_logits, graph_idx, num_graphs)
            shifted = attn_logits - max_per_graph[graph_idx]
            exp_shifted = jnp.exp(shifted)
            if x_mask is not None:
                exp_shifted = jnp.where(x_mask, exp_shifted, 0.0)
            sum_exp = jax.ops.segment_sum(exp_shifted, graph_idx, num_graphs)
            attn_weights = exp_shifted / (sum_exp[graph_idx] + 1e-8)
            graph_features = jax.ops.segment_sum(
                attn_weights[:, None] * h, graph_idx, num_graphs
            )

        g = self.backbone(graph_features)
        return self.head(g)   # (G, num_outputs)


class MolecularMultiPropertyReadout(nnx.Module):
    """Per-layer molecular readout for multi-property models.

    Wraps ScalarMolecularMultiHead and maintains per-layer softmax weights.
    Each layer's graph-level predictions are combined via a learned weighted sum.

    Args:
        num_scalars_in : Width of incoming node feature vectors.
        num_outputs    : Number of molecular properties to predict.
        hidden_dim     : Hidden width passed through to ScalarMolecularMultiHead.
        num_layers     : Total number of layers (including layer 0).
        rngs           : Flax RNG container.
    """

    def __init__(
        self,
        num_scalars_in: int,
        num_outputs: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 3,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        if rngs is None:
            rngs = nnx.Rngs(0)
        self.num_layers = num_layers

        self.head = ScalarMolecularMultiHead(
            num_scalars_in=num_scalars_in,
            num_outputs=num_outputs,
            hidden_dim=hidden_dim,
            rngs=rngs,
        )

        self.layer_weights = nnx.Param(jnp.zeros(num_layers))

    def read_and_combine(
        self,
        node_features_per_layer: list,
        x_mask: Optional[jnp.ndarray] = None,
        graph_idx: Optional[jnp.ndarray] = None,
        num_graphs: int = 1,
    ) -> jnp.ndarray:
        """
        Returns:
            (G, num_outputs) weighted combination of per-layer predictions.
        """
        w = jax.nn.softmax(self.layer_weights.value)
        return sum(
            w[i] * self.head(features, x_mask, graph_idx, num_graphs)
            for i, features in enumerate(node_features_per_layer)
        )  # (G, num_outputs)


# =============================================================================
# No-geometry message-passing layers (topology only, no r_ij / G filter)
# =============================================================================

class NoGeoNodeEncoder(nnx.Module):
    """Layer-0 node encoder without geometric filtering.

    Replaces ScalarNodeEncoder for topology-only models.
    Aggregates neighbour embeddings, concatenates with self embedding, MLP.

    h_i = MLP(emb_i, sum_j emb_j)  for j in up_neighbours(i)
    """

    def __init__(
        self,
        num_scalar_out: int,
        embedding_dim: int,
        hidden_dim: int = 64,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        if rngs is None:
            rngs = nnx.Rngs(0)

        self.mlp = nnx.Sequential(
            nnx.Linear(2 * embedding_dim, hidden_dim, rngs=rngs),
            nnx.silu,
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
            nnx.silu,
            nnx.Linear(hidden_dim, num_scalar_out, rngs=rngs),
        )

    def __call__(
        self,
        chem_embeddings: jnp.ndarray,
        x_mask: jnp.ndarray,
        up_senders: jnp.ndarray,
        up_receivers: jnp.ndarray,
        up_mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        num_nodes = chem_embeddings.shape[0]

        emb_s = chem_embeddings[up_senders]
        if up_mask is not None:
            emb_s = jnp.where(up_mask[:, None], emb_s, 0.0)

        neighbour_sum = jax.ops.segment_sum(emb_s, up_receivers, num_nodes)
        mlp_input = jnp.concatenate([chem_embeddings, neighbour_sum], axis=-1)
        node_features = self.mlp(mlp_input)

        if x_mask is not None:
            node_features = jnp.where(x_mask[:, None], node_features, 0.0)
        return node_features


class NoGeoEdgeEncoder(nnx.Module):
    """Layer-0 edge encoder without geometric filtering.

    Replaces ScalarEdgeEncoder for topology-only models.
    Aggregates endpoint node features (inner MLP → segment_sum → outer MLP),
    same structure as ScalarEdgeEncoder without the gyration-tensor gate.

    h_edge = OutMLP( agg_norm( sum_k∈∂(edge) InnerMLP(h_k) ) )
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

        self.inner_mlp = nnx.Sequential(
            nnx.Linear(num_node_scalars, hidden_dim, rngs=rngs),
            nnx.silu,
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
        )
        self.agg_norm = nnx.LayerNorm(num_features=hidden_dim, rngs=rngs)
        self.outer_mlp = nnx.Sequential(
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
            nnx.silu,
            nnx.Linear(hidden_dim, num_edge_scalars, rngs=rngs),
        )

    def __call__(
        self,
        boundary_x: jnp.ndarray,
        boundary_senders: jnp.ndarray,
        boundary_receivers: jnp.ndarray,
        boundary_mask: Optional[jnp.ndarray] = None,
        x_mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        num_edges = x_mask.shape[0] if x_mask is not None else int(boundary_receivers.max()) + 1

        node_contrib = self.inner_mlp(boundary_x[boundary_senders])
        if boundary_mask is not None:
            node_contrib = jnp.where(boundary_mask[:, None], node_contrib, 0.0)

        edge_hidden = jax.ops.segment_sum(node_contrib, boundary_receivers, num_edges)
        edge_hidden = self.agg_norm(edge_hidden)
        edge_features = self.outer_mlp(edge_hidden)

        if x_mask is not None:
            edge_features = jnp.where(x_mask[:, None], edge_features, 0.0)
        return edge_features


class NoGeoNodeMessages(nnx.Module):
    """Node-to-node messages without geometric filtering (layers > 0).

    Replaces ScalarNodeUpMessages. Optionally includes intermediary edge
    features (edge-enriched messages). No r_ij gate.

    message_ij = MLP(h_r, h_s [, e_ij])
    """

    def __init__(
        self,
        num_scalars: int,
        hidden_dim: int = 64,
        num_edge_scalars: int = 0,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        if rngs is None:
            rngs = nnx.Rngs(0)
        self.num_edge_scalars = num_edge_scalars

        mlp_input_dim = 2 * num_scalars + num_edge_scalars
        self.message_mlp = nnx.Sequential(
            nnx.Linear(mlp_input_dim, hidden_dim, rngs=rngs),
            nnx.silu,
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
            nnx.silu,
            nnx.Linear(hidden_dim, num_scalars, rngs=rngs),
        )

    def __call__(
        self,
        x: jnp.ndarray,
        x_mask: jnp.ndarray,
        up_senders: jnp.ndarray,
        up_receivers: jnp.ndarray,
        up_mask: Optional[jnp.ndarray] = None,
        edge_features: Optional[jnp.ndarray] = None,
        up_intermediaries: Optional[jnp.ndarray] = None,
        **kwargs,
    ) -> jnp.ndarray:
        num_nodes = x.shape[0]

        h_s = x[up_senders]
        h_r = x[up_receivers]

        if self.num_edge_scalars > 0 and edge_features is not None and up_intermediaries is not None:
            e_ij = edge_features[up_intermediaries]
            mlp_input = jnp.concatenate([h_r, h_s, e_ij], axis=-1)
        else:
            mlp_input = jnp.concatenate([h_r, h_s], axis=-1)

        messages = self.message_mlp(mlp_input)

        if up_mask is not None:
            messages = jnp.where(up_mask[:, None], messages, 0.0)

        update = jax.ops.segment_sum(messages, up_receivers, num_nodes)

        if x_mask is not None:
            update = jnp.where(x_mask[:, None], update, 0.0)
        return update


class NoGeoEdgeBoundaryMessages(nnx.Module):
    """Node-to-edge boundary messages without geometric filtering (layers > 0).

    Replaces ScalarEdgeBoundaryMessages. Aggregates node features at edge
    endpoints. No gyration-tensor gate.

    h_edge_update = OutMLP( agg_norm( sum_k∈∂(edge) MLP(h_k) ) )
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

    def __call__(
        self,
        x: jnp.ndarray,
        x_mask: jnp.ndarray,
        boundary_x: jnp.ndarray,
        boundary_senders: jnp.ndarray,
        boundary_receivers: jnp.ndarray,
        boundary_mask: Optional[jnp.ndarray] = None,
        **kwargs,
    ) -> jnp.ndarray:
        num_edges = x.shape[0]

        node_contrib = self.message_mlp(boundary_x[boundary_senders])
        if boundary_mask is not None:
            node_contrib = jnp.where(boundary_mask[:, None], node_contrib, 0.0)

        edge_hidden = jax.ops.segment_sum(node_contrib, boundary_receivers, num_edges)
        edge_hidden = self.agg_norm(edge_hidden)
        update = self.output_mlp(edge_hidden)

        if x_mask is not None:
            update = jnp.where(x_mask[:, None], update, 0.0)
        return update


class UpdateFromCochains(nnx.Module):
    """Gated residual update of graph-level (complex) features from cochain pooling.

    Analogous to ScalarDualChannelNodeUpdate but at the graph level: each cochain
    (nodes, edges, optionally rings) is pooled via segment_sum into a per-graph
    vector; each pooled vector independently gates a contribution to the complex
    feature with a sigmoid gate that is conditioned on both the current complex
    features and the pooled cochain.

    The update is purely FROM cochains TO complex (unidirectional), so the
    cochain message-passing graph remains unchanged.

    Args:
        d_cx       : Dimension of the complex feature vector.
        d_node     : Dimension of node features.
        d_edge     : Dimension of edge features.
        d_ring     : Dimension of ring features; None disables the ring channel.
        hidden_dim : Internal MLP / gate width.
        rngs       : Flax RNG container.
    """

    def __init__(
        self,
        d_cx: int,
        d_node: int,
        d_edge: int,
        d_ring: Optional[int] = None,
        hidden_dim: int = 64,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        if rngs is None:
            rngs = nnx.Rngs(0)
        self.has_rings = d_ring is not None

        self.norm = nnx.LayerNorm(num_features=d_cx, rngs=rngs)

        # Project each cochain's features to a common hidden_dim before pooling
        self.node_proj = nnx.Linear(d_node, hidden_dim, rngs=rngs)
        self.edge_proj = nnx.Linear(d_edge, hidden_dim, rngs=rngs)
        if self.has_rings:
            self.ring_proj = nnx.Linear(d_ring, hidden_dim, rngs=rngs)

        # Gate: sigmoid(Linear(concat(cx_norm, pooled_cochain))) → (G, d_cx)
        self.gate_node   = nnx.Linear(d_cx + hidden_dim, d_cx, rngs=rngs)
        self.gate_edge   = nnx.Linear(d_cx + hidden_dim, d_cx, rngs=rngs)
        if self.has_rings:
            self.gate_ring = nnx.Linear(d_cx + hidden_dim, d_cx, rngs=rngs)

        # Update value: Linear(pooled_cochain) → (G, d_cx)
        self.update_node   = nnx.Linear(hidden_dim, d_cx, rngs=rngs)
        self.update_edge   = nnx.Linear(hidden_dim, d_cx, rngs=rngs)
        if self.has_rings:
            self.update_ring = nnx.Linear(hidden_dim, d_cx, rngs=rngs)

    def __call__(
        self,
        cx: jnp.ndarray,
        node_features: jnp.ndarray,
        edge_features: jnp.ndarray,
        ring_features: Optional[jnp.ndarray] = None,
        node_graph_idx: Optional[jnp.ndarray] = None,
        edge_graph_idx: Optional[jnp.ndarray] = None,
        ring_graph_idx: Optional[jnp.ndarray] = None,
        node_mask: Optional[jnp.ndarray] = None,
        edge_mask: Optional[jnp.ndarray] = None,
        ring_mask: Optional[jnp.ndarray] = None,
        num_graphs: int = 1,
    ) -> jnp.ndarray:
        cx_norm = self.norm(cx)                                    # (G+1, d_cx)

        # Pool nodes → (G+1, hidden_dim).  cx has shape (G+1, d_cx) to include
        # an OOB "padding complex" row (index G), mirroring how cochain arrays
        # have mc+1 rows.  Padding cells own complex G, so their (zeroed) features
        # accumulate into that OOB row and never corrupt real complex features.
        h_n = jax.nn.silu(self.node_proj(node_features))
        if node_mask is not None:
            h_n = jnp.where(node_mask[:, None], h_n, 0.0)
        g_n = (jax.ops.segment_sum(h_n, node_graph_idx, num_graphs + 1)
               if node_graph_idx is not None else h_n.sum(axis=0, keepdims=True))

        # Pool edges → (G+1, hidden_dim)
        h_e = jax.nn.silu(self.edge_proj(edge_features))
        if edge_mask is not None:
            h_e = jnp.where(edge_mask[:, None], h_e, 0.0)
        g_e = (jax.ops.segment_sum(h_e, edge_graph_idx, num_graphs + 1)
               if edge_graph_idx is not None else h_e.sum(axis=0, keepdims=True))

        # Gated updates from nodes and edges
        gate_n = jax.nn.sigmoid(self.gate_node(jnp.concatenate([cx_norm, g_n], axis=-1)))
        gate_e = jax.nn.sigmoid(self.gate_edge(jnp.concatenate([cx_norm, g_e], axis=-1)))
        update = gate_n * self.update_node(g_n) + gate_e * self.update_edge(g_e)

        # Optional ring channel
        if self.has_rings and ring_features is not None:
            h_r = jax.nn.silu(self.ring_proj(ring_features))
            if ring_mask is not None:
                h_r = jnp.where(ring_mask[:, None], h_r, 0.0)
            g_r = (jax.ops.segment_sum(h_r, ring_graph_idx, num_graphs + 1)
                   if ring_graph_idx is not None else h_r.sum(axis=0, keepdims=True))
            gate_r = jax.nn.sigmoid(self.gate_ring(jnp.concatenate([cx_norm, g_r], axis=-1)))
            update = update + gate_r * self.update_ring(g_r)

        return cx + update


class ComplexMultiPropertyReadout(nnx.Module):
    """Per-layer readout directly from graph-level (complex) features.

    Because ``complex_features`` is already ``(G, d_cx)`` — no attention pooling
    is needed.  A shared backbone + single head are applied to each layer's
    complex features; results are combined via learned softmax-weighted sum.

    Args:
        d_cx        : Dimension of complex feature vectors.
        num_outputs : Number of molecular properties to predict.
        hidden_dim  : Backbone MLP hidden width.
        num_layers  : Total number of layers (sets the number of layer weights).
        rngs        : Flax RNG container.
    """

    def __init__(
        self,
        d_cx: int,
        num_outputs: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 3,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        if rngs is None:
            rngs = nnx.Rngs(0)
        self.num_layers = num_layers

        self.norm = nnx.LayerNorm(num_features=d_cx, rngs=rngs)
        self.backbone = nnx.Sequential(
            nnx.Linear(d_cx, hidden_dim, rngs=rngs),
            nnx.silu,
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
            nnx.silu,
        )
        self.head = nnx.Linear(hidden_dim, num_outputs, rngs=rngs)

        # One scalar weight per layer; init to 0 so softmax starts uniform
        self.layer_weights = nnx.Param(jnp.zeros(num_layers))

    def read_and_combine(
        self,
        complex_features_per_layer: list,
        **ignored,
    ) -> jnp.ndarray:
        """
        Args:
            complex_features_per_layer : list of (G, d_cx), one per layer.
        Returns:
            (G, num_outputs) weighted combination across layers.
        """
        w = jax.nn.softmax(self.layer_weights.value)              # (num_layers,)
        return sum(
            w[i] * self.head(self.backbone(self.norm(cx)))
            for i, cx in enumerate(complex_features_per_layer)
        )                                                          # (G, num_outputs)


# =============================================================================
# No-geometry message-passing layers (topology only, no r_ij / G filter)
# =============================================================================

class NoGeoEdgeDownMessages(nnx.Module):
    """Edge-to-edge down messages via shared node, without geometric filtering.

    Replaces ScalarEdgeDownMessages. Uses concatenated [e_receiver, e_sender,
    n_shared_node] → MLP. No ΔG gate.
    """

    def __init__(
        self,
        num_edge_scalars: int,
        num_node_scalars: int,
        hidden_dim: int = 64,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        if rngs is None:
            rngs = nnx.Rngs(0)

        self.message_mlp = nnx.Sequential(
            nnx.Linear(2 * num_edge_scalars + num_node_scalars, hidden_dim, rngs=rngs),
            nnx.silu,
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
            nnx.silu,
            nnx.Linear(hidden_dim, num_edge_scalars, rngs=rngs),
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
        **kwargs,
    ) -> jnp.ndarray:
        num_edges = x.shape[0]

        e_s = x[down_senders]
        e_r = x[down_receivers]
        n_k = node_features[down_intermediaries]

        messages = self.message_mlp(jnp.concatenate([e_r, e_s, n_k], axis=-1))

        if down_mask is not None:
            messages = jnp.where(down_mask[:, None], messages, 0.0)

        update = jax.ops.segment_sum(messages, down_receivers, num_edges)

        if x_mask is not None:
            update = jnp.where(x_mask[:, None], update, 0.0)
        return update
