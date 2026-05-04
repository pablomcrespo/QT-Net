"""
Model Zoo

Five scalar models for molecular property prediction:
1. DeepSets: No geometry, sum-aggregation baseline
2. ScalarBaseline: Nodes only with r_ij geometric gates
3. ScalarBaselineEdges: Nodes + fresh edge encoding (G + ΔG gates) + edge→node coboundary
4. ScalarGNN: Nodes + persistent edges with geometric gates (G + ΔG gates)
5. ScalarTPaiNN: Nodes + persistent edges + bags-of-bonds with dual-channel node update

Design principles:
- Pre-norm: LayerNorm before sub-layers (not after residuals)
- Gated residuals: x + sigmoid_gate(x) * update
- tanh geometric gates: sign-sensitive, bounded, for directional filtering
- ChemicalReminder at end of each cycle (before readout)
- No node FFN (ChemReminder subsumes it)
- Per-layer readout with learned softmax-weighted sum (all models)

All models predict per-atom properties:
- N (electron count): scalar
- LI (localization index): scalar
- Dipole moment: 3D vector (as scalar components)
- Quadrupole moment: 5-component traceless symmetric tensor (as scalar components)
"""

from typing import Optional, Dict, Tuple

import jax.numpy as jnp
import jax
import flax.nnx as nnx

from qtnet.jax_models.scalar_layers import (
    ChemicalEmbedding,
    ScalarChemicalReminder,
    ScalarNodeUpMessages,
    ScalarGatedResidual,
    ScalarEdgeBoundaryMessages,
    ScalarEdgeDownMessages,
    ScalarEdgeFFN,
    ScalarNodeCoboundaryMessages,
    ScalarBagEncoder,
    ScalarEdgeCoboundaryMessages,
    ScalarNodeBoundaryMessages,
    ScalarDualChannelNodeUpdate,
    ScalarNodeEncoder,
    ScalarEdgeEncoder,
    ScalarAtomicHead,
    PerLayerReadout,
)
from qtnet.jax_models.representations import ComplexBatch


'''
# =============================================================================
# DEEPSETS BASELINE
# =============================================================================

class DeepSetsLayer(nnx.Module):
    """A single DeepSets layer (for layers > 0).
    
    No geometric information. Message passing cycle:
        sum neighbor features → MLP → gated residual → ChemicalReminder
    """
    
    def __init__(
        self,
        num_node_scalars: int,
        embedding_dim: int,
        hidden_dim: int = 64,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        if rngs is None:
            rngs = nnx.Rngs(0)
        
        # Message MLP: takes [h_i, Σ_j h_j]
        self.message_mlp = nnx.Sequential(
            nnx.Linear(2 * num_node_scalars, hidden_dim, rngs=rngs),
            nnx.silu,
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
            nnx.silu,
            nnx.Linear(hidden_dim, num_node_scalars, rngs=rngs),
        )
        self.message_block = ScalarGatedResidual(num_node_scalars, hidden_dim, rngs=rngs)
        
        # Chemical reminder (end of cycle)
        self.chemical_reminder = ScalarChemicalReminder(
            num_scalars=num_node_scalars,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            rngs=rngs,
        )
    
    def __call__(
        self,
        node_features: jnp.ndarray,
        chem_embeddings: jnp.ndarray,
        node_adj: Dict,
    ) -> jnp.ndarray:
        
        num_nodes = node_features.shape[0]
        
        # Sum neighbor features (no geometry)
        neighbor_features = node_features[node_adj['up_senders']]
        if node_adj.get('up_mask') is not None:
            neighbor_features = jnp.where(node_adj['up_mask'][:, None], neighbor_features, 0.0)
        summed = jax.ops.segment_sum(neighbor_features, node_adj['up_receivers'], num_nodes)
        
        # MLP on [self, summed_neighbors] → gated residual
        normed = self.message_block.pre_norm(node_features)
        mlp_input = jnp.concatenate([normed, summed], axis=-1)
        update = self.message_mlp(mlp_input)
        node_features = self.message_block.residual(node_features, update, node_adj['x_mask'])
        
        # Chemical reminder (end of cycle)
        node_features = self.chemical_reminder(
            x=node_features,
            chem_embeddings=chem_embeddings,
            x_mask=node_adj['x_mask'],
        )
        
        return node_features


class DeepSets(nnx.Module):
    """
    DeepSets baseline model.
    
    No geometric information used at all. Purely based on chemical identity
    and neighbor aggregation (sum). A minimal baseline to quantify how much
    geometric information contributes to prediction quality.
    
    Layer 0: ChemicalEmbedding → sum neighbor embeddings → MLP → norm
    Layers 1+: DeepSetsLayer (sum → MLP → gated residual → ChemReminder)
    """
    
    def __init__(
        self,
        num_species: int,
        num_node_scalars: int = 32,
        embedding_dim: int = 32,
        hidden_dim: int = 64,
        num_layers: int = 3,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        if rngs is None:
            rngs = nnx.Rngs(0)
        self.rngs = rngs
        
        self.num_node_scalars = num_node_scalars
        self.num_layers = num_layers
        
        # Chemical embedding
        self.chemical_embedding = ChemicalEmbedding(
            num_species=num_species,
            embedding_dim=embedding_dim,
            rngs=rngs,
        )
        
        # Layer 0: Initial encoding MLP from [emb_i, Σ_j emb_j]
        self.init_mlp = nnx.Sequential(
            nnx.Linear(2 * embedding_dim, hidden_dim, rngs=rngs),
            nnx.silu,
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
            nnx.silu,
            nnx.Linear(hidden_dim, num_node_scalars, rngs=rngs),
        )
        self.init_norm = nnx.LayerNorm(num_features=num_node_scalars, rngs=rngs)
        
        # Layers 1+
        self.layers = [
            DeepSetsLayer(
                num_node_scalars=num_node_scalars,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                rngs=rngs,
            )
            for _ in range(num_layers - 1)
        ]
        
        # Per-layer readout
        self.readout = PerLayerReadout(
            num_scalars_in=num_node_scalars,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            rngs=rngs,
        )
    
    def __call__(
        self,
        complex_batch: ComplexBatch,
    ) -> Dict[str, jnp.ndarray]:
        
        node_batch = complex_batch.cochain_batches[0]
        
        node_adj = {
            'up_senders': node_batch.up_senders,
            'up_receivers': node_batch.up_receivers,
            'up_mask': node_batch.up_mask,
            'x_mask': node_batch.x_mask,
        }
        
        node_static = node_batch.static
        
        if node_static is None or 'Z' not in node_static:
            raise ValueError("node_static must contain 'Z'")
        species_indices = node_static['Z']
        
        # Chemical embeddings
        chem_embeddings = self.chemical_embedding(species_indices)
        
        # Layer 0: sum neighbor embeddings → MLP
        num_nodes = chem_embeddings.shape[0]
        neighbor_embs = chem_embeddings[node_adj['up_senders']]
        if node_adj['up_mask'] is not None:
            neighbor_embs = jnp.where(node_adj['up_mask'][:, None], neighbor_embs, 0.0)
        summed_embs = jax.ops.segment_sum(neighbor_embs, node_adj['up_receivers'], num_nodes)
        
        mlp_input = jnp.concatenate([chem_embeddings, summed_embs], axis=-1)
        node_features = self.init_mlp(mlp_input)
        node_features = self.init_norm(node_features)
        if node_adj['x_mask'] is not None:
            node_features = jnp.where(node_adj['x_mask'][:, None], node_features, 0.0)
        
        # Collect per-layer node features
        layer_features = [node_features]
        
        # Layers 1+
        for layer in self.layers:
            node_features = layer(
                node_features=node_features,
                chem_embeddings=chem_embeddings,
                node_adj=node_adj,
            )
            layer_features.append(node_features)
        
        # Per-layer readout
        output = self.readout.read_and_combine(layer_features, node_adj['x_mask'])
        
        return {
            'node_features': node_features,
            'scalars': output['scalars'],
            'vectors': output['vectors'],
            'tensors': output['tensors'],
            'x_mask': node_adj['x_mask'],
        }

'''

# =============================================================================
# SCALAR BASELINE (nodes only, with geometry)
# =============================================================================

class ScalarBaselineLayer(nnx.Module):
    """A single ScalarBaseline layer (for layers > 0).
    
    Uses ScalarNodeUpMessages (with tanh r_ij gate) but no edge features.
    
    Message passing cycle:
        NodeUpMessages (pre-norm → gated residual) → ChemicalReminder
    """
    
    def __init__(
        self,
        num_node_scalars: int,
        embedding_dim: int,
        hidden_dim: int = 64,
        geometric_filter_dim: Optional[int] = None,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        if rngs is None:
            rngs = nnx.Rngs(0)
        
        # Node up messages (with geometric r_ij gate)
        self.node_up = ScalarNodeUpMessages(
            num_scalars=num_node_scalars,
            hidden_dim=hidden_dim,
            geometric_filter_dim=geometric_filter_dim,
            rngs=rngs,
        )
        self.node_up_block = ScalarGatedResidual(num_node_scalars, hidden_dim, rngs=rngs)
        
        # Chemical reminder (end of cycle)
        self.chemical_reminder = ScalarChemicalReminder(
            num_scalars=num_node_scalars,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            rngs=rngs,
        )
    
    def __call__(
        self,
        node_features: jnp.ndarray,
        chem_embeddings: jnp.ndarray,
        node_adj: Dict,
        node_static: Optional[Dict] = None,
    ) -> jnp.ndarray:
        
        # Node up messages (with geometry) → pre-norm → gated residual
        normed = self.node_up_block.pre_norm(node_features)
        node_up_update = self.node_up(
            x=normed,
            x_mask=node_adj['x_mask'],
            up_senders=node_adj['up_senders'],
            up_receivers=node_adj['up_receivers'],
            up_mask=node_adj.get('up_mask'),
            static=node_static,
        )
        node_features = self.node_up_block.residual(node_features, node_up_update, node_adj['x_mask'])
        
        # Chemical reminder (end of cycle)
        node_features = self.chemical_reminder(
            x=node_features,
            chem_embeddings=chem_embeddings,
            x_mask=node_adj['x_mask'],
        )
        
        return node_features


class ScalarBaseline(nnx.Module):
    """
    Scalar Baseline model (nodes only, with geometry).
    
    Uses ScalarNodeEncoder and ScalarNodeUpMessages with tanh r_ij gates,
    but no edge features are stored or passed.
    
    Layer 0: ChemicalEmbedding → ScalarNodeEncoder → norm
    Layers 1+: ScalarBaselineLayer (NodeUp → gated residual → ChemReminder)
    """
    
    def __init__(
        self,
        num_species: int,
        num_node_scalars: int = 32,
        embedding_dim: int = 32,
        hidden_dim: int = 64,
        geometric_filter_dim: Optional[int] = None,
        num_layers: int = 3,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        if rngs is None:
            rngs = nnx.Rngs(0)
        self.rngs = rngs
        
        self.num_node_scalars = num_node_scalars
        self.num_layers = num_layers
        
        # Chemical embedding
        self.chemical_embedding = ChemicalEmbedding(
            num_species=num_species,
            embedding_dim=embedding_dim,
            rngs=rngs,
        )
        
        # Node encoder (with geometric r_ij gate)
        self.node_encoder = ScalarNodeEncoder(
            num_scalar_out=num_node_scalars,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            geometric_filter_dim=geometric_filter_dim,
            rngs=rngs,
        )
        self.node_encoder_norm = nnx.LayerNorm(num_features=num_node_scalars, rngs=rngs)
        
        # Layers 1+
        self.layers = [
            ScalarBaselineLayer(
                num_node_scalars=num_node_scalars,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                geometric_filter_dim=geometric_filter_dim,
                rngs=rngs,
            )
            for _ in range(num_layers - 1)
        ]
        
        # Per-layer readout
        self.readout = PerLayerReadout(
            num_scalars_in=num_node_scalars,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            rngs=rngs,
        )
    
    def __call__(
        self,
        complex_batch: ComplexBatch,
    ) -> Dict[str, jnp.ndarray]:
        
        node_batch = complex_batch.cochain_batches[0]
        
        node_adj = {
            'up_senders': node_batch.up_senders,
            'up_receivers': node_batch.up_receivers,
            'up_mask': node_batch.up_mask,
            'x_mask': node_batch.x_mask,
        }
        
        node_static = node_batch.static
        
        if node_static is None or 'Z' not in node_static:
            raise ValueError("node_static must contain 'Z'")
        species_indices = node_static['Z']
        
        # Chemical embeddings
        chem_embeddings = self.chemical_embedding(species_indices)
        
        # Node encoder (with geometry)
        node_features = self.node_encoder(
            chem_embeddings=chem_embeddings,
            x_mask=node_adj['x_mask'],
            up_senders=node_adj['up_senders'],
            up_receivers=node_adj['up_receivers'],
            up_mask=node_adj['up_mask'],
            static=node_static,
        )
        node_features = self.node_encoder_norm(node_features)
        if node_adj['x_mask'] is not None:
            node_features = jnp.where(node_adj['x_mask'][:, None], node_features, 0.0)
        
        # Collect per-layer node features
        layer_features = [node_features]
        
        # Layers 1+
        for layer in self.layers:
            node_features = layer(
                node_features=node_features,
                chem_embeddings=chem_embeddings,
                node_adj=node_adj,
                node_static=node_static,
            )
            layer_features.append(node_features)
        
        # Per-layer readout
        output = self.readout.read_and_combine(layer_features, node_adj['x_mask'])
        
        return {
            'node_features': node_features,
            'scalars': output['scalars'],
            'vectors': output['vectors'],
            'tensors': output['tensors'],
            'x_mask': node_adj['x_mask'],
        }



# =============================================================================
# SCALAR BASELINE WITH EDGES (fresh edge encoding, no edge persistence)
# =============================================================================

class ScalarBaselineEdgesLayer(nnx.Module):
    """A single ScalarBaselineEdges layer (for layers > 0).
    
    Edges are re-encoded fresh each layer from current node features (G gate),
    then refined with EdgeDown (ΔG gate) for angular information.
    Previous layer's edges enrich NodeUp messages.
    
    Cycle:
        1. NodeUp (r_ij gate, edge-enriched) → gated residual
        2. EdgeEncoder (fresh, G gate) → norm
        3. EdgeDown (ΔG gate) → gated residual
        4. EdgeFFN → gated residual
        5. NodeCoboundary (edge→node) → gated residual
        6. ChemicalReminder
    """
    
    def __init__(
        self,
        num_node_scalars: int,
        num_edge_scalars: int,
        embedding_dim: int,
        hidden_dim: int = 64,
        geometric_filter_dim: Optional[int] = None,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        if rngs is None:
            rngs = nnx.Rngs(0)
        
        # 1. Node up messages (edge-enriched with r_ij gate)
        self.node_up = ScalarNodeUpMessages(
            num_scalars=num_node_scalars,
            hidden_dim=hidden_dim,
            num_edge_scalars=num_edge_scalars,
            geometric_filter_dim=geometric_filter_dim,
            rngs=rngs,
        )
        self.node_up_block = ScalarGatedResidual(num_node_scalars, hidden_dim, rngs=rngs)
        self.node_up_edge_norm = nnx.LayerNorm(num_features=num_edge_scalars, rngs=rngs)
        
        # 2. Fresh edge encoder (G gate)
        self.edge_encoder_node_norm = nnx.LayerNorm(num_features=num_node_scalars, rngs=rngs)
        self.edge_encoder = ScalarEdgeEncoder(
            num_node_scalars=num_node_scalars,
            num_scalar_out=num_edge_scalars,
            hidden_dim=hidden_dim,
            geometric_filter_dim=geometric_filter_dim,
            rngs=rngs,
        )
        self.edge_encoder_norm = nnx.LayerNorm(num_features=num_edge_scalars, rngs=rngs)
        
        # 3. Edge down messages (angular bond communication, ΔG gate)
        self.edge_down = ScalarEdgeDownMessages(
            num_edge_scalars=num_edge_scalars,
            num_node_scalars=num_node_scalars,
            hidden_dim=hidden_dim,
            geometric_filter_dim=geometric_filter_dim,
            rngs=rngs,
        )
        self.edge_down_block = ScalarGatedResidual(num_edge_scalars, hidden_dim, rngs=rngs)
        self.edge_down_node_norm = nnx.LayerNorm(num_features=num_node_scalars, rngs=rngs)
        
        # 4. Edge FFN
        self.edge_ffn = ScalarEdgeFFN(num_edge_scalars, hidden_dim, rngs=rngs)
        self.edge_ffn_block = ScalarGatedResidual(num_edge_scalars, hidden_dim, rngs=rngs)
        
        # 5. Node coboundary messages (edge→node)
        self.node_coboundary = ScalarNodeCoboundaryMessages(
            num_node_scalars=num_node_scalars,
            num_edge_scalars=num_edge_scalars,
            hidden_dim=hidden_dim,
            rngs=rngs,
        )
        self.node_coboundary_block = ScalarGatedResidual(num_node_scalars, hidden_dim, rngs=rngs)
        
        # 6. Chemical reminder (end of cycle)
        self.chemical_reminder = ScalarChemicalReminder(
            num_scalars=num_node_scalars,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            rngs=rngs,
        )
    
    def __call__(
        self,
        node_features: jnp.ndarray,
        edge_features: jnp.ndarray,
        chem_embeddings: jnp.ndarray,
        node_adj: Dict,
        edge_adj: Dict,
        node_static: Optional[Dict] = None,
        edge_static: Optional[Dict] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        
        # 1. Edge-enriched node up messages (pre-norm → gated residual)
        normed = self.node_up_block.pre_norm(node_features)
        node_up_update = self.node_up(
            x=normed,
            x_mask=node_adj['x_mask'],
            up_senders=node_adj['up_senders'],
            up_receivers=node_adj['up_receivers'],
            up_mask=node_adj.get('up_mask'),
            static=node_static,
            edge_features=self.node_up_edge_norm(edge_features),
            up_intermediaries=node_adj.get('up_intermediaries'),
        )
        node_features = self.node_up_block.residual(node_features, node_up_update, node_adj['x_mask'])
        
        # 2. Fresh edge encoding (G gate)
        edge_features = self.edge_encoder(
            boundary_x=self.edge_encoder_node_norm(node_features),
            boundary_senders=edge_adj['boundary_senders'],
            boundary_receivers=edge_adj['boundary_receivers'],
            boundary_mask=edge_adj.get('boundary_mask'),
            x_mask=edge_adj['x_mask'],
            static=edge_static,
        )
        edge_features = self.edge_encoder_norm(edge_features)
        if edge_adj['x_mask'] is not None:
            edge_features = jnp.where(edge_adj['x_mask'][:, None], edge_features, 0.0)
        
        # 3. Edge down messages (pre-norm → ΔG gate → gated residual)
        normed_edges = self.edge_down_block.pre_norm(edge_features)
        edge_down_update = self.edge_down(
            x=normed_edges,
            x_mask=edge_adj['x_mask'],
            node_features=self.edge_down_node_norm(node_features),
            down_senders=edge_adj['down_senders'],
            down_receivers=edge_adj['down_receivers'],
            down_intermediaries=edge_adj['down_intermediaries'],
            down_mask=edge_adj.get('down_mask'),
            static=edge_static,
        )
        edge_features = self.edge_down_block.residual(edge_features, edge_down_update, edge_adj['x_mask'])
        
        # 4. Edge FFN (pre-norm → gated residual)
        normed_edges = self.edge_ffn_block.pre_norm(edge_features)
        edge_ffn_update = self.edge_ffn(normed_edges, edge_adj['x_mask'])
        edge_features = self.edge_ffn_block.residual(edge_features, edge_ffn_update, edge_adj['x_mask'])
        
        # 5. Node coboundary messages (edge→node, pre-norm → gated residual)
        normed = self.node_coboundary_block.pre_norm(node_features)
        node_cob_update = self.node_coboundary(
            x=normed,
            x_mask=node_adj['x_mask'],
            coboundary_x=edge_features,
            coboundary_senders=node_adj['coboundary_senders'],
            coboundary_receivers=node_adj['coboundary_receivers'],
            coboundary_mask=node_adj.get('coboundary_mask'),
        )
        node_features = self.node_coboundary_block.residual(node_features, node_cob_update, node_adj['x_mask'])
        
        # 6. Chemical reminder (end of cycle)
        node_features = self.chemical_reminder(
            x=node_features,
            chem_embeddings=chem_embeddings,
            x_mask=node_adj['x_mask'],
        )
        
        return node_features, edge_features


class ScalarBaselineEdges(nnx.Module):
    """
    Scalar Baseline with Edge Encoding.
    
    Adds fresh edge encoding (G gate), EdgeDown (ΔG gate) for angular
    refinement, and edge→node coboundary messages on top of ScalarBaseline.
    Edges are NOT persistent — they are re-encoded from current node
    features each layer, then refined with angular information.
    Previous layer's edges enrich NodeUp messages via up_intermediaries.
    
    Complexity ladder:
    ScalarBaseline (r_ij only) < ScalarBaselineEdges (r_ij + G + ΔG, fresh) < ScalarGNN (persistent edges)
    
    The only difference vs ScalarGNN is fresh edge encoding each layer
    instead of persistent edge updates via EdgeBoundary residuals.
    
    Layer 0: ChemicalEmbedding → NodeEncoder(r_ij) → EdgeEncoder(G) → EdgeDown(ΔG) → EdgeFFN → NodeCoboundary → ChemReminder
    Layers 1+: NodeUp(edge-enriched) → EdgeEncoder(fresh, G) → EdgeDown(ΔG) → EdgeFFN → NodeCoboundary → ChemReminder
    """
    
    def __init__(
        self,
        num_species: int,
        num_node_scalars: int = 32,
        num_edge_scalars: int = 32,
        embedding_dim: int = 32,
        hidden_dim: int = 64,
        geometric_filter_dim: Optional[int] = None,
        num_layers: int = 3,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        if rngs is None:
            rngs = nnx.Rngs(0)
        self.rngs = rngs
        
        self.num_node_scalars = num_node_scalars
        self.num_edge_scalars = num_edge_scalars
        self.num_layers = num_layers
        
        # Chemical embedding
        self.chemical_embedding = ChemicalEmbedding(
            num_species=num_species,
            embedding_dim=embedding_dim,
            rngs=rngs,
        )
        
        # Node encoder (r_ij gate)
        self.node_encoder = ScalarNodeEncoder(
            num_scalar_out=num_node_scalars,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            geometric_filter_dim=geometric_filter_dim,
            rngs=rngs,
        )
        self.node_encoder_norm = nnx.LayerNorm(num_features=num_node_scalars, rngs=rngs)
        
        # Edge encoder (G gate)
        self.edge_encoder = ScalarEdgeEncoder(
            num_node_scalars=num_node_scalars,
            num_scalar_out=num_edge_scalars,
            hidden_dim=hidden_dim,
            geometric_filter_dim=geometric_filter_dim,
            rngs=rngs,
        )
        self.edge_encoder_norm = nnx.LayerNorm(num_features=num_edge_scalars, rngs=rngs)
        
        # Layer 0: Edge down messages (ΔG gate)
        self.edge_down_init = ScalarEdgeDownMessages(
            num_edge_scalars=num_edge_scalars,
            num_node_scalars=num_node_scalars,
            hidden_dim=hidden_dim,
            geometric_filter_dim=geometric_filter_dim,
            rngs=rngs,
        )
        self.edge_down_block_init = ScalarGatedResidual(num_edge_scalars, hidden_dim, rngs=rngs)
        self.edge_down_node_norm_init = nnx.LayerNorm(num_features=num_node_scalars, rngs=rngs)
        
        # Layer 0: Edge FFN
        self.edge_ffn_init = ScalarEdgeFFN(num_edge_scalars, hidden_dim, rngs=rngs)
        self.edge_ffn_block_init = ScalarGatedResidual(num_edge_scalars, hidden_dim, rngs=rngs)
        
        # Layer 0: Node coboundary (edge→node)
        self.node_coboundary_init = ScalarNodeCoboundaryMessages(
            num_node_scalars=num_node_scalars,
            num_edge_scalars=num_edge_scalars,
            hidden_dim=hidden_dim,
            rngs=rngs,
        )
        self.node_coboundary_block_init = ScalarGatedResidual(num_node_scalars, hidden_dim, rngs=rngs)
        self.node_coboundary_edge_norm_init = nnx.LayerNorm(num_features=num_edge_scalars, rngs=rngs)
        
        # Layer 0: Chemical reminder
        self.chemical_reminder_init = ScalarChemicalReminder(
            num_scalars=num_node_scalars,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            rngs=rngs,
        )
        
        # Layers 1+
        self.layers = [
            ScalarBaselineEdgesLayer(
                num_node_scalars=num_node_scalars,
                num_edge_scalars=num_edge_scalars,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                geometric_filter_dim=geometric_filter_dim,
                rngs=rngs,
            )
            for _ in range(num_layers - 1)
        ]
        
        # Per-layer readout
        self.readout = PerLayerReadout(
            num_scalars_in=num_node_scalars,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            rngs=rngs,
        )
    
    def __call__(
        self,
        complex_batch: ComplexBatch,
    ) -> Dict[str, jnp.ndarray]:
        
        node_batch = complex_batch.cochain_batches[0]
        edge_batch = complex_batch.cochain_batches[1]
        
        node_adj = {
            'up_senders': node_batch.up_senders,
            'up_receivers': node_batch.up_receivers,
            'up_intermediaries': node_batch.up_intermediaries,
            'up_mask': node_batch.up_mask,
            'x_mask': node_batch.x_mask,
            'coboundary_senders': node_batch.coboundary_senders,
            'coboundary_receivers': node_batch.coboundary_receivers,
            'coboundary_mask': node_batch.coboundary_mask,
        }
        
        edge_adj = {
            'boundary_senders': edge_batch.boundary_senders,
            'boundary_receivers': edge_batch.boundary_receivers,
            'boundary_mask': edge_batch.boundary_mask,
            'down_senders': edge_batch.down_senders,
            'down_receivers': edge_batch.down_receivers,
            'down_intermediaries': edge_batch.down_intermediaries,
            'down_mask': edge_batch.down_mask,
            'x_mask': edge_batch.x_mask,
        }
        
        node_static = node_batch.static
        edge_static = edge_batch.static
        
        if node_static is None or 'Z' not in node_static:
            raise ValueError("node_static must contain 'Z'")
        species_indices = node_static['Z']
        
        # Chemical embeddings
        chem_embeddings = self.chemical_embedding(species_indices)
        
        # ---- Layer 0: Encoding ----
        
        # Node encoder (r_ij gate)
        node_features = self.node_encoder(
            chem_embeddings=chem_embeddings,
            x_mask=node_adj['x_mask'],
            up_senders=node_adj['up_senders'],
            up_receivers=node_adj['up_receivers'],
            up_mask=node_adj['up_mask'],
            static=node_static,
        )
        node_features = self.node_encoder_norm(node_features)
        if node_adj['x_mask'] is not None:
            node_features = jnp.where(node_adj['x_mask'][:, None], node_features, 0.0)
        
        # Edge encoder (G gate)
        edge_features = self.edge_encoder(
            boundary_x=node_features,
            boundary_senders=edge_adj['boundary_senders'],
            boundary_receivers=edge_adj['boundary_receivers'],
            boundary_mask=edge_adj['boundary_mask'],
            x_mask=edge_adj['x_mask'],
            static=edge_static,
        )
        edge_features = self.edge_encoder_norm(edge_features)
        if edge_adj['x_mask'] is not None:
            edge_features = jnp.where(edge_adj['x_mask'][:, None], edge_features, 0.0)
        
        # Layer 0: Edge down messages (ΔG gate)
        normed_edges = self.edge_down_block_init.pre_norm(edge_features)
        edge_down_update = self.edge_down_init(
            x=normed_edges,
            x_mask=edge_adj['x_mask'],
            node_features=self.edge_down_node_norm_init(node_features),
            down_senders=edge_adj['down_senders'],
            down_receivers=edge_adj['down_receivers'],
            down_intermediaries=edge_adj['down_intermediaries'],
            down_mask=edge_adj.get('down_mask'),
            static=edge_static,
        )
        edge_features = self.edge_down_block_init.residual(edge_features, edge_down_update, edge_adj['x_mask'])
        
        # Layer 0: Edge FFN
        normed_edges = self.edge_ffn_block_init.pre_norm(edge_features)
        edge_ffn_update = self.edge_ffn_init(normed_edges, edge_adj['x_mask'])
        edge_features = self.edge_ffn_block_init.residual(edge_features, edge_ffn_update, edge_adj['x_mask'])
        
        # Layer 0: Node coboundary (edge→node)
        normed_nodes = self.node_coboundary_block_init.pre_norm(node_features)
        node_cob_update = self.node_coboundary_init(
            x=normed_nodes,
            x_mask=node_adj['x_mask'],
            coboundary_x=self.node_coboundary_edge_norm_init(edge_features),
            coboundary_senders=node_adj['coboundary_senders'],
            coboundary_receivers=node_adj['coboundary_receivers'],
            coboundary_mask=node_adj['coboundary_mask'],
        )
        node_features = self.node_coboundary_block_init.residual(node_features, node_cob_update, node_adj['x_mask'])
        
        # Layer 0: Chemical reminder
        node_features = self.chemical_reminder_init(
            x=node_features,
            chem_embeddings=chem_embeddings,
            x_mask=node_adj['x_mask'],
        )
        
        # Collect per-layer node features
        layer_features = [node_features]
        
        # ---- Layers 1+ ----
        for layer in self.layers:
            node_features, edge_features = layer(
                node_features=node_features,
                edge_features=edge_features,
                chem_embeddings=chem_embeddings,
                node_adj=node_adj,
                edge_adj=edge_adj,
                node_static=node_static,
                edge_static=edge_static,
            )
            layer_features.append(node_features)
        
        # Per-layer readout
        output = self.readout.read_and_combine(layer_features, node_adj['x_mask'])
        
        return {
            'node_features': node_features,
            'edge_features': edge_features,
            'scalars': output['scalars'],
            'vectors': output['vectors'],
            'tensors': output['tensors'],
            'x_mask': node_adj['x_mask'],
        }



# =============================================================================
# SCALAR GNN
# =============================================================================

class ScalarGNNLayer(nnx.Module):
    """A single scalar GNN layer.
    
    Cycle: NodeUp+edge → EdgeBoundary → EdgeDown → EdgeFFN → NodeCoboundary → ChemReminder
    No node FFN (ChemReminder subsumes it).
    """
    
    def __init__(
        self,
        num_node_scalars: int,
        num_edge_scalars: int,
        embedding_dim: int,
        hidden_dim: int = 64,
        geometric_filter_dim: Optional[int] = None,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        if rngs is None:
            rngs = nnx.Rngs(0)
        
        # Node up messages (edge-enriched)
        self.node_up = ScalarNodeUpMessages(
            num_scalars=num_node_scalars,
            hidden_dim=hidden_dim,
            num_edge_scalars=num_edge_scalars,
            geometric_filter_dim=geometric_filter_dim,
            rngs=rngs,
        )
        self.node_up_block = ScalarGatedResidual(num_node_scalars, hidden_dim, rngs=rngs)
        self.node_up_edge_norm = nnx.LayerNorm(num_features=num_edge_scalars, rngs=rngs)
        
        # Edge boundary messages
        self.edge_boundary = ScalarEdgeBoundaryMessages(
            num_node_scalars=num_node_scalars,
            num_edge_scalars=num_edge_scalars,
            hidden_dim=hidden_dim,
            geometric_filter_dim=geometric_filter_dim,
            rngs=rngs,
        )
        self.edge_boundary_block = ScalarGatedResidual(num_edge_scalars, hidden_dim, rngs=rngs)
        self.edge_boundary_node_norm = nnx.LayerNorm(num_features=num_node_scalars, rngs=rngs)
        
        # Edge down messages (angular bond communication)
        self.edge_down = ScalarEdgeDownMessages(
            num_edge_scalars=num_edge_scalars,
            num_node_scalars=num_node_scalars,
            hidden_dim=hidden_dim,
            geometric_filter_dim=geometric_filter_dim,
            rngs=rngs,
        )
        self.edge_down_block = ScalarGatedResidual(num_edge_scalars, hidden_dim, rngs=rngs)
        self.edge_down_node_norm = nnx.LayerNorm(num_features=num_node_scalars, rngs=rngs)
        
        # Edge FFN
        self.edge_ffn = ScalarEdgeFFN(num_edge_scalars, hidden_dim, rngs=rngs)
        self.edge_ffn_block = ScalarGatedResidual(num_edge_scalars, hidden_dim, rngs=rngs)
        
        # Node coboundary messages
        self.node_coboundary = ScalarNodeCoboundaryMessages(
            num_node_scalars=num_node_scalars,
            num_edge_scalars=num_edge_scalars,
            hidden_dim=hidden_dim,
            rngs=rngs,
        )
        self.node_coboundary_block = ScalarGatedResidual(num_node_scalars, hidden_dim, rngs=rngs)
        self.node_coboundary_edge_norm = nnx.LayerNorm(num_features=num_edge_scalars, rngs=rngs)
        
        # Chemical reminder (end of cycle)
        self.chemical_reminder = ScalarChemicalReminder(
            num_scalars=num_node_scalars,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            rngs=rngs,
        )
    
    def __call__(
        self,
        node_features: jnp.ndarray,
        edge_features: jnp.ndarray,
        chem_embeddings: jnp.ndarray,
        node_adj: Dict,
        edge_adj: Dict,
        node_static: Optional[Dict] = None,
        edge_static: Optional[Dict] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        
        # 1. Edge-enriched node up messages (pre-norm → gated residual)
        normed_nodes = self.node_up_block.pre_norm(node_features)
        node_up_update = self.node_up(
            x=normed_nodes,
            x_mask=node_adj['x_mask'],
            up_senders=node_adj['up_senders'],
            up_receivers=node_adj['up_receivers'],
            up_mask=node_adj.get('up_mask'),
            static=node_static,
            edge_features=self.node_up_edge_norm(edge_features),
            up_intermediaries=node_adj.get('up_intermediaries'),
        )
        node_features = self.node_up_block.residual(node_features, node_up_update, node_adj['x_mask'])
        
        # 2. Edge boundary messages (gated residual)
        edge_boundary_update = self.edge_boundary(
            x=edge_features,
            x_mask=edge_adj['x_mask'],
            boundary_x=self.edge_boundary_node_norm(node_features),
            boundary_senders=edge_adj['boundary_senders'],
            boundary_receivers=edge_adj['boundary_receivers'],
            boundary_mask=edge_adj.get('boundary_mask'),
            static=edge_static,
        )
        edge_features = self.edge_boundary_block.residual(edge_features, edge_boundary_update, edge_adj['x_mask'])
        
        # 3. Edge down messages (pre-norm → ΔG gate → gated residual)
        normed_edges = self.edge_down_block.pre_norm(edge_features)
        edge_down_update = self.edge_down(
            x=normed_edges,
            x_mask=edge_adj['x_mask'],
            node_features=self.edge_down_node_norm(node_features),
            down_senders=edge_adj['down_senders'],
            down_receivers=edge_adj['down_receivers'],
            down_intermediaries=edge_adj['down_intermediaries'],
            down_mask=edge_adj.get('down_mask'),
            static=edge_static,
        )
        edge_features = self.edge_down_block.residual(edge_features, edge_down_update, edge_adj['x_mask'])
        
        # 4. Edge FFN (pre-norm → gated residual)
        normed_edges = self.edge_ffn_block.pre_norm(edge_features)
        edge_ffn_update = self.edge_ffn(normed_edges, edge_adj['x_mask'])
        edge_features = self.edge_ffn_block.residual(edge_features, edge_ffn_update, edge_adj['x_mask'])
        
        # 5. Node coboundary messages (pre-norm → gated residual)
        normed_nodes = self.node_coboundary_block.pre_norm(node_features)
        node_cob_update = self.node_coboundary(
            x=normed_nodes,
            x_mask=node_adj['x_mask'],
            coboundary_x=self.node_coboundary_edge_norm(edge_features),
            coboundary_senders=node_adj['coboundary_senders'],
            coboundary_receivers=node_adj['coboundary_receivers'],
            coboundary_mask=node_adj.get('coboundary_mask'),
        )
        node_features = self.node_coboundary_block.residual(node_features, node_cob_update, node_adj['x_mask'])
        
        # 6. Chemical reminder (end of cycle)
        node_features = self.chemical_reminder(
            x=node_features,
            chem_embeddings=chem_embeddings,
            x_mask=node_adj['x_mask'],
        )
        
        return node_features, edge_features


class ScalarGNN(nnx.Module):
    """
    Scalar Graph Neural Network.
    
    Uses only L=0 (scalar) features for nodes and edges.
    Geometric information is encoded via components:
    - Node encoder/up: uses 3 components of r_ij vector (tanh gate)
    - Edge encoder/boundary: uses 5 components of gyration tensor (tanh gate)
    - Edge down: uses 5 components of relative gyration tensor ΔG (tanh gate)
    
    NOT equivariant - outputs are treated as flat scalar features.
    """
    
    def __init__(
        self,
        num_species: int,
        num_node_scalars: int = 32,
        num_edge_scalars: int = 32,
        embedding_dim: int = 32,
        hidden_dim: int = 64,
        geometric_filter_dim: Optional[int] = None,
        num_layers: int = 3,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        if rngs is None:
            rngs = nnx.Rngs(0)
        self.rngs = rngs
        
        self.num_node_scalars = num_node_scalars
        self.num_edge_scalars = num_edge_scalars
        self.num_layers = num_layers
        
        # Chemical embedding
        self.chemical_embedding = ChemicalEmbedding(
            num_species=num_species,
            embedding_dim=embedding_dim,
            rngs=rngs,
        )
        
        # Node encoder
        self.node_encoder = ScalarNodeEncoder(
            num_scalar_out=num_node_scalars,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            geometric_filter_dim=geometric_filter_dim,
            rngs=rngs,
        )
        self.node_encoder_norm = nnx.LayerNorm(num_features=num_node_scalars, rngs=rngs)
        
        # Edge encoder
        self.edge_encoder = ScalarEdgeEncoder(
            num_node_scalars=num_node_scalars,
            num_scalar_out=num_edge_scalars,
            hidden_dim=hidden_dim,
            geometric_filter_dim=geometric_filter_dim,
            rngs=rngs,
        )
        self.edge_encoder_norm = nnx.LayerNorm(num_features=num_edge_scalars, rngs=rngs)
        
        # Layer 0: Edge down messages
        self.edge_down_init = ScalarEdgeDownMessages(
            num_edge_scalars=num_edge_scalars,
            num_node_scalars=num_node_scalars,
            hidden_dim=hidden_dim,
            geometric_filter_dim=geometric_filter_dim,
            rngs=rngs,
        )
        self.edge_down_block_init = ScalarGatedResidual(num_edge_scalars, hidden_dim, rngs=rngs)
        self.edge_down_node_norm_init = nnx.LayerNorm(num_features=num_node_scalars, rngs=rngs)
        
        # Layer 0: Edge FFN
        self.edge_ffn_init = ScalarEdgeFFN(num_edge_scalars, hidden_dim, rngs=rngs)
        self.edge_ffn_block_init = ScalarGatedResidual(num_edge_scalars, hidden_dim, rngs=rngs)
        
        # Layer 0: Node coboundary
        self.node_coboundary_init = ScalarNodeCoboundaryMessages(
            num_node_scalars=num_node_scalars,
            num_edge_scalars=num_edge_scalars,
            hidden_dim=hidden_dim,
            rngs=rngs,
        )
        self.node_coboundary_block_init = ScalarGatedResidual(num_node_scalars, hidden_dim, rngs=rngs)
        self.node_coboundary_edge_norm_init = nnx.LayerNorm(num_features=num_edge_scalars, rngs=rngs)
        
        # Layer 0: Chemical reminder
        self.chemical_reminder_init = ScalarChemicalReminder(
            num_scalars=num_node_scalars,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            rngs=rngs,
        )
        
        # Layers 1+
        self.layers = [
            ScalarGNNLayer(
                num_node_scalars=num_node_scalars,
                num_edge_scalars=num_edge_scalars,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                geometric_filter_dim=geometric_filter_dim,
                rngs=rngs,
            )
            for _ in range(num_layers - 1)
        ]
        
        # Per-layer readout
        self.readout = PerLayerReadout(
            num_scalars_in=num_node_scalars,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            rngs=rngs,
        )
    
    def __call__(
        self,
        complex_batch: ComplexBatch,
    ) -> Dict[str, jnp.ndarray]:
        
        node_batch = complex_batch.cochain_batches[0]
        edge_batch = complex_batch.cochain_batches[1]
        
        node_adj = {
            'up_senders': node_batch.up_senders,
            'up_receivers': node_batch.up_receivers,
            'up_intermediaries': node_batch.up_intermediaries,
            'up_mask': node_batch.up_mask,
            'x_mask': node_batch.x_mask,
            'coboundary_senders': node_batch.coboundary_senders,
            'coboundary_receivers': node_batch.coboundary_receivers,
            'coboundary_mask': node_batch.coboundary_mask,
        }
        
        edge_adj = {
            'boundary_senders': edge_batch.boundary_senders,
            'boundary_receivers': edge_batch.boundary_receivers,
            'boundary_mask': edge_batch.boundary_mask,
            'down_senders': edge_batch.down_senders,
            'down_receivers': edge_batch.down_receivers,
            'down_intermediaries': edge_batch.down_intermediaries,
            'down_mask': edge_batch.down_mask,
            'x_mask': edge_batch.x_mask,
        }
        
        node_static = node_batch.static
        edge_static = edge_batch.static
        
        if node_static is None or 'Z' not in node_static:
            raise ValueError("node_static must contain 'Z'")
        species_indices = node_static['Z']
        
        # Chemical embeddings
        chem_embeddings = self.chemical_embedding(species_indices)
        
        # Node encoder
        node_features = self.node_encoder(
            chem_embeddings=chem_embeddings,
            x_mask=node_adj['x_mask'],
            up_senders=node_adj['up_senders'],
            up_receivers=node_adj['up_receivers'],
            up_mask=node_adj['up_mask'],
            static=node_static,
        )
        node_features = self.node_encoder_norm(node_features)
        if node_adj['x_mask'] is not None:
            node_features = jnp.where(node_adj['x_mask'][:, None], node_features, 0.0)
        
        # Edge encoder
        edge_features = self.edge_encoder(
            boundary_x=node_features,
            boundary_senders=edge_adj['boundary_senders'],
            boundary_receivers=edge_adj['boundary_receivers'],
            boundary_mask=edge_adj['boundary_mask'],
            x_mask=edge_adj['x_mask'],
            static=edge_static,
        )
        edge_features = self.edge_encoder_norm(edge_features)
        if edge_adj['x_mask'] is not None:
            edge_features = jnp.where(edge_adj['x_mask'][:, None], edge_features, 0.0)
        
        # Layer 0: Edge down messages
        normed_edges = self.edge_down_block_init.pre_norm(edge_features)
        edge_down_update = self.edge_down_init(
            x=normed_edges,
            x_mask=edge_adj['x_mask'],
            node_features=self.edge_down_node_norm_init(node_features),
            down_senders=edge_adj['down_senders'],
            down_receivers=edge_adj['down_receivers'],
            down_intermediaries=edge_adj['down_intermediaries'],
            down_mask=edge_adj.get('down_mask'),
            static=edge_static,
        )
        edge_features = self.edge_down_block_init.residual(edge_features, edge_down_update, edge_adj['x_mask'])
        
        # Layer 0: Edge FFN
        normed_edges = self.edge_ffn_block_init.pre_norm(edge_features)
        edge_ffn_update = self.edge_ffn_init(normed_edges, edge_adj['x_mask'])
        edge_features = self.edge_ffn_block_init.residual(edge_features, edge_ffn_update, edge_adj['x_mask'])
        
        # Layer 0: Node coboundary
        normed_nodes = self.node_coboundary_block_init.pre_norm(node_features)
        node_cob_update = self.node_coboundary_init(
            x=normed_nodes,
            x_mask=node_adj['x_mask'],
            coboundary_x=self.node_coboundary_edge_norm_init(edge_features),
            coboundary_senders=node_adj['coboundary_senders'],
            coboundary_receivers=node_adj['coboundary_receivers'],
            coboundary_mask=node_adj['coboundary_mask'],
        )
        node_features = self.node_coboundary_block_init.residual(node_features, node_cob_update, node_adj['x_mask'])
        
        # Layer 0: Chemical reminder
        node_features = self.chemical_reminder_init(
            x=node_features,
            chem_embeddings=chem_embeddings,
            x_mask=node_adj['x_mask'],
        )
        
        # Collect per-layer node features
        layer_features = [node_features]
        
        # Layers 1+
        for layer in self.layers:
            node_features, edge_features = layer(
                node_features=node_features,
                edge_features=edge_features,
                chem_embeddings=chem_embeddings,
                node_adj=node_adj,
                edge_adj=edge_adj,
                node_static=node_static,
                edge_static=edge_static,
            )
            layer_features.append(node_features)
        
        # Per-layer readout
        output = self.readout.read_and_combine(layer_features, node_adj['x_mask'])
        
        return {
            'node_features': node_features,
            'edge_features': edge_features,
            'scalars': output['scalars'],
            'vectors': output['vectors'],
            'tensors': output['tensors'],
            'x_mask': node_adj['x_mask'],
        }



# =============================================================================
# SCALAR TPaiNN (bags-of-bonds for long-range communication)
# =============================================================================

class ScalarTPaiNNLayer(nnx.Module):
    """A single scalar bag-of-bonds TPaiNN layer.
    
    Cycle:
        1. NodeUp+edge (pre-norm → tanh gate → gated residual)
        2. EdgeBoundary (G-tanh gate → gated residual)
        3. EdgeDown (ΔG-tanh gate → gated residual)
        4. BagEncoder (fresh: edges → bags, no geometric gate)
        5. EdgeCoboundary (sigmoid gate → gated residual on edges)
        6. EdgeFFN (pre-norm → gated residual)
        7. DualChannelNodeUpdate (edge→node + bag→node, dual-gated)
        8. ChemicalReminder (pre-norm + chem_emb → gated residual)
    """
    
    def __init__(
        self,
        num_node_scalars: int,
        num_edge_scalars: int,
        num_bag_scalars: int,
        embedding_dim: int,
        hidden_dim: int = 64,
        geometric_filter_dim: Optional[int] = None,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        if rngs is None:
            rngs = nnx.Rngs(0)
        
        # 1. Node up messages (edge-enriched)
        self.node_up = ScalarNodeUpMessages(
            num_scalars=num_node_scalars,
            hidden_dim=hidden_dim,
            num_edge_scalars=num_edge_scalars,
            geometric_filter_dim=geometric_filter_dim,
            rngs=rngs,
        )
        self.node_up_block = ScalarGatedResidual(num_node_scalars, hidden_dim, rngs=rngs)
        self.node_up_edge_norm = nnx.LayerNorm(num_features=num_edge_scalars, rngs=rngs)
        
        # 2. Edge boundary messages
        self.edge_boundary = ScalarEdgeBoundaryMessages(
            num_node_scalars=num_node_scalars,
            num_edge_scalars=num_edge_scalars,
            hidden_dim=hidden_dim,
            geometric_filter_dim=geometric_filter_dim,
            rngs=rngs,
        )
        self.edge_boundary_block = ScalarGatedResidual(num_edge_scalars, hidden_dim, rngs=rngs)
        self.edge_boundary_node_norm = nnx.LayerNorm(num_features=num_node_scalars, rngs=rngs)
        
        # 3. Edge down messages (angular bond communication)
        self.edge_down = ScalarEdgeDownMessages(
            num_edge_scalars=num_edge_scalars,
            num_node_scalars=num_node_scalars,
            hidden_dim=hidden_dim,
            geometric_filter_dim=geometric_filter_dim,
            rngs=rngs,
        )
        self.edge_down_block = ScalarGatedResidual(num_edge_scalars, hidden_dim, rngs=rngs)
        self.edge_down_node_norm = nnx.LayerNorm(num_features=num_node_scalars, rngs=rngs)
        
        # 4. Bag encoder (fresh each layer)
        self.bag_encoder = ScalarBagEncoder(
            num_edge_scalars=num_edge_scalars,
            num_bag_scalars=num_bag_scalars,
            hidden_dim=hidden_dim,
            rngs=rngs,
        )
        
        # 5. Edge coboundary messages (bag → edge)
        self.edge_coboundary = ScalarEdgeCoboundaryMessages(
            num_edge_scalars=num_edge_scalars,
            num_bag_scalars=num_bag_scalars,
            hidden_dim=hidden_dim,
            rngs=rngs,
        )
        self.bag_decode_block = ScalarGatedResidual(num_edge_scalars, hidden_dim, rngs=rngs)
        self.edge_coboundary_bag_norm = nnx.LayerNorm(num_features=num_bag_scalars, rngs=rngs)
        
        # 6. Edge FFN
        self.edge_ffn = ScalarEdgeFFN(num_edge_scalars, hidden_dim, rngs=rngs)
        self.edge_ffn_block = ScalarGatedResidual(num_edge_scalars, hidden_dim, rngs=rngs)
        
        # 7. Dual-channel node update (edge→node + bag→node)
        self.node_coboundary = ScalarNodeCoboundaryMessages(
            num_node_scalars=num_node_scalars,
            num_edge_scalars=num_edge_scalars,
            hidden_dim=hidden_dim,
            rngs=rngs,
        )
        self.node_boundary = ScalarNodeBoundaryMessages(
            num_node_scalars=num_node_scalars,
            num_bag_scalars=num_bag_scalars,
            hidden_dim=hidden_dim,
            rngs=rngs,
        )
        self.dual_channel = ScalarDualChannelNodeUpdate(
            num_node_scalars=num_node_scalars,
            hidden_dim=hidden_dim,
            rngs=rngs,
        )
        self.dual_channel_edge_norm = nnx.LayerNorm(num_features=num_edge_scalars, rngs=rngs)
        self.node_boundary_bag_norm = nnx.LayerNorm(num_features=num_bag_scalars, rngs=rngs)
        
        # 8. Chemical reminder (end of cycle)
        self.chemical_reminder = ScalarChemicalReminder(
            num_scalars=num_node_scalars,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            rngs=rngs,
        )
    
    def __call__(
        self,
        node_features: jnp.ndarray,
        edge_features: jnp.ndarray,
        chem_embeddings: jnp.ndarray,
        node_adj: Dict,
        edge_adj: Dict,
        bag_adj: Dict,
        node_static: Optional[Dict] = None,
        edge_static: Optional[Dict] = None,
        bag_static: Optional[Dict] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        
        # 1. Edge-enriched node up messages (pre-norm → gated residual)
        normed_nodes = self.node_up_block.pre_norm(node_features)
        node_up_update = self.node_up(
            x=normed_nodes,
            x_mask=node_adj['x_mask'],
            up_senders=node_adj['up_senders'],
            up_receivers=node_adj['up_receivers'],
            up_mask=node_adj.get('up_mask'),
            static=node_static,
            edge_features=self.node_up_edge_norm(edge_features),
            up_intermediaries=node_adj.get('up_intermediaries'),
        )
        node_features = self.node_up_block.residual(node_features, node_up_update, node_adj['x_mask'])
        
        # 2. Edge boundary messages (gated residual)
        edge_boundary_update = self.edge_boundary(
            x=edge_features,
            x_mask=edge_adj['x_mask'],
            boundary_x=self.edge_boundary_node_norm(node_features),
            boundary_senders=edge_adj['boundary_senders'],
            boundary_receivers=edge_adj['boundary_receivers'],
            boundary_mask=edge_adj.get('boundary_mask'),
            static=edge_static,
        )
        edge_features = self.edge_boundary_block.residual(edge_features, edge_boundary_update, edge_adj['x_mask'])
        
        # 3. Edge down messages (pre-norm → ΔG gate → gated residual)
        normed_edges = self.edge_down_block.pre_norm(edge_features)
        edge_down_update = self.edge_down(
            x=normed_edges,
            x_mask=edge_adj['x_mask'],
            node_features=self.edge_down_node_norm(node_features),
            down_senders=edge_adj['down_senders'],
            down_receivers=edge_adj['down_receivers'],
            down_intermediaries=edge_adj['down_intermediaries'],
            down_mask=edge_adj.get('down_mask'),
            static=edge_static,
        )
        edge_features = self.edge_down_block.residual(edge_features, edge_down_update, edge_adj['x_mask'])
        
        # 4. Bag encoder (fresh each layer)
        bag_features = self.bag_encoder(
            boundary_x=edge_features,
            boundary_senders=bag_adj['boundary_senders'],
            boundary_receivers=bag_adj['boundary_receivers'],
            boundary_mask=bag_adj.get('boundary_mask'),
            x_mask=bag_adj['x_mask'],
        )
        
        # 5. Edge coboundary messages (bag → edge, gated residual)
        edge_cob_update = self.edge_coboundary(
            x=edge_features,
            x_mask=edge_adj['x_mask'],
            coboundary_x=self.edge_coboundary_bag_norm(bag_features),
            coboundary_senders=edge_adj['coboundary_senders'],
            coboundary_receivers=edge_adj['coboundary_receivers'],
            coboundary_mask=edge_adj.get('coboundary_mask'),
        )
        edge_features = self.bag_decode_block.residual(edge_features, edge_cob_update, edge_adj['x_mask'])
        
        # 6. Edge FFN (pre-norm → gated residual)
        normed_edges = self.edge_ffn_block.pre_norm(edge_features)
        edge_ffn_update = self.edge_ffn(normed_edges, edge_adj['x_mask'])
        edge_features = self.edge_ffn_block.residual(edge_features, edge_ffn_update, edge_adj['x_mask'])
        
        # 7. Dual-channel node update
        normed_nodes = self.dual_channel.pre_norm(node_features)
        
        edge_update = self.node_coboundary(
            x=normed_nodes,
            x_mask=node_adj['x_mask'],
            coboundary_x=self.dual_channel_edge_norm(edge_features),
            coboundary_senders=node_adj['coboundary_senders'],
            coboundary_receivers=node_adj['coboundary_receivers'],
            coboundary_mask=node_adj.get('coboundary_mask'),
        )
        
        bag_update = self.node_boundary(
            x=normed_nodes,
            x_mask=node_adj['x_mask'],
            boundary_x=self.node_boundary_bag_norm(bag_features),
            boundary_senders=node_adj['bag_senders'],
            boundary_receivers=node_adj['bag_receivers'],
            boundary_mask=node_adj.get('bag_mask'),
        )
        
        node_features = self.dual_channel.combine(
            node_features, edge_update, bag_update, node_adj['x_mask'],
        )
        
        # 8. Chemical reminder (end of cycle)
        node_features = self.chemical_reminder(
            x=node_features,
            chem_embeddings=chem_embeddings,
            x_mask=node_adj['x_mask'],
        )
        
        return node_features, edge_features


class ScalarTPaiNN(nnx.Module):
    """
    Scalar TPaiNN.
    
    Uses only L=0 (scalar) features for nodes, edges, and bags-of-bonds.
    Geometric information is encoded via tanh-bounded gates:
    - Node encoder/up: 3 components of r_ij vector
    - Edge encoder/boundary: 5 components of gyration tensor
    - Edge down: 5 components of relative gyration tensor ΔG
    
    No geometric gates at bag level — edge features already encode geometry
    from the EdgeEncoder and EdgeDown steps.
    
    Bags are re-encoded fresh each layer (communication device, not
    persistent representation).
    
    Dual-channel node update combines edge→node and bag→node information
    with independently learned per-feature sigmoid gates.
    
    Per-layer readout with learned weighted combination (softmax weights
    per property type: scalars, vectors, tensors).
    """
    
    def __init__(
        self,
        num_species: int,
        num_node_scalars: int = 32,
        num_edge_scalars: int = 32,
        num_bag_scalars: int = 32,
        embedding_dim: int = 32,
        hidden_dim: int = 64,
        geometric_filter_dim: Optional[int] = None,
        num_layers: int = 3,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        if rngs is None:
            rngs = nnx.Rngs(0)
        self.rngs = rngs
        
        self.num_node_scalars = num_node_scalars
        self.num_edge_scalars = num_edge_scalars
        self.num_bag_scalars = num_bag_scalars
        self.num_layers = num_layers
        
        # Chemical embedding
        self.chemical_embedding = ChemicalEmbedding(
            num_species=num_species,
            embedding_dim=embedding_dim,
            rngs=rngs,
        )
        
        # Node encoder
        self.node_encoder = ScalarNodeEncoder(
            num_scalar_out=num_node_scalars,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            geometric_filter_dim=geometric_filter_dim,
            rngs=rngs,
        )
        self.node_encoder_norm = nnx.LayerNorm(num_features=num_node_scalars, rngs=rngs)
        
        # Edge encoder
        self.edge_encoder = ScalarEdgeEncoder(
            num_node_scalars=num_node_scalars,
            num_scalar_out=num_edge_scalars,
            hidden_dim=hidden_dim,
            geometric_filter_dim=geometric_filter_dim,
            rngs=rngs,
        )
        self.edge_encoder_norm = nnx.LayerNorm(num_features=num_edge_scalars, rngs=rngs)
        
        # Layer 0: Edge down messages
        self.edge_down_init = ScalarEdgeDownMessages(
            num_edge_scalars=num_edge_scalars,
            num_node_scalars=num_node_scalars,
            hidden_dim=hidden_dim,
            geometric_filter_dim=geometric_filter_dim,
            rngs=rngs,
        )
        self.edge_down_block_init = ScalarGatedResidual(num_edge_scalars, hidden_dim, rngs=rngs)
        self.edge_down_node_norm_init = nnx.LayerNorm(num_features=num_node_scalars, rngs=rngs)
        
        # Layer 0: Bag encoder
        self.bag_encoder_init = ScalarBagEncoder(
            num_edge_scalars=num_edge_scalars,
            num_bag_scalars=num_bag_scalars,
            hidden_dim=hidden_dim,
            rngs=rngs,
        )
        
        # Layer 0: Edge coboundary messages (bag → edge)
        self.edge_coboundary_init = ScalarEdgeCoboundaryMessages(
            num_edge_scalars=num_edge_scalars,
            num_bag_scalars=num_bag_scalars,
            hidden_dim=hidden_dim,
            rngs=rngs,
        )
        self.bag_decode_block_init = ScalarGatedResidual(num_edge_scalars, hidden_dim, rngs=rngs)
        self.edge_coboundary_bag_norm_init = nnx.LayerNorm(num_features=num_bag_scalars, rngs=rngs)
        
        # Layer 0: Edge FFN
        self.edge_ffn_init = ScalarEdgeFFN(num_edge_scalars, hidden_dim, rngs=rngs)
        self.edge_ffn_block_init = ScalarGatedResidual(num_edge_scalars, hidden_dim, rngs=rngs)
        
        # Layer 0: Dual-channel node update
        self.node_coboundary_init = ScalarNodeCoboundaryMessages(
            num_node_scalars=num_node_scalars,
            num_edge_scalars=num_edge_scalars,
            hidden_dim=hidden_dim,
            rngs=rngs,
        )
        self.node_boundary_init = ScalarNodeBoundaryMessages(
            num_node_scalars=num_node_scalars,
            num_bag_scalars=num_bag_scalars,
            hidden_dim=hidden_dim,
            rngs=rngs,
        )
        self.dual_channel_init = ScalarDualChannelNodeUpdate(
            num_node_scalars=num_node_scalars,
            hidden_dim=hidden_dim,
            rngs=rngs,
        )
        self.dual_channel_edge_norm_init = nnx.LayerNorm(num_features=num_edge_scalars, rngs=rngs)
        self.node_boundary_bag_norm_init = nnx.LayerNorm(num_features=num_bag_scalars, rngs=rngs)
        
        # Layer 0: Chemical reminder
        self.chemical_reminder_init = ScalarChemicalReminder(
            num_scalars=num_node_scalars,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            rngs=rngs,
        )
        
        # Layers 1+
        self.layers = [
            ScalarTPaiNNLayer(
                num_node_scalars=num_node_scalars,
                num_edge_scalars=num_edge_scalars,
                num_bag_scalars=num_bag_scalars,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                geometric_filter_dim=geometric_filter_dim,
                rngs=rngs,
            )
            for _ in range(num_layers - 1)
        ]
        
        # Per-layer readout
        self.readout = PerLayerReadout(
            num_scalars_in=num_node_scalars,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            rngs=rngs,
        )
    
    def __call__(
        self,
        complex_batch: ComplexBatch,
    ) -> Dict[str, jnp.ndarray]:
        
        node_batch = complex_batch.cochain_batches[0]
        edge_batch = complex_batch.cochain_batches[1]
        bag_batch = complex_batch.cochain_batches[2]
        
        node_adj = {
            'up_senders': node_batch.up_senders,
            'up_receivers': node_batch.up_receivers,
            'up_intermediaries': node_batch.up_intermediaries,
            'up_mask': node_batch.up_mask,
            'x_mask': node_batch.x_mask,
            'coboundary_senders': node_batch.coboundary_senders,
            'coboundary_receivers': node_batch.coboundary_receivers,
            'coboundary_mask': node_batch.coboundary_mask,
            # Bag→node connectivity (from node's boundary relation with bags)
            'bag_senders': node_batch.boundary_senders,
            'bag_receivers': node_batch.boundary_receivers,
            'bag_mask': node_batch.boundary_mask,
        }
        
        edge_adj = {
            'boundary_senders': edge_batch.boundary_senders,
            'boundary_receivers': edge_batch.boundary_receivers,
            'boundary_mask': edge_batch.boundary_mask,
            'down_senders': edge_batch.down_senders,
            'down_receivers': edge_batch.down_receivers,
            'down_intermediaries': edge_batch.down_intermediaries,
            'down_mask': edge_batch.down_mask,
            'coboundary_senders': edge_batch.coboundary_senders,
            'coboundary_receivers': edge_batch.coboundary_receivers,
            'coboundary_mask': edge_batch.coboundary_mask,
            'x_mask': edge_batch.x_mask,
        }
        
        bag_adj = {
            'boundary_senders': bag_batch.boundary_senders,
            'boundary_receivers': bag_batch.boundary_receivers,
            'boundary_mask': bag_batch.boundary_mask,
            'x_mask': bag_batch.x_mask,
        }
        
        node_static = node_batch.static
        edge_static = edge_batch.static
        bag_static = bag_batch.static
        
        if node_static is None or 'Z' not in node_static:
            raise ValueError("node_static must contain 'Z'")
        species_indices = node_static['Z']
        
        # Chemical embeddings
        chem_embeddings = self.chemical_embedding(species_indices)
        
        # ---- Layer 0: Encoding ----
        
        # Node encoder
        node_features = self.node_encoder(
            chem_embeddings=chem_embeddings,
            x_mask=node_adj['x_mask'],
            up_senders=node_adj['up_senders'],
            up_receivers=node_adj['up_receivers'],
            up_mask=node_adj['up_mask'],
            static=node_static,
        )
        node_features = self.node_encoder_norm(node_features)
        if node_adj['x_mask'] is not None:
            node_features = jnp.where(node_adj['x_mask'][:, None], node_features, 0.0)
        
        # Edge encoder
        edge_features = self.edge_encoder(
            boundary_x=node_features,
            boundary_senders=edge_adj['boundary_senders'],
            boundary_receivers=edge_adj['boundary_receivers'],
            boundary_mask=edge_adj['boundary_mask'],
            x_mask=edge_adj['x_mask'],
            static=edge_static,
        )
        edge_features = self.edge_encoder_norm(edge_features)
        if edge_adj['x_mask'] is not None:
            edge_features = jnp.where(edge_adj['x_mask'][:, None], edge_features, 0.0)
        
        # Layer 0: Edge down messages
        normed_edges = self.edge_down_block_init.pre_norm(edge_features)
        edge_down_update = self.edge_down_init(
            x=normed_edges,
            x_mask=edge_adj['x_mask'],
            node_features=self.edge_down_node_norm_init(node_features),
            down_senders=edge_adj['down_senders'],
            down_receivers=edge_adj['down_receivers'],
            down_intermediaries=edge_adj['down_intermediaries'],
            down_mask=edge_adj.get('down_mask'),
            static=edge_static,
        )
        edge_features = self.edge_down_block_init.residual(edge_features, edge_down_update, edge_adj['x_mask'])
        
        # Layer 0: Bag encoder (fresh)
        bag_features = self.bag_encoder_init(
            boundary_x=edge_features,
            boundary_senders=bag_adj['boundary_senders'],
            boundary_receivers=bag_adj['boundary_receivers'],
            boundary_mask=bag_adj['boundary_mask'],
            x_mask=bag_adj['x_mask'],
        )
        
        # Layer 0: Edge coboundary messages (bag → edge)
        edge_cob_update = self.edge_coboundary_init(
            x=edge_features,
            x_mask=edge_adj['x_mask'],
            coboundary_x=self.edge_coboundary_bag_norm_init(bag_features),
            coboundary_senders=edge_adj['coboundary_senders'],
            coboundary_receivers=edge_adj['coboundary_receivers'],
            coboundary_mask=edge_adj['coboundary_mask'],
        )
        edge_features = self.bag_decode_block_init.residual(edge_features, edge_cob_update, edge_adj['x_mask'])
        
        # Layer 0: Edge FFN
        normed_edges = self.edge_ffn_block_init.pre_norm(edge_features)
        edge_ffn_update = self.edge_ffn_init(normed_edges, edge_adj['x_mask'])
        edge_features = self.edge_ffn_block_init.residual(edge_features, edge_ffn_update, edge_adj['x_mask'])
        
        # Layer 0: Dual-channel node update
        normed_nodes = self.dual_channel_init.pre_norm(node_features)
        
        edge_update = self.node_coboundary_init(
            x=normed_nodes,
            x_mask=node_adj['x_mask'],
            coboundary_x=self.dual_channel_edge_norm_init(edge_features),
            coboundary_senders=node_adj['coboundary_senders'],
            coboundary_receivers=node_adj['coboundary_receivers'],
            coboundary_mask=node_adj['coboundary_mask'],
        )
        
        bag_update = self.node_boundary_init(
            x=normed_nodes,
            x_mask=node_adj['x_mask'],
            boundary_x=self.node_boundary_bag_norm_init(bag_features),
            boundary_senders=node_adj['bag_senders'],
            boundary_receivers=node_adj['bag_receivers'],
            boundary_mask=node_adj.get('bag_mask'),
        )
        
        node_features = self.dual_channel_init.combine(
            node_features, edge_update, bag_update, node_adj['x_mask'],
        )
        
        # Layer 0: Chemical reminder
        node_features = self.chemical_reminder_init(
            x=node_features,
            chem_embeddings=chem_embeddings,
            x_mask=node_adj['x_mask'],
        )
        
        # Collect per-layer node features
        layer_features = [node_features]
        
        # ---- Layers 1+ ----
        for layer in self.layers:
            node_features, edge_features = layer(
                node_features=node_features,
                edge_features=edge_features,
                chem_embeddings=chem_embeddings,
                node_adj=node_adj,
                edge_adj=edge_adj,
                bag_adj=bag_adj,
                node_static=node_static,
                edge_static=edge_static,
                bag_static=bag_static,
            )
            layer_features.append(node_features)
        
        # Per-layer readout
        output = self.readout.read_and_combine(layer_features, node_adj['x_mask'])
        
        return {
            'node_features': node_features,
            'edge_features': edge_features,
            'scalars': output['scalars'],
            'vectors': output['vectors'],
            'tensors': output['tensors'],
            'x_mask': node_adj['x_mask'],
        }
