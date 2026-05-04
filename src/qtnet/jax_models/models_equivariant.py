from typing import Optional, Dict, List, Tuple

import jax.numpy as jnp
import jax
import flax.nnx as nnx

from qtnet.jax_models.scalar_layers import ChemicalEmbedding
from qtnet.jax_models.equivariant_layers import (
    # Initial encoders
    EquivariantNodeEncoder,
    EquivariantEdgeEncoder,
    EquivariantBagEncoder,
    # Message passing
    EquivariantNodeUpMessages,
    EquivariantEdgeBoundaryMessages,
    EquivariantEdgeDownMessages,
    EquivariantNodeCoboundaryMessages,
    EquivariantNodeBoundaryMessages,
    EquivariantEdgeCoboundaryMessages,
    # Normalization and residual
    EquivariantNorm,
    EquivariantGatedResidual,
    # Dual-channel node update
    EquivariantDualChannelUpdate,
    # Feed-forward (EdgeFFN)
    EquivariantFFN,
    # Reminders
    ChemicalReminder,
    GeometryReminder,
    # Output heads
    NodeHead,
    EquivariantPerLayerReadout,
)

from qtnet.jax_models.representations import ComplexBatch


class EquivariantGNNLayer(nnx.Module):
    """
    A single equivariant GNN layer (for layers > 1).
    
    Message passing cycle with pre-norm + gated residuals:
        1. NodeUp → gated_residual
        2. EdgeBoundary → gated_residual
        3. EdgeDown → gated_residual
        4. EdgeFFN (EdgeFFN) → gated_residual
        5. NodeCoboundary → gated_residual
        6. ChemicalReminder (END)
    """
    
    def __init__(
        self,
        node_irreps: str,
        edge_irreps: str,
        embedding_dim: int,
        hidden_dim: int = 64,
        hidden_l1_channels: int = 4,
        hidden_l2_channels: int = 4,
        geometric_filter_dim: int = 32,
        geo_basis_dim: int = 16,
        cutoff: float = 5.0,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        if rngs is None:
            rngs = nnx.Rngs(0)
        self.rngs = rngs
        
        self.node_irreps = node_irreps
        self.edge_irreps = edge_irreps
        
        # === 1. Node Up Messages ===
        self.node_up_messages = EquivariantNodeUpMessages(
            irreps_in=node_irreps,
            hidden_dim=hidden_dim,
            hidden_l1_channels=hidden_l1_channels,
            hidden_l2_channels=hidden_l2_channels,
            geometric_filter_dim=geometric_filter_dim,
            geo_basis_dim=geo_basis_dim,
            cutoff=cutoff,
            edge_irreps=edge_irreps,
            rngs=rngs,
        )

        self.geo_reinjection = GeometryReminder(
            irreps_in=node_irreps,
            hidden_dim=hidden_dim,
            hidden_l1_channels=hidden_l1_channels,
            hidden_l2_channels=hidden_l2_channels,
            geometric_filter_dim=geometric_filter_dim,
            geo_basis_dim=geo_basis_dim,
            cutoff=cutoff,
            edge_irreps=edge_irreps,   # intermediary — same as NodeUp
            rngs=rngs,
        )

        # Dual update from GeometryReminder and NodeUp:
        self.node_up_dual = EquivariantDualChannelUpdate(
            irreps=node_irreps,
            gate_hidden_dim=hidden_dim,
            rngs=rngs,
        )

        self.node_up_edge_norm = EquivariantNorm(irreps=edge_irreps, rngs=rngs)
        
        # === 2. Edge Boundary Messages (from nodes) ===
        self.edge_boundary_messages = EquivariantEdgeBoundaryMessages(
            node_irreps=node_irreps,
            edge_irreps=edge_irreps,
            hidden_dim=hidden_dim,
            hidden_l1_channels=hidden_l1_channels,
            hidden_l2_channels=hidden_l2_channels,
            geometric_filter_dim=geometric_filter_dim,
            geo_basis_dim=geo_basis_dim,
            cutoff=cutoff,
            rngs=rngs,
        )
        self.edge_boundary_res = EquivariantGatedResidual(
            irreps=edge_irreps,
            gate_hidden_dim=hidden_dim,
            rngs=rngs,
        )
        self.edge_boundary_node_norm = EquivariantNorm(irreps=node_irreps, rngs=rngs)
        
        # === 3. Edge Down Messages (edge-to-edge via shared nodes) ===
        self.edge_down_messages = EquivariantEdgeDownMessages(
            edge_irreps=edge_irreps,
            node_irreps=node_irreps,
            hidden_dim=hidden_dim,
            hidden_l1_channels=hidden_l1_channels,
            hidden_l2_channels=hidden_l2_channels,
            geometric_filter_dim=geometric_filter_dim,
            geo_basis_dim=geo_basis_dim,
            rngs=rngs,
        )
        self.edge_down_res = EquivariantGatedResidual(
            irreps=edge_irreps,
            gate_hidden_dim=hidden_dim,
            rngs=rngs,
        )
        self.edge_down_node_norm = EquivariantNorm(irreps=node_irreps, rngs=rngs)
        
        # === 4. Edge Self-Mixing (EdgeFFN) ===
        self.edge_self_mixing = EquivariantFFN(
            irreps=edge_irreps,
            hidden_dim=hidden_dim,
            rngs=rngs,
        )
        self.edge_mix_res = EquivariantGatedResidual(
            irreps=edge_irreps,
            gate_hidden_dim=hidden_dim,
            rngs=rngs,
        )
        
        # === 5. Node Coboundary Messages (from edges) ===
        self.node_coboundary_messages = EquivariantNodeCoboundaryMessages(
            node_irreps=node_irreps,
            edge_irreps=edge_irreps,
            hidden_dim=hidden_dim,
            hidden_l1_channels=hidden_l1_channels,
            hidden_l2_channels=hidden_l2_channels,
            rngs=rngs,
        )
        self.node_coboundary_res = EquivariantGatedResidual(
            irreps=node_irreps,
            gate_hidden_dim=hidden_dim,
            rngs=rngs,
        )
        self.node_coboundary_edge_norm = EquivariantNorm(irreps=edge_irreps, rngs=rngs)
        
        # === 6. Chemical Reminder (END of cycle) ===
        self.chemical_reminder = ChemicalReminder(
            irreps=node_irreps,
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
        """Forward pass for a single GNN layer."""
        
        # === 1.1 Node Up Messages ===
        normed_nodes = self.node_up_dual.pre_norm(node_features, node_adj['x_mask'])
        normed_edges = self.node_up_edge_norm(edge_features, edge_adj['x_mask'])
        
        node_up_out = self.node_up_messages(
            x=normed_nodes,
            x_mask=node_adj['x_mask'],
            up_senders=node_adj['up_senders'],
            up_receivers=node_adj['up_receivers'],
            up_mask=node_adj.get('up_mask'),
            static=node_static,
            edge_features=normed_edges,
            up_intermediaries=node_adj.get('up_intermediaries'),
        )

        # === 1.2 GeometryReminder ===

        geometry_out = self.geo_reinjection(
            x=normed_nodes,
            x_mask=node_adj['x_mask'],
            up_senders=node_adj['up_senders'],
            up_receivers=node_adj['up_receivers'],
            up_mask=node_adj.get('up_mask'),
            static=node_static,
            edge_features=normed_edges,
            up_intermediaries=node_adj.get('up_intermediaries'), 
        )

        node_features = self.node_up_dual.combine(
            node_features, node_up_out['x'], geometry_out['x'], node_adj['x_mask'],
        )
        
        # === 2. Edge Boundary Messages (from nodes) ===
        normed_edges = self.edge_boundary_res.pre_norm(edge_features, edge_adj['x_mask'])
        edge_boundary_out = self.edge_boundary_messages(
            x=normed_edges,
            x_mask=edge_adj['x_mask'],
            boundary_x=self.edge_boundary_node_norm(node_features, node_adj['x_mask']),
            boundary_senders=edge_adj['boundary_senders'],
            boundary_receivers=edge_adj['boundary_receivers'],
            boundary_mask=edge_adj.get('boundary_mask'),
            static=edge_static,
        )
        edge_features = self.edge_boundary_res.residual(
            edge_features, edge_boundary_out['x'], edge_adj['x_mask'],
        )
        
        # === 3. Edge Down Messages ===
        normed_edges = self.edge_down_res.pre_norm(edge_features, edge_adj['x_mask'])
        edge_down_out = self.edge_down_messages(
            x=normed_edges,
            x_mask=edge_adj['x_mask'],
            node_features=self.edge_down_node_norm(node_features, node_adj['x_mask']),
            down_senders=edge_adj['down_senders'],
            down_receivers=edge_adj['down_receivers'],
            down_intermediaries=edge_adj['down_intermediaries'],
            down_mask=edge_adj.get('down_mask'),
            static=edge_static,
        )
        edge_features = self.edge_down_res.residual(
            edge_features, edge_down_out['x'], edge_adj['x_mask'],
        )
        
        # === 4. Edge Self-Mixing (EdgeFFN) ===
        normed_edges = self.edge_mix_res.pre_norm(edge_features, edge_adj['x_mask'])
        edge_mix_out = self.edge_self_mixing(
            x=normed_edges,
            x_mask=edge_adj['x_mask'],
        )
        edge_features = self.edge_mix_res.residual(
            edge_features, edge_mix_out['x'], edge_adj['x_mask'],
        )
        
        # === 5. Node Coboundary Messages (from edges) ===
        normed_nodes = self.node_coboundary_res.pre_norm(node_features, node_adj['x_mask'])
        node_coboundary_out = self.node_coboundary_messages(
            x=normed_nodes,
            x_mask=node_adj['x_mask'],
            coboundary_x=self.node_coboundary_edge_norm(edge_features, edge_adj['x_mask']),
            coboundary_senders=node_adj['coboundary_senders'],
            coboundary_receivers=node_adj['coboundary_receivers'],
            coboundary_mask=node_adj.get('coboundary_mask'),
        )
        node_features = self.node_coboundary_res.residual(
            node_features, node_coboundary_out['x'], node_adj['x_mask'],
        )
        
        # === 6. Chemical Reminder (END) ===
        node_features = self.chemical_reminder(
            x=node_features,
            chem_embeddings=chem_embeddings,
            x_mask=node_adj['x_mask'],
        )
        
        return node_features, edge_features


class EquivariantGNN(nnx.Module):
    """
    Equivariant Graph Neural Network (without bags/bags).
    
    Node features: L=0 (scalars) + L=1 (vectors) + L=2 (tensors)
    Edge features: L=0 (scalars) + L=1 (vectors) + L=2 (tensors)
    
    Layer 0: NodeEncoder(L=0+L=1+L=2) → Norm → EdgeEncoder → Norm →
             EdgeDown → gated_res → EdgeFFN → gated_res →
             NodeCoboundary → gated_res → ChemReminder
    
    Layers 1+: EquivariantGNNLayer (pre-norm + gated residual cycle)
    
    Output: EquivariantPerLayerReadout (softmax-weighted per-layer heads)
    """
    
    def __init__(
        self,
        num_species: int,
        num_node_scalars: int = 8,
        num_node_vectors: int = 4,
        num_node_tensors: int = 2,
        num_edge_scalars: int = 8,
        num_edge_vectors: int = 4,
        num_edge_tensors: int = 4,
        embedding_dim: int = 32,
        hidden_dim: int = 64,
        hidden_l1_channels: int = 8,
        hidden_l2_channels: int = 8,
        num_layers: int = 3,
        geometric_filter_dim: int = 32,
        geo_basis_dim: int = 16,
        cutoff: float = 5.0,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        if rngs is None:
            rngs = nnx.Rngs(0)
        self.rngs = rngs
        
        # Build irreps strings
        self.node_irreps = f"{num_node_scalars}x0e + {num_node_vectors}x1o + {num_node_tensors}x2e"
        self.edge_irreps = f"{num_edge_scalars}x0e + {num_edge_vectors}x1o + {num_edge_tensors}x2e"
        
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.hidden_l1_channels = hidden_l1_channels
        self.hidden_l2_channels = hidden_l2_channels
        
        # === Chemical Embedding ===
        self.chemical_embedding = ChemicalEmbedding(
            num_species=num_species,
            embedding_dim=embedding_dim,
            rngs=rngs,
        )
        
        # === Layer 0: Initial Encoders ===
        # NodeEncoder now produces L=0 + L=1 + L=2 (from r̂⊗r̂)
        self.node_encoder = EquivariantNodeEncoder(
            num_scalar_out=num_node_scalars,
            num_vector_out=num_node_vectors,
            num_tensor_out=num_node_tensors,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            geometric_filter_dim=geometric_filter_dim,
            geo_basis_dim=geo_basis_dim,
            cutoff=cutoff,
            rngs=rngs,
        )
        self.node_encoder_norm = EquivariantNorm(
            irreps=self.node_irreps,
            rngs=rngs,
        )
        
        # Edge Encoder
        self.edge_encoder = EquivariantEdgeEncoder(
            node_irreps_in=self.node_irreps,
            num_scalar_out=num_edge_scalars,
            num_l1_out=num_edge_vectors,
            num_l2_out=num_edge_tensors,
            hidden_dim=hidden_dim,
            geometric_filter_dim=geometric_filter_dim,
            geo_basis_dim=geo_basis_dim,
            cutoff=cutoff,
            rngs=rngs,
        )
        self.edge_encoder_norm = EquivariantNorm(
            irreps=self.edge_irreps,
            rngs=rngs,
        )
        
        # Edge Down (layer 0)
        self.edge_down_init = EquivariantEdgeDownMessages(
            edge_irreps=self.edge_irreps,
            node_irreps=self.node_irreps,
            hidden_dim=hidden_dim,
            hidden_l1_channels=hidden_l1_channels,
            hidden_l2_channels=hidden_l2_channels,
            geometric_filter_dim=geometric_filter_dim,
            geo_basis_dim=geo_basis_dim,
            rngs=rngs,
        )
        self.edge_down_res_init = EquivariantGatedResidual(
            irreps=self.edge_irreps,
            gate_hidden_dim=hidden_dim,
            rngs=rngs,
        )
        self.edge_down_node_norm_init = EquivariantNorm(
            irreps=self.node_irreps,
            rngs=rngs,
        )
        
        # Edge Self-Mixing / EdgeFFN (layer 0)
        self.edge_self_mixing_init = EquivariantFFN(
            irreps=self.edge_irreps,
            hidden_dim=hidden_dim,
            rngs=rngs,
        )
        self.edge_mix_res_init = EquivariantGatedResidual(
            irreps=self.edge_irreps,
            gate_hidden_dim=hidden_dim,
            rngs=rngs,
        )
        
        # Node Coboundary (layer 0)
        self.node_coboundary_init = EquivariantNodeCoboundaryMessages(
            node_irreps=self.node_irreps,
            edge_irreps=self.edge_irreps,
            hidden_dim=hidden_dim,
            hidden_l1_channels=hidden_l1_channels,
            hidden_l2_channels=hidden_l2_channels,
            rngs=rngs,
        )
        self.node_coboundary_res_init = EquivariantGatedResidual(
            irreps=self.node_irreps,
            gate_hidden_dim=hidden_dim,
            rngs=rngs,
        )
        self.node_coboundary_edge_norm_init = EquivariantNorm(
            irreps=self.edge_irreps,
            rngs=rngs,
        )
        
        # Chemical Reminder at end of layer 0 (Fix I)
        self.chemical_reminder_init = ChemicalReminder(
            irreps=self.node_irreps,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            rngs=rngs,
        )
        
        # === Layers 1+ ===
        self.layers = [
            EquivariantGNNLayer(
                node_irreps=self.node_irreps,
                edge_irreps=self.edge_irreps,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                hidden_l1_channels=hidden_l1_channels,
                hidden_l2_channels=hidden_l2_channels,
                geometric_filter_dim=geometric_filter_dim,
                geo_basis_dim=geo_basis_dim,
                cutoff=cutoff,
                rngs=rngs,
            )
            for _ in range(num_layers - 1)
        ]
        
        # === Per-Layer Readout (Fix E) ===
        self.readout = EquivariantPerLayerReadout(
            node_irreps=self.node_irreps,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            rngs=rngs,
        )
    
    def __call__(
        self,
        complex_batch: ComplexBatch,
    ) -> Dict[str, jnp.ndarray]:
        """Forward pass through the model."""
        
        # Extract cochain batches
        node_batch = complex_batch.cochain_batches[0]
        edge_batch = complex_batch.cochain_batches[1]
        
        # Build adjacency dictionaries
        node_adj = {
            'up_senders': node_batch.up_senders,
            'up_receivers': node_batch.up_receivers,
            'up_mask': node_batch.up_mask,
            'up_intermediaries': node_batch.up_intermediaries if hasattr(node_batch, 'up_intermediaries') else None,
            'x_mask': node_batch.x_mask,
            'coboundary_senders': node_batch.coboundary_senders,
            'coboundary_receivers': node_batch.coboundary_receivers,
            'coboundary_mask': node_batch.coboundary_mask,
        }
        
        edge_adj = {
            'boundary_senders': edge_batch.boundary_senders,
            'boundary_receivers': edge_batch.boundary_receivers,
            'boundary_mask': edge_batch.boundary_mask,
            'down_senders': edge_batch.down_senders if hasattr(edge_batch, 'down_senders') else None,
            'down_receivers': edge_batch.down_receivers if hasattr(edge_batch, 'down_receivers') else None,
            'down_intermediaries': edge_batch.down_intermediaries if hasattr(edge_batch, 'down_intermediaries') else None,
            'down_mask': edge_batch.down_mask if hasattr(edge_batch, 'down_mask') else None,
            'x_mask': edge_batch.x_mask,
        }
        
        node_static = node_batch.static
        edge_static = edge_batch.static
        
        # Get species indices
        if node_static is None or 'Z' not in node_static:
            raise ValueError("node_static must contain 'Z' key with species indices")
        species_indices = node_static['Z']
        
        # === Chemical Embeddings ===
        chem_embeddings = self.chemical_embedding(species_indices)
        
        # === Layer 0 ===
        # Node Encoder (now produces L=0 + L=1 + L=2)
        node_features = self.node_encoder(
            chem_embeddings=chem_embeddings,
            x_mask=node_adj['x_mask'],
            up_senders=node_adj['up_senders'],
            up_receivers=node_adj['up_receivers'],
            up_mask=node_adj['up_mask'],
            static=node_static,
        )
        node_features = self.node_encoder_norm(
            x=node_features,
            x_mask=node_adj['x_mask'],
        )
        
        # Edge Encoder
        edge_out = self.edge_encoder(
            boundary_x=node_features,
            boundary_senders=edge_adj['boundary_senders'],
            boundary_receivers=edge_adj['boundary_receivers'],
            boundary_mask=edge_adj['boundary_mask'],
            x_mask=edge_adj['x_mask'],
            static=edge_static,
        )
        edge_features = edge_out['x']
        edge_features = self.edge_encoder_norm(
            x=edge_features,
            x_mask=edge_adj['x_mask'],
        )
        
        # Edge Down (layer 0)
        normed_edges = self.edge_down_res_init.pre_norm(edge_features, edge_adj['x_mask'])
        edge_down_out = self.edge_down_init(
            x=normed_edges,
            x_mask=edge_adj['x_mask'],
            node_features=self.edge_down_node_norm_init(node_features, node_adj['x_mask']),
            down_senders=edge_adj['down_senders'],
            down_receivers=edge_adj['down_receivers'],
            down_intermediaries=edge_adj['down_intermediaries'],
            down_mask=edge_adj.get('down_mask'),
            static=edge_static,
        )
        edge_features = self.edge_down_res_init.residual(
            edge_features, edge_down_out['x'], edge_adj['x_mask'],
        )
        
        # Edge Self-Mixing (layer 0)
        normed_edges = self.edge_mix_res_init.pre_norm(edge_features, edge_adj['x_mask'])
        edge_mix_out = self.edge_self_mixing_init(
            x=normed_edges,
            x_mask=edge_adj['x_mask'],
        )
        edge_features = self.edge_mix_res_init.residual(
            edge_features, edge_mix_out['x'], edge_adj['x_mask'],
        )
        
        # Node Coboundary (layer 0)
        normed_nodes = self.node_coboundary_res_init.pre_norm(node_features, node_adj['x_mask'])
        normed_edges_for_cob = self.node_coboundary_edge_norm_init(edge_features, edge_adj['x_mask'])
        node_coboundary_out = self.node_coboundary_init(
            x=normed_nodes,
            x_mask=node_adj['x_mask'],
            coboundary_x=normed_edges_for_cob,
            coboundary_senders=node_adj['coboundary_senders'],
            coboundary_receivers=node_adj['coboundary_receivers'],
            coboundary_mask=node_adj['coboundary_mask'],
        )
        node_features = self.node_coboundary_res_init.residual(
            node_features, node_coboundary_out['x'], node_adj['x_mask'],
        )
        
        # Chemical Reminder at end of layer 0 (Fix I)
        node_features = self.chemical_reminder_init(
            x=node_features,
            chem_embeddings=chem_embeddings,
            x_mask=node_adj['x_mask'],
        )
        
        # Collect per-layer features for readout
        node_features_per_layer = [node_features]
        
        # === Layers 1+ ===
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
            node_features_per_layer.append(node_features)
        
        # === Per-Layer Readout ===
        head_output = self.readout(
            node_features_per_layer=node_features_per_layer,
            x_mask=node_adj['x_mask'],
        )
        
        return {
            'node_features': node_features,
            'edge_features': edge_features,
            'scalars': head_output['scalars'],
            'vectors': head_output['vectors'],
            'tensors': head_output['tensors'],
            'x_mask': node_adj['x_mask'],
        }



class TPaiNNLayer(nnx.Module):
    """
    A single T-PaiNN layer (for layers > 0).
    
    Message passing cycle with pre-norm + gated residuals:
        1. NodeUp → gated_residual
        2. EdgeBoundary → gated_residual
        3. EdgeDown → gated_residual
        4. BagEncoder → EdgeCoboundary → gated_residual
        5. EdgeFFN → gated_residual
        6. DualChannelNodeUpdate (NodeCoboundary + NodeBoundary, dual-gated)
        7. ChemicalReminder (END)
    """
    
    def __init__(
        self,
        node_irreps: str,
        edge_irreps: str,
        bag_irreps: str,
        embedding_dim: int,
        hidden_dim: int = 64,
        hidden_l1_channels: int = 16,
        hidden_l2_channels: int = 16,
        geometric_filter_dim: int = 32,
        geo_basis_dim: int = 16,
        cutoff: float = 5.0,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        if rngs is None:
            rngs = nnx.Rngs(0)
        self.rngs = rngs
        
        self.node_irreps = node_irreps
        self.edge_irreps = edge_irreps
        self.bag_irreps = bag_irreps
        
        # === 1. Node Up Messages ===
        self.node_up_messages = EquivariantNodeUpMessages(
            irreps_in=node_irreps,
            hidden_dim=hidden_dim,
            hidden_l1_channels=hidden_l1_channels,
            hidden_l2_channels=hidden_l2_channels,
            geometric_filter_dim=geometric_filter_dim,
            geo_basis_dim=geo_basis_dim,
            cutoff=cutoff,
            edge_irreps=edge_irreps,
            rngs=rngs,
        )
        self.geo_reinjection = GeometryReminder(
            irreps_in=node_irreps,
            hidden_dim=hidden_dim,
            hidden_l1_channels=hidden_l1_channels,
            hidden_l2_channels=hidden_l2_channels,
            geometric_filter_dim=geometric_filter_dim,
            geo_basis_dim=geo_basis_dim,
            cutoff=cutoff,
            edge_irreps=edge_irreps,   # intermediary — same as NodeUp
            rngs=rngs,
        )

        # Dual update from GeometryReminder and NodeUp:
        self.node_up_dual = EquivariantDualChannelUpdate(
            irreps=node_irreps,
            gate_hidden_dim=hidden_dim,
            rngs=rngs,
        )

        self.node_up_edge_norm = EquivariantNorm(irreps=edge_irreps, rngs=rngs)
        
        # === 2. Edge Boundary Messages (from nodes) ===
        self.edge_boundary_messages = EquivariantEdgeBoundaryMessages(
            node_irreps=node_irreps,
            edge_irreps=edge_irreps,
            hidden_dim=hidden_dim,
            hidden_l1_channels=hidden_l1_channels,
            hidden_l2_channels=hidden_l2_channels,
            geometric_filter_dim=geometric_filter_dim,
            geo_basis_dim=geo_basis_dim,
            cutoff=cutoff,
            rngs=rngs,
        )
        self.edge_boundary_res = EquivariantGatedResidual(
            irreps=edge_irreps,
            gate_hidden_dim=hidden_dim,
            rngs=rngs,
        )
        self.edge_boundary_node_norm = EquivariantNorm(
            irreps=node_irreps,
            rngs=rngs,
        )
        
        # === 3. Edge Down Messages ===
        self.edge_down_messages = EquivariantEdgeDownMessages(
            edge_irreps=edge_irreps,
            node_irreps=node_irreps,
            hidden_dim=hidden_dim,
            hidden_l1_channels=hidden_l1_channels,
            hidden_l2_channels=hidden_l2_channels,
            geometric_filter_dim=geometric_filter_dim,
            geo_basis_dim=geo_basis_dim,
            rngs=rngs,
        )
        self.edge_down_res = EquivariantGatedResidual(
            irreps=edge_irreps,
            gate_hidden_dim=hidden_dim,
            rngs=rngs,
        )
        self.edge_down_node_norm = EquivariantNorm(
            irreps=node_irreps,
            rngs=rngs,
        )
        
        # === 4. Bag Encoder / Decoder ===
        import e3nn_jax as e3nn
        _bag_irreps = e3nn.Irreps(bag_irreps)
        _bag_s = sum(mul for mul, ir in _bag_irreps if ir.l == 0)
        _bag_l1 = sum(mul for mul, ir in _bag_irreps if ir.l == 1)
        _bag_l2 = sum(mul for mul, ir in _bag_irreps if ir.l == 2)
        self.bag_encoder = EquivariantBagEncoder(
            edge_irreps_in=edge_irreps,
            num_scalar_out=_bag_s,
            num_l1_out=_bag_l1,
            num_l2_out=_bag_l2,
            hidden_dim=hidden_dim,
            rngs=rngs,
        )
        self.edge_coboundary = EquivariantEdgeCoboundaryMessages(
            edge_irreps=edge_irreps,
            bag_irreps=bag_irreps,
            hidden_dim=hidden_dim,
            hidden_l1_channels=hidden_l1_channels,
            hidden_l2_channels=hidden_l2_channels,
            rngs=rngs,
        )
        self.bag_decode_res = EquivariantGatedResidual(
            irreps=edge_irreps,
            gate_hidden_dim=hidden_dim,
            rngs=rngs,
        )
        self.edge_coboundary_bag_norm = EquivariantNorm(
            irreps=bag_irreps,
            rngs=rngs,
        )
        
        # === 5. Edge Self-Mixing (EdgeFFN) ===
        self.edge_self_mixing = EquivariantFFN(
            irreps=edge_irreps,
            hidden_dim=hidden_dim,
            rngs=rngs,
        )
        self.edge_mix_res = EquivariantGatedResidual(
            irreps=edge_irreps,
            gate_hidden_dim=hidden_dim,
            rngs=rngs,
        )
        
        # === 6. Dual-Channel Node Update (edge→node + bag→node) ===
        self.node_coboundary_messages = EquivariantNodeCoboundaryMessages(
            node_irreps=node_irreps,
            edge_irreps=edge_irreps,
            hidden_dim=hidden_dim,
            hidden_l1_channels=hidden_l1_channels,
            hidden_l2_channels=hidden_l2_channels,
            rngs=rngs,
        )
        self.node_boundary_messages = EquivariantNodeBoundaryMessages(
            node_irreps=node_irreps,
            bag_irreps=bag_irreps,
            hidden_dim=hidden_dim,
            hidden_l1_channels=hidden_l1_channels,
            hidden_l2_channels=hidden_l2_channels,
            rngs=rngs,
        )
        self.dual_channel_node_update = EquivariantDualChannelUpdate(
            irreps=node_irreps,
            gate_hidden_dim=hidden_dim,
            rngs=rngs,
        )
        self.dual_channel_edge_norm = EquivariantNorm(
            irreps=edge_irreps,
            rngs=rngs,
        )
        self.node_boundary_bag_norm = EquivariantNorm(
            irreps=bag_irreps,
            rngs=rngs,
        )
        
        # === 7. Chemical Reminder (END of cycle) ===
        self.chemical_reminder = ChemicalReminder(
            irreps=node_irreps,
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
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass for a single T-PaiNN layer."""
        
        # === 1.1 Node Up Messages ===
        normed_nodes = self.node_up_dual.pre_norm(node_features, node_adj['x_mask'])
        normed_edges = self.node_up_edge_norm(edge_features, edge_adj['x_mask'])
        
        node_up_out = self.node_up_messages(
            x=normed_nodes,
            x_mask=node_adj['x_mask'],
            up_senders=node_adj['up_senders'],
            up_receivers=node_adj['up_receivers'],
            up_mask=node_adj.get('up_mask'),
            static=node_static,
            edge_features=normed_edges,
            up_intermediaries=node_adj.get('up_intermediaries'),
        )

        # === 1.2 GeometryReminder ===

        geometry_out = self.geo_reinjection(
            x=normed_nodes,
            x_mask=node_adj['x_mask'],
            up_senders=node_adj['up_senders'],
            up_receivers=node_adj['up_receivers'],
            up_mask=node_adj.get('up_mask'),
            static=node_static,
            edge_features=normed_edges,
            up_intermediaries=node_adj.get('up_intermediaries'), 
        )

        node_features = self.node_up_dual.combine(
            node_features, node_up_out['x'], geometry_out['x'], node_adj['x_mask'],
        )
        
        # === 2. Edge Boundary Messages ===
        normed_edges = self.edge_boundary_res.pre_norm(edge_features, edge_adj['x_mask'])
        edge_boundary_out = self.edge_boundary_messages(
            x=normed_edges,
            x_mask=edge_adj['x_mask'],
            boundary_x=self.edge_boundary_node_norm(node_features, node_adj['x_mask']),
            boundary_senders=edge_adj['boundary_senders'],
            boundary_receivers=edge_adj['boundary_receivers'],
            boundary_mask=edge_adj.get('boundary_mask'),
            static=edge_static,
        )
        edge_features = self.edge_boundary_res.residual(
            edge_features, edge_boundary_out['x'], edge_adj['x_mask'],
        )
        
        # === 3. Edge Down Messages ===
        normed_edges = self.edge_down_res.pre_norm(edge_features, edge_adj['x_mask'])
        edge_down_out = self.edge_down_messages(
            x=normed_edges,
            x_mask=edge_adj['x_mask'],
            node_features=self.edge_down_node_norm(node_features, node_adj['x_mask']),
            down_senders=edge_adj['down_senders'],
            down_receivers=edge_adj['down_receivers'],
            down_intermediaries=edge_adj['down_intermediaries'],
            down_mask=edge_adj.get('down_mask'),
            static=edge_static,
        )
        edge_features = self.edge_down_res.residual(
            edge_features, edge_down_out['x'], edge_adj['x_mask'],
        )
        
        # === 4. Bag Encoder → Decoder ===
        bag_out = self.bag_encoder(
            boundary_x=edge_features,
            boundary_senders=bag_adj['boundary_senders'],
            boundary_receivers=bag_adj['boundary_receivers'],
            boundary_mask=bag_adj.get('boundary_mask'),
            x_mask=bag_adj['x_mask'],
        )
        bag_features = bag_out['x']
        
        bag_decode_out = self.edge_coboundary(
            x=edge_features,
            x_mask=edge_adj['x_mask'],
            coboundary_x=self.edge_coboundary_bag_norm(bag_features, bag_adj['x_mask']),
            coboundary_senders=edge_adj['coboundary_senders'],
            coboundary_receivers=edge_adj['coboundary_receivers'],
            coboundary_mask=edge_adj.get('coboundary_mask'),
        )
        edge_features = self.bag_decode_res.residual(
            edge_features, bag_decode_out['x'], edge_adj['x_mask'],
        )
        
        # === 5. Edge Self-Mixing (EdgeFFN) ===
        normed_edges = self.edge_mix_res.pre_norm(edge_features, edge_adj['x_mask'])
        edge_mix_out = self.edge_self_mixing(
            x=normed_edges,
            x_mask=edge_adj['x_mask'],
        )
        edge_features = self.edge_mix_res.residual(
            edge_features, edge_mix_out['x'], edge_adj['x_mask'],
        )
        
        # === 6. Dual-Channel Node Update ===
        normed_nodes = self.dual_channel_node_update.pre_norm(node_features, node_adj['x_mask'])
        normed_edges_for_cob = self.dual_channel_edge_norm(edge_features, edge_adj['x_mask'])
        
        # Edge→node channel
        node_coboundary_out = self.node_coboundary_messages(
            x=normed_nodes,
            x_mask=node_adj['x_mask'],
            coboundary_x=normed_edges_for_cob,
            coboundary_senders=node_adj['coboundary_senders'],
            coboundary_receivers=node_adj['coboundary_receivers'],
            coboundary_mask=node_adj.get('coboundary_mask'),
        )
        
        # Bag→node channel
        node_boundary_out = self.node_boundary_messages(
            x=normed_nodes,
            x_mask=node_adj['x_mask'],
            boundary_x=self.node_boundary_bag_norm(bag_features, bag_adj['x_mask']),
            boundary_senders=node_adj['bag_senders'],
            boundary_receivers=node_adj['bag_receivers'],
            boundary_mask=node_adj.get('bag_mask'),
        )
        
        # Combine with dual gating
        node_features = self.dual_channel_node_update.combine(
            node_features, node_coboundary_out['x'], node_boundary_out['x'],
            node_adj['x_mask'],
        )
        
        # === 7. Chemical Reminder (END) ===
        node_features = self.chemical_reminder(
            x=node_features,
            chem_embeddings=chem_embeddings,
            x_mask=node_adj['x_mask'],
        )
        
        return node_features, edge_features


class TPaiNN(nnx.Module):
    """
    Topological PaiNN (T-PaiNN) Model.
    
    A topological message passing neural network operating on cell complexes
    with nodes (0-cells), edges (1-cells), and bags/bags (2-cells).
    
    Layer 0: NodeEncoder(L=0+L=1+L=2) → Norm → EdgeEncoder → Norm →
             EdgeDown → gated_res → BagEncoder → EdgeCoboundary → gated_res →
             EdgeFFN → gated_res → DualChannelNodeUpdate (NodeCoboundary + NodeBoundary) → ChemReminder
    
    Layers 1+: TPaiNNLayer (pre-norm + gated residual cycle with bags)
    
    Output: EquivariantPerLayerReadout (softmax-weighted per-layer heads)
    """
    
    def __init__(
        self,
        num_species: int,
        num_node_scalars: int = 8,
        num_node_vectors: int = 4,
        num_node_tensors: int = 2,
        num_edge_scalars: int = 8,
        num_edge_vectors: int = 4,
        num_edge_tensors: int = 4,
        num_bag_scalars: int = 8,
        num_bag_vectors: int = 4,
        num_bag_tensors: int = 4,
        embedding_dim: int = 32,
        hidden_dim: int = 64,
        hidden_l1_channels: int = 8,
        hidden_l2_channels: int = 8,
        num_layers: int = 3,
        geometric_filter_dim: int = 32,
        geo_basis_dim: int = 16,
        cutoff: float = 5.0,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        if rngs is None:
            rngs = nnx.Rngs(0)
        self.rngs = rngs
        
        # Build irreps strings
        self.node_irreps = f"{num_node_scalars}x0e + {num_node_vectors}x1o + {num_node_tensors}x2e"
        self.edge_irreps = f"{num_edge_scalars}x0e + {num_edge_vectors}x1o + {num_edge_tensors}x2e"
        self.bag_irreps = f"{num_bag_scalars}x0e + {num_bag_vectors}x1o + {num_bag_tensors}x2e"
        
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.hidden_l1_channels = hidden_l1_channels
        self.hidden_l2_channels = hidden_l2_channels
        
        # === Chemical Embedding ===
        self.chemical_embedding = ChemicalEmbedding(
            num_species=num_species,
            embedding_dim=embedding_dim,
            rngs=rngs,
        )
        
        # === Layer 0: Initial Encoders ===
        # NodeEncoder now produces L=0 + L=1 + L=2 (from r̂⊗r̂)
        self.node_encoder = EquivariantNodeEncoder(
            num_scalar_out=num_node_scalars,
            num_vector_out=num_node_vectors,
            num_tensor_out=num_node_tensors,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            geometric_filter_dim=geometric_filter_dim,
            geo_basis_dim=geo_basis_dim,
            cutoff=cutoff,
            rngs=rngs,
        )
        self.node_encoder_norm = EquivariantNorm(
            irreps=self.node_irreps,
            rngs=rngs,
        )
        
        # Edge Encoder
        self.edge_encoder = EquivariantEdgeEncoder(
            node_irreps_in=self.node_irreps,
            num_scalar_out=num_edge_scalars,
            num_l1_out=num_edge_vectors,
            num_l2_out=num_edge_tensors,
            hidden_dim=hidden_dim,
            geometric_filter_dim=geometric_filter_dim,
            geo_basis_dim=geo_basis_dim,
            cutoff=cutoff,
            rngs=rngs,
        )
        self.edge_encoder_norm = EquivariantNorm(
            irreps=self.edge_irreps,
            rngs=rngs,
        )
        
        # Edge Down (layer 0)
        self.edge_down_init = EquivariantEdgeDownMessages(
            edge_irreps=self.edge_irreps,
            node_irreps=self.node_irreps,
            hidden_dim=hidden_dim,
            hidden_l1_channels=hidden_l1_channels,
            hidden_l2_channels=hidden_l2_channels,
            geometric_filter_dim=geometric_filter_dim,
            geo_basis_dim=geo_basis_dim,
            rngs=rngs,
        )
        self.edge_down_res_init = EquivariantGatedResidual(
            irreps=self.edge_irreps,
            gate_hidden_dim=hidden_dim,
            rngs=rngs,
        )
        self.edge_down_node_norm_init = EquivariantNorm(
            irreps=self.node_irreps,
            rngs=rngs,
        )
        
        # Bag Encoder (layer 0)
        self.bag_encoder_init = EquivariantBagEncoder(
            edge_irreps_in=self.edge_irreps,
            num_scalar_out=num_bag_scalars,
            num_l1_out=num_bag_vectors,
            num_l2_out=num_bag_tensors,
            hidden_dim=hidden_dim,
            rngs=rngs,
        )
        
        # Bag Decoder (layer 0)
        self.edge_coboundary_init = EquivariantEdgeCoboundaryMessages(
            edge_irreps=self.edge_irreps,
            bag_irreps=self.bag_irreps,
            hidden_dim=hidden_dim,
            hidden_l1_channels=hidden_l1_channels,
            hidden_l2_channels=hidden_l2_channels,
            rngs=rngs,
        )
        self.bag_decode_res_init = EquivariantGatedResidual(
            irreps=self.edge_irreps,
            gate_hidden_dim=hidden_dim,
            rngs=rngs,
        )
        self.edge_coboundary_bag_norm_init = EquivariantNorm(
            irreps=self.bag_irreps,
            rngs=rngs,
        )
        
        # Edge Self-Mixing / EdgeFFN (layer 0)
        self.edge_self_mixing_init = EquivariantFFN(
            irreps=self.edge_irreps,
            hidden_dim=hidden_dim,
            rngs=rngs,
        )
        self.edge_mix_res_init = EquivariantGatedResidual(
            irreps=self.edge_irreps,
            gate_hidden_dim=hidden_dim,
            rngs=rngs,
        )
        
        # Node Coboundary + Boundary → Dual-Channel Update (layer 0)
        self.node_coboundary_init = EquivariantNodeCoboundaryMessages(
            node_irreps=self.node_irreps,
            edge_irreps=self.edge_irreps,
            hidden_dim=hidden_dim,
            hidden_l1_channels=hidden_l1_channels,
            hidden_l2_channels=hidden_l2_channels,
            rngs=rngs,
        )
        self.node_boundary_init = EquivariantNodeBoundaryMessages(
            node_irreps=self.node_irreps,
            bag_irreps=self.bag_irreps,
            hidden_dim=hidden_dim,
            hidden_l1_channels=hidden_l1_channels,
            hidden_l2_channels=hidden_l2_channels,
            rngs=rngs,
        )
        self.dual_channel_node_update_init = EquivariantDualChannelUpdate(
            irreps=self.node_irreps,
            gate_hidden_dim=hidden_dim,
            rngs=rngs,
        )
        self.dual_channel_edge_norm_init = EquivariantNorm(
            irreps=self.edge_irreps,
            rngs=rngs,
        )
        self.node_boundary_bag_norm_init = EquivariantNorm(
            irreps=self.bag_irreps,
            rngs=rngs,
        )
        
        # Chemical Reminder at end of layer 0
        self.chemical_reminder_init = ChemicalReminder(
            irreps=self.node_irreps,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            rngs=rngs,
        )
        
        # === Layers 1+ ===
        self.layers = [
            TPaiNNLayer(
                node_irreps=self.node_irreps,
                edge_irreps=self.edge_irreps,
                bag_irreps=self.bag_irreps,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                hidden_l1_channels=hidden_l1_channels,
                hidden_l2_channels=hidden_l2_channels,
                geometric_filter_dim=geometric_filter_dim,
                geo_basis_dim=geo_basis_dim,
                cutoff=cutoff,
                rngs=rngs,
            )
            for _ in range(num_layers - 1)
        ]
        
        # === Per-Layer Readout ===
        self.readout = EquivariantPerLayerReadout(
            node_irreps=self.node_irreps,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            rngs=rngs,
        )
    
    def __call__(
        self,
        complex_batch: ComplexBatch,
    ) -> Dict[str, jnp.ndarray]:
        """Forward pass through the T-PaiNN model."""
        
        # Extract cochain batches
        node_batch = complex_batch.cochain_batches[0]
        edge_batch = complex_batch.cochain_batches[1]
        bag_batch = complex_batch.cochain_batches[2]
        
        # Build adjacency dictionaries
        node_adj = {
            'up_senders': node_batch.up_senders,
            'up_receivers': node_batch.up_receivers,
            'up_mask': node_batch.up_mask,
            'up_intermediaries': node_batch.up_intermediaries if hasattr(node_batch, 'up_intermediaries') else None,
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
            'down_senders': edge_batch.down_senders if hasattr(edge_batch, 'down_senders') else None,
            'down_receivers': edge_batch.down_receivers if hasattr(edge_batch, 'down_receivers') else None,
            'down_intermediaries': edge_batch.down_intermediaries if hasattr(edge_batch, 'down_intermediaries') else None,
            'down_mask': edge_batch.down_mask if hasattr(edge_batch, 'down_mask') else None,
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
        
        # Get static features
        node_static = node_batch.static
        edge_static = edge_batch.static
        
        # Get species indices
        if node_static is None or 'Z' not in node_static:
            raise ValueError("node_static must contain 'Z' key with species indices")
        species_indices = node_static['Z']
        
        # === Chemical Embeddings ===
        chem_embeddings = self.chemical_embedding(species_indices)
        
        # === Layer 0 ===
        # Node Encoder (produces L=0 + L=1 + L=2)
        node_features = self.node_encoder(
            chem_embeddings=chem_embeddings,
            x_mask=node_adj['x_mask'],
            up_senders=node_adj['up_senders'],
            up_receivers=node_adj['up_receivers'],
            up_mask=node_adj['up_mask'],
            static=node_static,
        )
        node_features = self.node_encoder_norm(
            x=node_features,
            x_mask=node_adj['x_mask'],
        )
        
        # Edge Encoder
        edge_out = self.edge_encoder(
            boundary_x=node_features,
            boundary_senders=edge_adj['boundary_senders'],
            boundary_receivers=edge_adj['boundary_receivers'],
            boundary_mask=edge_adj['boundary_mask'],
            x_mask=edge_adj['x_mask'],
            static=edge_static,
        )
        edge_features = edge_out['x']
        edge_features = self.edge_encoder_norm(
            x=edge_features,
            x_mask=edge_adj['x_mask'],
        )
        
        # Edge Down (layer 0)
        normed_edges = self.edge_down_res_init.pre_norm(edge_features, edge_adj['x_mask'])
        edge_down_out = self.edge_down_init(
            x=normed_edges,
            x_mask=edge_adj['x_mask'],
            node_features=self.edge_down_node_norm_init(node_features, node_adj['x_mask']),
            down_senders=edge_adj['down_senders'],
            down_receivers=edge_adj['down_receivers'],
            down_intermediaries=edge_adj['down_intermediaries'],
            down_mask=edge_adj.get('down_mask'),
            static=edge_static,
        )
        edge_features = self.edge_down_res_init.residual(
            edge_features, edge_down_out['x'], edge_adj['x_mask'],
        )
        
        # Bag Encoder → Decoder (layer 0)
        bag_out = self.bag_encoder_init(
            boundary_x=edge_features,
            boundary_senders=bag_adj['boundary_senders'],
            boundary_receivers=bag_adj['boundary_receivers'],
            boundary_mask=bag_adj['boundary_mask'],
            x_mask=bag_adj['x_mask'],
        )
        bag_features = bag_out['x']
        
        bag_decode_out = self.edge_coboundary_init(
            x=edge_features,
            x_mask=edge_adj['x_mask'],
            coboundary_x=self.edge_coboundary_bag_norm_init(bag_features, bag_adj['x_mask']),
            coboundary_senders=edge_adj['coboundary_senders'],
            coboundary_receivers=edge_adj['coboundary_receivers'],
            coboundary_mask=edge_adj['coboundary_mask'],
        )
        edge_features = self.bag_decode_res_init.residual(
            edge_features, bag_decode_out['x'], edge_adj['x_mask'],
        )
        
        # Edge Self-Mixing (layer 0)
        normed_edges = self.edge_mix_res_init.pre_norm(edge_features, edge_adj['x_mask'])
        edge_mix_out = self.edge_self_mixing_init(
            x=normed_edges,
            x_mask=edge_adj['x_mask'],
        )
        edge_features = self.edge_mix_res_init.residual(
            edge_features, edge_mix_out['x'], edge_adj['x_mask'],
        )
        
        # Dual-Channel Node Update (layer 0)
        normed_nodes = self.dual_channel_node_update_init.pre_norm(node_features, node_adj['x_mask'])
        normed_edges_for_cob = self.dual_channel_edge_norm_init(edge_features, edge_adj['x_mask'])
        
        # Edge→node channel
        node_coboundary_out = self.node_coboundary_init(
            x=normed_nodes,
            x_mask=node_adj['x_mask'],
            coboundary_x=normed_edges_for_cob,
            coboundary_senders=node_adj['coboundary_senders'],
            coboundary_receivers=node_adj['coboundary_receivers'],
            coboundary_mask=node_adj['coboundary_mask'],
        )
        
        # Bag→node channel
        node_boundary_out = self.node_boundary_init(
            x=normed_nodes,
            x_mask=node_adj['x_mask'],
            boundary_x=self.node_boundary_bag_norm_init(bag_features, bag_adj['x_mask']),
            boundary_senders=node_adj['bag_senders'],
            boundary_receivers=node_adj['bag_receivers'],
            boundary_mask=node_adj.get('bag_mask'),
        )
        
        # Combine with dual gating
        node_features = self.dual_channel_node_update_init.combine(
            node_features, node_coboundary_out['x'], node_boundary_out['x'],
            node_adj['x_mask'],
        )
        
        # Chemical Reminder at end of layer 0
        node_features = self.chemical_reminder_init(
            x=node_features,
            chem_embeddings=chem_embeddings,
            x_mask=node_adj['x_mask'],
        )
        
        # Collect per-layer features for readout
        node_features_per_layer = [node_features]
        
        # === Layers 1+ ===
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
            )
            node_features_per_layer.append(node_features)
        
        # === Per-Layer Readout ===
        head_output = self.readout(
            node_features_per_layer=node_features_per_layer,
            x_mask=node_adj['x_mask'],
        )
        
        return {
            'node_features': node_features,
            'edge_features': edge_features,
            'scalars': head_output['scalars'],
            'vectors': head_output['vectors'],
            'tensors': head_output['tensors'],
            'x_mask': node_adj['x_mask'],
        }
