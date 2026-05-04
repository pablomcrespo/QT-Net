"""
Benchmark utilities for T-PaiNN evaluation.

This module provides:
- Metrics computation for equivariant predictions
- Utility functions for model evaluation on test sets
- Conversion utilities for results interpretation

L=2 Tensor Representation:
    Both predictions and targets use 5-component format [xy, xz, yz, (xx-yy)/2, zz].
    For reporting, we convert to 6-component Cartesian for interpretability.
"""
import time
from typing import Dict, List

import numpy as np
import jax
import flax.nnx as nnx

from qtnet.data_utils import comp5_to_cartesian
# from qtnet.benchmark_utils import prepare_batch_targets


# =============================================================================
# Metrics Computation
# =============================================================================

def compute_metrics(predictions: Dict[str, np.ndarray], 
                    targets: Dict[str, np.ndarray], 
                    node_mask: np.ndarray) -> Dict:
    """
    Compute comprehensive prediction metrics.
    
    Both predictions['tensors'] and targets['quadrupole'] are in 5-component format:
    [xy, xz, yz, (xx-yy)/2, zz]
    
    For reporting, we convert to 6-component Cartesian for interpretability.
    
    Args:
        predictions: Dict with 'scalars', 'vectors', 'tensors' as numpy/jax arrays
        targets: Dict with 'N', 'LI', 'dipole', 'quadrupole'
        node_mask: Boolean mask for valid nodes
        
    Returns:
        Dict with metrics for each property type
    """
    valid = np.array(node_mask)
    
    def r2(pred, target):
        """Coefficient of determination."""
        ss_res = np.sum((target - pred) ** 2)
        ss_tot = np.sum((target - np.mean(target)) ** 2)
        return 1 - ss_res / (ss_tot + 1e-10)
    
    def pearson(pred, target):
        """Pearson correlation coefficient."""
        if np.std(pred) < 1e-10 or np.std(target) < 1e-10:
            return 0.0
        return np.corrcoef(pred, target)[0, 1]
    
    def mae(pred, target):
        """Mean absolute error."""
        return np.mean(np.abs(pred - target))
    
    def rmse(pred, target):
        """Root mean squared error."""
        return np.sqrt(np.mean((pred - target) ** 2))
    
    metrics = {}
    
    # Scalars: N and LI
    for i, name in enumerate(['N', 'LI']):
        pred = np.array(predictions['scalars'][:, i][valid])
        tgt = np.array(targets[name][valid])
        metrics[name] = {
            'r': float(pearson(pred, tgt)),
            'r2': float(r2(pred, tgt)),
            'mae': float(mae(pred, tgt)),
            'rmse': float(rmse(pred, tgt))
        }
    
    # Vectors: Dipole
    pred_vec = np.array(predictions['vectors'][valid])
    tgt_vec = np.array(targets['dipole'][valid])
    
    metrics['dipole'] = {'components': {}}
    for i, c in enumerate(['X', 'Y', 'Z']):
        metrics['dipole']['components'][c] = {
            'r': float(pearson(pred_vec[:, i], tgt_vec[:, i])),
            'r2': float(r2(pred_vec[:, i], tgt_vec[:, i])),
            'mae': float(mae(pred_vec[:, i], tgt_vec[:, i])),
            'rmse': float(rmse(pred_vec[:, i], tgt_vec[:, i]))
        }
    
    # Cosine similarity
    pred_norm = np.linalg.norm(pred_vec, axis=1)
    tgt_norm = np.linalg.norm(tgt_vec, axis=1)
    cos_sim = np.sum(pred_vec * tgt_vec, axis=1) / (pred_norm * tgt_norm + 1e-10)
    metrics['dipole']['cosine_sim'] = {
        'mean': float(np.mean(cos_sim)),
        'median': float(np.median(cos_sim)),
        'std': float(np.std(cos_sim))
    }
    
    # Norm correlation
    metrics['dipole']['norm_correlation'] = {
        'r': float(pearson(pred_norm, tgt_norm)),
        'r2': float(r2(pred_norm, tgt_norm))
    }
    
    # Tensors: Quadrupole - both in 5-component format
    pred_t5 = np.array(predictions['tensors'][valid])
    tgt_t5 = np.array(targets['quadrupole'][valid])
    
    # Convert to 6-component Cartesian for interpretable reporting
    pred_t6 = comp5_to_cartesian(pred_t5)
    tgt_t6 = comp5_to_cartesian(tgt_t5)
    
    metrics['quadrupole'] = {'components': {}}
    for i, c in enumerate(['XX', 'XY', 'XZ', 'YY', 'YZ', 'ZZ']):
        metrics['quadrupole']['components'][c] = {
            'r': float(pearson(pred_t6[:, i], tgt_t6[:, i])),
            'r2': float(r2(pred_t6[:, i], tgt_t6[:, i])),
            'mae': float(mae(pred_t6[:, i], tgt_t6[:, i])),
            'rmse': float(rmse(pred_t6[:, i], tgt_t6[:, i]))
        }
    
    # Internal 5-component metrics
    comp_names_5 = ['xy', 'xz', 'yz', 'aniso', 'zz']
    metrics['quadrupole']['internal_components'] = {}
    for i, c in enumerate(comp_names_5):
        metrics['quadrupole']['internal_components'][c] = {
            'r': float(pearson(pred_t5[:, i], tgt_t5[:, i])),
            'r2': float(r2(pred_t5[:, i], tgt_t5[:, i]))
        }
    
    # Tensor cosine similarity (Frobenius inner product normalized)
    def frob_norm(t6):
        """Frobenius norm for symmetric tensor."""
        return np.sqrt(t6[:, 0]**2 + t6[:, 3]**2 + t6[:, 5]**2 + 
                      2*(t6[:, 1]**2 + t6[:, 2]**2 + t6[:, 4]**2))
    
    def tensor_dot(a, b):
        """Frobenius inner product for symmetric tensors."""
        return (a[:, 0]*b[:, 0] + a[:, 3]*b[:, 3] + a[:, 5]*b[:, 5] + 
               2*(a[:, 1]*b[:, 1] + a[:, 2]*b[:, 2] + a[:, 4]*b[:, 4]))
    
    pred_frob = frob_norm(pred_t6)
    tgt_frob = frob_norm(tgt_t6)
    tensor_cos = tensor_dot(pred_t6, tgt_t6) / (pred_frob * tgt_frob + 1e-10)
    
    metrics['quadrupole']['tensor_cosine'] = {
        'mean': float(np.mean(tensor_cos)),
        'median': float(np.median(tensor_cos)),
        'std': float(np.std(tensor_cos))
    }
    
    # Frobenius norm correlation
    metrics['quadrupole']['frobenius_norm'] = {
        'r': float(pearson(pred_frob, tgt_frob)),
        'r2': float(r2(pred_frob, tgt_frob))
    }
    
    # Average component R²
    metrics['quadrupole']['avg_r2'] = np.mean(
        [m['r2'] for m in metrics['quadrupole']['components'].values()]
    )
    
    return metrics


def print_metrics(metrics: Dict, include_internal: bool = False):
    """
    Pretty-print evaluation metrics.
    
    Args:
        metrics: Dict from compute_metrics()
        include_internal: Whether to print internal 5-component tensor metrics
    """
    print("=" * 70)
    print("PREDICTION QUALITY METRICS")
    print("=" * 70)
    
    print("\n📊 SCALARS:")
    for name in ['N', 'LI']:
        m = metrics[name]
        print(f"  {name:4s}: r = {m['r']:.4f}, R² = {m['r2']:.4f}, "
              f"MAE = {m['mae']:.4f}, RMSE = {m['rmse']:.4f}")
    
    print("\n🔷 DIPOLE (Vector):")
    for c in ['X', 'Y', 'Z']:
        m = metrics['dipole']['components'][c]
        print(f"  {c}: r = {m['r']:.4f}, R² = {m['r2']:.4f}")
    cs = metrics['dipole']['cosine_sim']
    print(f"  Cosine similarity: mean = {cs['mean']:.4f}, median = {cs['median']:.4f}")
    nc = metrics['dipole']['norm_correlation']
    print(f"  Norm correlation:  r = {nc['r']:.4f}, R² = {nc['r2']:.4f}")
    
    print("\n🔶 QUADRUPOLE (Tensor) - Cartesian Components:")
    for c in ['XX', 'XY', 'XZ', 'YY', 'YZ', 'ZZ']:
        m = metrics['quadrupole']['components'][c]
        print(f"  {c}: r = {m['r']:.4f}, R² = {m['r2']:.4f}")
    
    if include_internal:
        print("\n🔶 QUADRUPOLE (Tensor) - Internal 5-Component:")
        for c in ['xy', 'xz', 'yz', 'aniso', 'zz']:
            m = metrics['quadrupole']['internal_components'][c]
            print(f"  {c:5s}: r = {m['r']:.4f}, R² = {m['r2']:.4f}")
    
    print(f"\n  Avg Cartesian R²: {metrics['quadrupole']['avg_r2']:.4f}")
    tc = metrics['quadrupole']['tensor_cosine']
    print(f"  Tensor cosine sim: mean = {tc['mean']:.4f}, median = {tc['median']:.4f}")
    fn = metrics['quadrupole']['frobenius_norm']
    print(f"  Frobenius norm: r = {fn['r']:.4f}, R² = {fn['r2']:.4f}")


# =============================================================================
# Evaluation on Batches
# =============================================================================

def evaluate_model(model, 
                   batches: List, 
                   targets: List[Dict],
                   batch_size: int = 32,
                   verbose: bool = True) -> Dict:
    """
    Evaluate a model on a set of batches.
    
    Args:
        model: T-PaiNN model
        batches: List of padded ComplexBatch objects
        targets: List of per-molecule target dicts
        batch_size: Batch size (for indexing)
        verbose: Whether to print metrics
        
    Returns:
        Dict with aggregated metrics
    """
    all_preds = {'scalars': [], 'vectors': [], 'tensors': []}
    all_tgts = {'N': [], 'LI': [], 'dipole': [], 'quadrupole': []}
    all_masks = []
    
    for batch_idx, batch in enumerate(batches):
        start_idx = batch_idx * batch_size
        batch_indices = list(range(start_idx, min(start_idx + batch_size, len(targets))))
        batch_targets = prepare_batch_targets(targets, batch_indices, batch)
        
        predictions = model(batch)
        mask = np.array(predictions['x_mask'])
        
        all_preds['scalars'].append(np.array(predictions['scalars']))
        all_preds['vectors'].append(np.array(predictions['vectors']))
        all_preds['tensors'].append(np.array(predictions['tensors']))
        
        all_tgts['N'].append(np.array(batch_targets['N']))
        all_tgts['LI'].append(np.array(batch_targets['LI']))
        all_tgts['dipole'].append(np.array(batch_targets['dipole']))
        all_tgts['quadrupole'].append(np.array(batch_targets['quadrupole']))
        
        all_masks.append(mask)
    
    # Concatenate all batches
    concat_preds = {k: np.concatenate(v, axis=0) for k, v in all_preds.items()}
    concat_tgts = {k: np.concatenate(v, axis=0) for k, v in all_tgts.items()}
    concat_mask = np.concatenate(all_masks, axis=0)
    
    # Compute metrics
    metrics = compute_metrics(concat_preds, concat_tgts, concat_mask)
    
    if verbose:
        print_metrics(metrics)
    
    return metrics


def evaluate_singletask(model, 
                        batches: List, 
                        targets: List[Dict],
                        task_name: str,
                        batch_size: int = 32,
                        verbose: bool = True) -> Dict:
    """
    Evaluate a single-task model.
    
    Args:
        model: T-PaiNN model
        batches: List of padded ComplexBatch objects
        targets: List of per-molecule target dicts
        task_name: One of 'N', 'LI', 'dipole', 'quadrupole'
        batch_size: Batch size
        verbose: Whether to print metrics
        
    Returns:
        Dict with task-specific metrics
    """
    all_preds = []
    all_tgts = []
    all_masks = []
    
    for batch_idx, batch in enumerate(batches):
        start_idx = batch_idx * batch_size
        batch_indices = list(range(start_idx, min(start_idx + batch_size, len(targets))))
        batch_targets = prepare_batch_targets(targets, batch_indices, batch)
        
        predictions = model(batch)
        mask = np.array(predictions['x_mask'])
        
        if task_name == 'N':
            all_preds.append(np.array(predictions['scalars'][:, 0]))
            all_tgts.append(np.array(batch_targets['N']))
        elif task_name == 'LI':
            all_preds.append(np.array(predictions['scalars'][:, 1]))
            all_tgts.append(np.array(batch_targets['LI']))
        elif task_name == 'dipole':
            all_preds.append(np.array(predictions['vectors']))
            all_tgts.append(np.array(batch_targets['dipole']))
        elif task_name == 'quadrupole':
            all_preds.append(np.array(predictions['tensors']))
            all_tgts.append(np.array(batch_targets['quadrupole']))
        
        all_masks.append(mask)
    
    # Concatenate
    concat_preds = np.concatenate(all_preds, axis=0)
    concat_tgts = np.concatenate(all_tgts, axis=0)
    concat_mask = np.concatenate(all_masks, axis=0)
    valid = concat_mask
    
    def r2(pred, target):
        ss_res = np.sum((target - pred) ** 2)
        ss_tot = np.sum((target - np.mean(target)) ** 2)
        return 1 - ss_res / (ss_tot + 1e-10)
    
    def pearson(pred, target):
        if np.std(pred) < 1e-10 or np.std(target) < 1e-10:
            return 0.0
        return np.corrcoef(pred, target)[0, 1]
    
    if task_name in ['N', 'LI']:
        pred_valid = concat_preds[valid]
        tgt_valid = concat_tgts[valid]
        metrics = {
            'r': float(pearson(pred_valid, tgt_valid)),
            'r2': float(r2(pred_valid, tgt_valid)),
            'mae': float(np.mean(np.abs(pred_valid - tgt_valid))),
            'rmse': float(np.sqrt(np.mean((pred_valid - tgt_valid) ** 2)))
        }
    elif task_name == 'dipole':
        pred_valid = concat_preds[valid]
        tgt_valid = concat_tgts[valid]
        metrics = {'components': {}}
        for i, c in enumerate(['X', 'Y', 'Z']):
            metrics['components'][c] = {
                'r': float(pearson(pred_valid[:, i], tgt_valid[:, i])),
                'r2': float(r2(pred_valid[:, i], tgt_valid[:, i]))
            }
        # Cosine similarity
        pred_norm = np.linalg.norm(pred_valid, axis=1)
        tgt_norm = np.linalg.norm(tgt_valid, axis=1)
        cos_sim = np.sum(pred_valid * tgt_valid, axis=1) / (pred_norm * tgt_norm + 1e-10)
        metrics['cosine_sim'] = {'mean': float(np.mean(cos_sim)), 'median': float(np.median(cos_sim))}
    elif task_name == 'quadrupole':
        pred_t5 = concat_preds[valid]
        tgt_t5 = concat_tgts[valid]
        pred_t6 = comp5_to_cartesian(pred_t5)
        tgt_t6 = comp5_to_cartesian(tgt_t5)
        metrics = {'components': {}}
        for i, c in enumerate(['XX', 'XY', 'XZ', 'YY', 'YZ', 'ZZ']):
            metrics['components'][c] = {
                'r': float(pearson(pred_t6[:, i], tgt_t6[:, i])),
                'r2': float(r2(pred_t6[:, i], tgt_t6[:, i]))
            }
        metrics['avg_r2'] = np.mean([m['r2'] for m in metrics['components'].values()])
    
    if verbose:
        print(f"\n{task_name} Metrics:")
        if task_name in ['N', 'LI']:
            print(f"  r = {metrics['r']:.4f}, R² = {metrics['r2']:.4f}, "
                  f"MAE = {metrics['mae']:.4f}, RMSE = {metrics['rmse']:.4f}")
        elif task_name == 'dipole':
            for c in ['X', 'Y', 'Z']:
                m = metrics['components'][c]
                print(f"  {c}: r = {m['r']:.4f}, R² = {m['r2']:.4f}")
            print(f"  Cosine sim: mean = {metrics['cosine_sim']['mean']:.4f}")
        elif task_name == 'quadrupole':
            for c in ['XX', 'XY', 'XZ', 'YY', 'YZ', 'ZZ']:
                m = metrics['components'][c]
                print(f"  {c}: r = {m['r']:.4f}, R² = {m['r2']:.4f}")
            print(f"  Avg R² = {metrics['avg_r2']:.4f}")
    
    return metrics


def benchmark_model(model, batch, num_warmup=5, num_runs=50):
    """
    Benchmark a model's forward pass throughput.
    
    Returns:
        dict with 'mean_time_ms', 'std_time_ms', 'throughput' (runs/s)
    """
    @nnx.jit
    def forward(model):
        return model(batch)
    
    # Warmup (compile + fill caches)
    for _ in range(num_warmup):
        out = forward(model)
        jax.block_until_ready(out)
    
    # Timed runs
    times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        out = forward(model)
        jax.block_until_ready(out)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    
    times = np.array(times)
    return {
        'mean_time_ms': float(np.mean(times) * 1000),
        'std_time_ms': float(np.std(times) * 1000),
        'throughput': float(1.0 / np.mean(times)),
    }


def count_flops(model, batch):
    """
    Count FLOPs for a single forward pass using JAX's HLO cost analysis.
    
    Uses nnx.split/merge to make the model compatible with jax.jit tracing.
    """
    graphdef, state = nnx.split(model)
    
    def forward_fn(state):
        mdl = nnx.merge(graphdef, state)
        return mdl(batch)
    
    try:
        lowered = jax.jit(forward_fn).lower(state)
        compiled = lowered.compile()
        cost = compiled.cost_analysis()
        
        # cost_analysis() returns a dict (not a list) on this JAX version
        if isinstance(cost, dict):
            flops = cost.get('flops', 0)
        elif isinstance(cost, (list, tuple)) and len(cost) > 0:
            flops = cost[0].get('flops', 0)
        else:
            flops = 0
        
        flops = int(flops)
            
    except Exception as e:
        print(f"  cost_analysis failed ({type(e).__name__}: {e})")
        flops = 0
    
    return flops


def format_flops(flops):
    """Format FLOP count in human-readable form."""
    if flops == 0:
        return "N/A"
    elif flops >= 1e12:
        return f"{flops/1e12:.2f} TFLOPs"
    elif flops >= 1e9:
        return f"{flops/1e9:.2f} GFLOPs"
    elif flops >= 1e6:
        return f"{flops/1e6:.2f} MFLOPs"
    elif flops >= 1e3:
        return f"{flops/1e3:.2f} KFLOPs"
    else:
        return f"{flops:.0f} FLOPs"


def format_time(ms):
    """Format time in ms."""
    if ms >= 1000:
        return f"{ms/1000:.2f} s"
    elif ms >= 1:
        return f"{ms:.2f} ms"
    else:
        return f"{ms*1000:.1f} µs"