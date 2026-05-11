<h1><p align="center">QT-Net</p></h1>

QT-Net is a Python package for learning per-atom properties from the quantum theory of atoms in molecules (QTAIM) — and any other atomic-level scalar/vector/tensor target — from molecular geometry. Molecules are represented as *topological cell complexes* (0-cells for atoms, 1-cells for edges, 2-cells for higher-order relations), and the models perform message passing across this hierarchy.

The package is built around two concerns that go beyond raw predictive performance:

1. **Clustering of atomic environments** into chemically meaningful groups, so that test performance can be reported per-cluster rather than as a single pooled metric. This makes it possible to hold out specific atomic environments at evaluation time and quantify how each architecture generalises to them.
2. **Rigorous statistical model comparison** via repeated-measures ANOVA + Tukey HSD on repeat-level metrics, so that architecture rankings reflect statistically meaningful differences rather than fold-level noise.

QT-Net provides E(3)-equivariant JAX/Flax networks (which become SO(3)-equivariant when the optional `EdgeGeometryReminder` cross-product term is enabled), scalar GNNs that incorporate the same topological structure without enforcing equivariance, and classical baselines built on XGBoost with Morgan fingerprints or ChemProp message-passing networks.

---

## 🔭 What can QT-Net predict?

QT-Net targets two broad categories of properties.

**Atomic-level properties** are predicted per-atom and include electron density (N), localization index (LI), dipole moment components (µ_x, µ_y, µ_z), and quadrupole moment components (Q_xy, Q_xz, Q_yz, Q_aniso, Q_zz). These are derived from the AIMEl dataset of QTAIM-partitioned wavefunctions.

**Molecular-level properties** come from QM9 and include the isotropic polarizability (α), HOMO-LUMO gap, internal energy at 0 K (U0), and heat capacity (Cv). Both blind (topology-only) and informed (topology + per-atom QTAIM features) variants of the molecular models are supported.

---

## 🤖 Models

### Atomic models

These are the architectures discussed in the paper, exposed via `scripts/atomic/train_multitask.py --model-type ...`:

| Identifier | Full name | Description |
|---|---|---|
| `EGNN` | EquivariantGNN | E(3)-equivariant GNN over the cell complex |
| `EGNF` | EquivariantGNN_FC | Fully-connected variant of `EGNN` |
| `SGNN` | ScalarGNN | Topology-aware, non-equivariant scalar GNN |
| `SGNF` | ScalarGNN_FC | Fully-connected variant of `SGNN` |
| `SGN2` | SGNN_v2 | Updated scalar architecture |
| `SNF2` | SGNN_v2_FC | Fully-connected variant of `SGN2` |

Equivariant models are E(3)-equivariant by default; enabling the `EdgeGeometryReminder` term (which uses a cross product) reduces them to SO(3)-equivariance. Rotation augmentation during training is available.

The training script registers a few additional architectures (`ETNN`, `STNN`, `EGN2`, `EGNX`, `SBAS`, `SBAE`, …) for experimentation; they are not part of the paper's evaluation.

### Molecular models

The molecular pipeline focuses on `ScalarGNNMolecular`, available through `scripts/molecular/jax/run_hpo_molecular.py` and `scripts/molecular/jax/train_molecular_jax.py`. It can be trained as **blind** (topology only) or **informed** (`--use-atom-features`, injects per-atom N/LI/µ/Q descriptors as additional node features).

> [!NOTE]
> Equivariant models are currently only supported for atomic-level targets.

---

## ⚙️ Installation

QT-Net uses [`uv`](https://github.com/astral-sh/uv) for environment and dependency management.

> [!NOTE]
> QT-Net requires **Python 3.10 or later**. Make sure `uv` is installed on your system (`pip install uv` or via the [official installer](https://docs.astral.sh/uv/getting-started/installation/)) before proceeding.

Start by cloning the repository and entering the project directory:

```bash
git clone <repo-url>
cd QT-Net
```

Then create a virtual environment and install the package with its dependencies in one step:

```bash
uv venv
uv pip install -e ".[dev]"
```

This will install all required dependencies, including JAX, Flax, ChemProp, XGBoost, RDKit, and Optuna, along with development tools like Jupyter and Matplotlib. The `-e` flag installs the package in editable mode so that local changes to the source are reflected immediately without reinstalling.

> [!WARNING]
> JAX GPU support requires a separate installation step that depends on your CUDA version. After the base install, follow the [official JAX GPU installation guide](https://jax.readthedocs.io/en/latest/installation.html) and install the appropriate `jax[cuda12]` or `jax[cuda11_pip]` variant. The default install pulls CPU-only JAX. We recommend to read the JAX documentation for taking full advantage of these models.


---

## 🗄️ Data Preparation

The script `scripts/setup_data.sh` decompresses all the necessary data files for replicating
the experiments in the paper.

Before training, you need to precompute topological representations from the raw datasets. These precomputation steps cache the cell complexes and geometric encodings to disk so that training runs do not need to recompute them every epoch.

Raw and curated data live in `data_curation/`:

```
data_curation/
├── atomic/
│   ├── data/aimel_dataset_with_components.csv
│   └── cluster_analysis/   # train/val/test split + clustering of atomic environments
└── molecular/
    ├── aimel_clustered_molecular.pkl   # AIM-labelled molecules used for QT-Net training
    ├── qm9_filtered.pkl                # full QM9 with cleaned features
    └── qm9_inferred.pkl                # qm9_filtered + per-atom AIM properties (produced by inference pipeline)
```

### ⚛️ Atomic data

The atomic dataset is sourced from the .sumviz files of AIMEl, a collection of QTAIM-partitioned atomic properties: https://zenodo.org/records/11406726

The `data_curation/atomic/cluster_analysis/` directory contains the notebook (`AIMEl_csv_to_datasets.ipynb`) and helpers (`atomic_env_split.py`, `cluster_cooccurrence.py`) that turn the raw CSV into a clustered train/val/test split, with held-out cluster labels (e.g. `H_10`, `C_11`, `N_13`, `O_10`) used as out-of-distribution probes during evaluation.

Once the raw data is in place, generate the topology cache by running:

```bash
python scripts/atomic/precompute_complexes.py --cutoff 8.0 --max-neighbors 12
```

This produces `data_curation/atomic/precomputed_complexes.pkl`, which contains the cell complexes along with RBF-encoded interatomic distances for each molecule in the dataset. Coordinates are stored in Bohr units internally and converted to Ångström during loading in `qtnet.data_utils`.

### 🧪 Molecular data

The molecular dataset is loaded from `data_curation/molecular/aimel_clustered_molecular.pkl`, also obtained from `AIMEl_csv_to_datasets.ipynb`. Two variants of the topology cache are generated — one for the *blind* models (topology only, no atomic features) and one for the *informed* models (topology with atomic QTAIM features):

```bash
python scripts/molecular/jax/pregenerate_batches_molecular.py
```

This outputs `data_curation/molecular/precomputed_blind.pkl` and `data_curation/molecular/precomputed_gta.pkl`. Cross-validation splits are scaffold-grouped using Bemis-Murcko scaffolds to avoid data leakage across structurally similar molecules.

> [!NOTE]
> Precomputation can be memory-intensive for large datasets. If you run into out-of-memory errors, reduce batch size and/or number of nearest neighbors.

---

## 🏋️ Training

QT-Net uses a repeated 5×5 scaffold-based cross-validation scheme, yielding 25 folds in total (indexed 0–24, with `repeat = fold // 5`). The typical workflow is to first run hyperparameter optimisation (HPO) on fold 0, then train all folds using the best configuration found.

The training scripts read their hyperparameters from a JSON file in `optimal_hyperparams/` by default (override with `--optuna-file` or, equivalently, the relevant `--*-file` flag). Every training run also writes a `config.json` next to its checkpoints containing the full set of hyperparameters and the random seed actually used, so any individual fold can be reproduced from that file alone.

### ⚛️ Atomic models

Run HPO for a given model architecture:

```bash
python scripts/atomic/run_hpo.py --model-name ScalarGNN --n-trials 100
```

You can then train a specific fold using the saved configuration:

```bash
python scripts/atomic/train_multitask.py --model-type SGNN --fold 0 --epochs 500
```

Checkpoints and predictions are written to `experiments/atomic/<model_type>/fold_<n>/`:

```
checkpoints/    # periodic + best + final Orbax checkpoints
loss/           # loss_history.json
config.json     # full kwargs, seed, lr, wd, n_params
val_preds.pkl   # un-regularized predictions on the val split
test_preds.pkl  # un-regularized predictions on the held-out test set
```

Training uses the AdamW optimizer with ReduceOnPlateau learning rate scheduling and Orbax for checkpoint management.

### 🧪 Molecular models

The molecular training pipeline mirrors the atomic one. Run HPO first:

```bash
python scripts/molecular/jax/run_hpo_molecular.py \
    --model-name ScalarGNNMolecular --use-atom-features --n-trials 50
```

Train a fold × fraction with:

```bash
# informed (default)
python scripts/molecular/jax/train_molecular_jax.py --fold 0 --fraction 1.0

# blind
python scripts/molecular/jax/train_molecular_jax.py --fold 0 --fraction 0.1 --blind
```

The `--fraction` argument controls the fraction of the **training split of the current fold** that is actually used for training (val and test splits are unaffected), enabling data-efficiency experiments. Outputs land in `experiments/molecular/<variant>/fold_<n>/frac_<f>/` with the same `checkpoints/`, `loss/`, `config.json`, `val_preds.pkl`, `test_preds.pkl` layout as the atomic models.

### 📊 Baselines

XGBoost and ChemProp baselines follow the same HPO-then-train pattern:

```bash
# XGBoost
python scripts/molecular/xgboost/run_hpo_xgboost.py
python scripts/molecular/xgboost/train_xgboost.py
python scripts/molecular/xgboost/predict_xgboost.py

# ChemProp
python scripts/molecular/chemprop/run_hpo_chemprop.py
python scripts/molecular/chemprop/train_chemprop.py
```

> [!WARNING]
> XGBoost and ChemProp baselines are under active development and are not fully integrated in QT-Net yet.

---

## 🔮 QTAIM Inference Pipeline

To run the *informed* molecular variant on QM9 molecules outside the AIM-labelled subset, QT-Net imputes per-atom AIM properties via an ensemble of atomic models. The full pipeline lives in `scripts/inference/`:

```bash
# 1) Build the atomic-side topology cache for AIMEl
python scripts/inference/precompute_complexes_aimel.py --cutoff 8.0 --max-neighbors 12

# 2) Train 5 ensemble members (one per --member-idx; suited for SLURM job arrays)
for i in 0 1 2 3 4; do
  python scripts/inference/train_QTNet_ensemble.py \
      --model-type SGN2 --ensemble-label qtnet_v1 --member-idx $i
done

# 3) Build the QM9 atomic-side topology cache
python scripts/inference/precompute_complexes_QM9.py

# 4) Run ensemble inference → qm9_inferred.pkl (per-atom N, LI, Mu, Q + ensemble stds)
python scripts/inference/infer_QTAIM_QM9.py --model-type SGN2

# 5) Build molecular complexes for QM9 (blind + informed)
python scripts/inference/precompute_molecular_QM9.py

# 6) Predict molecular properties on QM9 with trained molecular ensembles
python scripts/inference/predict_from_inferred.py \
    --variants informed blind --fractions 0.1 0.5 1.0
```

Step 6 selects 5 folds per `(variant, fraction)` — one per repeat, the best by `best_val_so_far` from `loss_history.json` — and writes ensemble-averaged predictions to `data_curation/molecular/qm9_molecular_preds.pkl` with `{variant}_{fraction}_pred_{prop}` and `{variant}_{fraction}_std_{prop}` columns.

Ensemble outputs land in `experiments/inference/<model_type>/model_<i>/` (`config.json`, `stats.json`, `checkpoints/`, `loss/`) — mirroring the atomic experiments layout but indexed by ensemble member instead of fold. Per-atom denormalisation must use **each member's own** `stats.json`; predictions are denormalised *before* averaging across members.

An auxiliary script, `scripts/inference/molecule_dipoles_from_inferred.py`, recomputes molecular dipole moments from the inferred per-atom dipoles for downstream analysis.

---

## 📈 Evaluation and Analysis

The `analysis/` directory contains the libraries and notebooks used to aggregate fold-level results across the 25 folds, run RM-ANOVA + Tukey HSD model comparisons, and produce the figures and tables in the paper.

```
analysis/
├── atomic/
│   ├── result_analysis.py       # canonical RM-ANOVA + Tukey HSD; LaTeX tables
│   ├── tmp_plot_cell_updated.py # forest plots, parity plots, box plots
│   ├── analyze_CV_augmented.ipynb
│   └── analyze_CV_nonaug.ipynb
├── molecular/
│   ├── analysis_molecular.py
│   └── analyze_molecular_training_fracs.ipynb
└── inference/
    ├── analyse_inference.py
    └── inference_results.ipynb
```

The 5×5 CV convention used throughout: 25 folds → repeat-level mean (5 folds per repeat) → RM-ANOVA on the 5 repeat-level means per model. See `analysis/atomic/result_analysis.py` for the canonical Tukey HSD implementation; the molecular and inference analyses reuse it.


---

## 🗂️ Project Structure

```
QT-Net/
├── src/qtnet/             # Core package: models, data utilities, topological representations
│   ├── jax_models/        # JAX/Flax model families (equivariant, scalar, molecular, inference)
│   ├── chemprop_models/
│   ├── xgb_models/
│   └── data_utils.py
├── scripts/               # Training, HPO, and inference entry points
│   ├── atomic/
│   ├── molecular/{jax,xgboost,chemprop}/
│   └── inference/
├── data_curation/         # Datasets and preprocessing notebooks
│   ├── atomic/
│   └── molecular/
├── analysis/              # Analysis libraries and notebooks
│   ├── atomic/
│   ├── molecular/
│   └── inference/
├── experiments/           # Output: logs, checkpoints, predictions
│   ├── atomic/<model>/fold_<n>/
│   ├── molecular/<variant>/fold_<n>/frac_<f>/
│   └── inference/<model_type>/model_<i>/
└── optimal_hyperparams/   # Saved Optuna HPO results (gitignored; created on first HPO run)
```

---

## 🤝 Contributing

Contributions are welcome. Please follow PEP 8 style conventions and annotate all public functions and classes with type hints. Use the `logging` module rather than `print` statements for any diagnostic output. Before opening a pull request, run HPO locally on at least fold 0 to verify that a new model or feature trains without errors.

<!-- TODO: add information on how to run linting/formatting (e.g., ruff, black) and whether a pre-commit config is provided -->

---

## 📄 License

QT-Net is released under the MIT License.

---

## 📚 Citation

If you use QT-Net in your research, please cite:

<!-- TODO: replace with full citation once the paper is published -->
```
[Citation placeholder — paper under preparation]
```
