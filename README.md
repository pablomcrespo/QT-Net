<h1><p align="center">QT-Net</p></h1>

QT-Net (pronounced *cutie-net* — we didn't choose the name, but we're not complaining) is a Python package for topological data analysis (TDA) and machine learning on molecular systems. It bridges the gap between classical cheminformatics and modern geometric deep learning by representing molecules as *topological cell complexes* — 0-cells for atoms, 1-cells for bonds, and 2-cells for higher-order triplet interactions — and using these richer representations as the basis for property prediction.

The package supports a spectrum of model families: JAX-based SO(3)-equivariant neural networks for rotationally invariant predictions, scalar graph neural networks (GNNs) that incorporate topological features without enforcing equivariance, and classical baselines built on XGBoost with Morgan fingerprints or ChemProp message-passing networks. This makes QT-Net useful both as a research platform for exploring topological representations and as a practical benchmarking toolkit.

---

## 🔭 What can QT-Net predict?

QT-Net targets two broad categories of properties.

**Atomic-level properties** are predicted per-atom and include electron density (N), localization index (LI), dipole moment components (µ_x, µ_y, µ_z), and quadrupole moment components (Q_xy, Q_xz, Q_yz, Q_aniso, Q_zz). These are derived from the AIMEl dataset of QTAIM-partitioned wavefunctions.

**Molecular-level properties** come from the QM9 benchmark and include the isotropic polarizability (α), HOMO-LUMO gap, internal energy at 0 K (U0), and heat capacity (Cv). Both blind (topology-only) and informed (topology + atomic features) variants of the molecular models are supported.

---

## 🤖 Models

QT-Net provides several model families, each with a short identifier used in training scripts:

| Identifier | Full name | Description |
|---|---|---|
| `ETNN` / `TPaiNN` | Topological PaiNN | SO(3)-equivariant GNN, suitable for vector/tensor targets |
| `EGNN` / `EquivariantGNN` | Equivariant GNN | Alternative equivariant architecture |
| `SGNN` / `ScalarGNN` | Scalar GNN | Topology-aware, non-equivariant |
| `STNN` / `ScalarTPaiNN` | Scalar Topological PaiNN | Scalar variant of TPaiNN |
| `SBAS` / `ScalarBaseline` | Scalar Baseline | Node-only features, no edge messages |
| `SBAE` / `ScalarBaselineEdges` | Scalar Baseline + Edges | Adds edge features to the baseline |

Equivariant models apply SO(3) rotation augmentation during training and use geometric precomputations (gyration tensors, relative positions, radial basis function encodings, and angular encodings) stored in the topology cache. Scalar models skip augmentation but still benefit from the topological cell complex structure.

> [!NOTE]
> Equivariant models are currently only supported for atomic-level targets. For molecular QM9 properties, use the `ScalarGNNMolecular` or `ScalarTPaiNNMolecular` variants.

---

## ⚙️ Installation

QT-Net uses [`uv`](https://github.com/astral-sh/uv) for environment and dependency management.

> [!NOTE]
> QT-Net requires **Python 3.10 or later**. Make sure `uv` is installed on your system (`pip install uv` or via the [official installer](https://docs.astral.sh/uv/getting-started/installation/)) before proceeding.

Start by cloning the repository and entering the project directory:

```bash
git clone https://github.com/pablomcrespo/QT-Net.git
cd QT-Net
```

Then create a virtual environment and install the package with its dependencies in one step:

```bash
uv venv
uv pip install -e ".[dev]"
```

This will install all required dependencies, including JAX, Flax, ChemProp, XGBoost, RDKit, and Optuna, along with development tools like Jupyter and Matplotlib. The `-e` flag installs the package in editable mode so that local changes to the source are reflected immediately without reinstalling.

> [!WARNING]
> JAX GPU support requires a separate installation step that depends on your CUDA version. After the base install, follow the [official JAX GPU installation guide](https://jax.readthedocs.io/en/latest/installation.html) and install the appropriate `jax[cuda12]` or `jax[cuda11_pip]` variant. The default install pulls CPU-only JAX.

> [!NOTE]
> If you are running on an HPC cluster with SLURM, you may need to load environment modules (e.g., CUDA, cuDNN) before activating the virtual environment. <!-- TODO: add cluster-specific setup notes, module load commands, and PYTHONPATH configuration if needed -->

---

## 🗄️ Data Preparation

Before training, you need to precompute topological representations from the raw datasets. These precomputation steps cache the cell complexes and geometric encodings to disk so that training runs do not need to recompute them every epoch.

### ⚛️ Atomic data

The atomic dataset is sourced from AIMEl, a collection of QTAIM-partitioned atomic properties. <!-- TODO: add link or citation for AIMEl dataset and instructions on where to download/place the raw files -->

Once the raw data is in place, generate the topology cache by running:

```bash
python scripts/atomic/precompute_complexes.py
```

This produces `data/precomputed_complexes.pkl`, which contains the cell complexes along with RBF-encoded interatomic distances for each molecule in the dataset. Coordinates are stored in Bohr units internally and converted to Ångström during loading in `data_utils.py`.

### 🧪 Molecular data

The molecular dataset is loaded from `data/aimel_clustered_molecular.pkl`. <!-- TODO: describe how this file is produced or where to obtain it --> Two variants of the topology cache are generated — one for the *blind* models (topology only, no atomic features) and one for the *informed* models (topology with atomic QTAIM features):

```bash
python scripts/molecular/jax/pregenerate_batches_molecular.py
```

This outputs `precomputed_blind.pkl` and `precomputed_gta.pkl`. Cross-validation splits are scaffold-grouped using Bemis-Murcko scaffolds to avoid data leakage across structurally similar molecules.

> [!NOTE]
> Precomputation can be memory-intensive for large datasets. If you run into out-of-memory errors, <!-- TODO: describe chunking or batching options if available -->.

---

## 🏋️ Training

QT-Net uses a repeated 5×5 scaffold-based cross-validation scheme, yielding 25 folds in total (indexed 0–24). The typical workflow is to first run hyperparameter optimisation (HPO) on fold 0, then train all folds using the best configuration found.

### ⚛️ Atomic models

Run HPO for a given model architecture:

```bash
python scripts/atomic/run_hpo.py --model-name TPaiNN --n-trials 100
```

The best hyperparameter configuration is saved to `optimal_hyperparams/<model>_optuna.json`. You can then train a specific fold using that configuration:

```bash
python scripts/atomic/train_multitask.py --model-type ETNN --fold 0 --epochs 500
```

Checkpoints and predictions are written to `experiments/atomic/<model>/fold_<n>/`, with `val_preds.pkl` and `test_preds.pkl` containing the held-out predictions for that fold. Training uses the AdamW optimizer with ReduceOnPlateau learning rate scheduling and Orbax for checkpoint management.

### 🧪 Molecular models

The molecular training pipeline mirrors the atomic one. Run HPO first:

```bash
python scripts/molecular/jax/run_hpo_molecular.py --model-name ScalarTPaiNNMolecular --n-trials 50
```

HPO outputs are saved as `optimal_hyperparams/<model>_blind_optuna.json` or `informed_optuna.json` depending on whether atom features are used. Train a fold with:

```bash
python scripts/molecular/jax/train_molecular_jax.py --model-type STNN_2 --fold 0 --fraction 1.0
```

The `--fraction` argument controls what fraction of the training data is used, which is useful for data efficiency experiments. <!-- TODO: clarify whether fraction applies to the full dataset or just the training split of the current fold -->

### 📊 Baselines

XGBoost and ChemProp baselines follow the same HPO-then-train pattern:

```bash
# XGBoost
python scripts/molecular/xgboost/run_hpo_xgboost.py
python scripts/molecular/xgboost/train_xgboost.py

# ChemProp
python scripts/molecular/chemprop/run_hpo_chemprop.py
python scripts/molecular/chemprop/train_chemprop.py
```

> [!NOTE]
> ChemProp baselines use its internal MPNN architecture and are trained using ChemProp's own trainer, not the JAX/Flax loop used by the GNN models. <!-- TODO: clarify whether ChemProp models also use the topological complex as input or operate on raw SMILES -->

> [!WARNING]
> XGBoost and ChemProp baselines are under active development and are not fully integrated in QT-Net yet.

---

## 🔮 Inference and Ensembling

Once models are trained across all 25 folds, you can run inference on new molecules and ensemble the fold predictions for more robust estimates.

For atomic properties:

```bash
python scripts/inference/infer_QTAIM_QM9.py  # <!-- TODO: document required input format -->
```

For molecular properties, predictions from individual folds can be aggregated with:

```bash
python scripts/inference/predict_from_inferred.py
python scripts/inference/train_QTNet_ensemble.py
```

<!-- TODO: describe the ensembling strategy (mean, weighted average, stacking?) and the expected input/output format for inference scripts -->

---

## 📈 Evaluation and Analysis

After training, the `analyze_CV_experiments.ipynb` notebook aggregates results across all 25 folds and computes summary statistics. The `visualize_molecules.ipynb` notebook provides tools for inspecting the cell complex representations of individual molecules.

> [!WARNING]
> The automated test suite (`pytest tests/`) is currently empty. There are no unit or integration tests in place. <!-- TODO: add tests covering at minimum data loading, complex precomputation, and a single forward pass for each model family -->

---

## 🗂️ Project Structure

```
QT-Net/
├── src/qtnet/              # Core package: models, data utilities, topological representations
├── scripts/            # Training, HPO, and inference entry points
│   ├── atomic/
│   ├── molecular/
│   │   ├── jax/
│   │   ├── xgboost/
│   │   └── chemprop/
│   └── inference/
├── data_curation/      # Preprocessing notebooks and scripts
├── notebooks/          # Analysis and visualisation notebooks
├── experiments/        # Output directory: logs, checkpoints, predictions
└── optimal_hyperparams/ # Saved Optuna HPO results
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
