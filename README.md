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

The molecular pipeline focuses on `ScalarGNNMolecular`, available through `scripts/molecular/jax/run_hpo_molecular.py` and `scripts/molecular/jax/train_molecular_jax.py`. It can be trained as **blind** (topology only) or **informed** (`--use-atom-features`, injects per-atom N/LI/µ/Q descriptors as additional node features). The environment-holdout study of the paper uses `GeoScalarGNNMolecular` instead, where both variants see geometry through the geometric filters (see [Environment-Holdout Molecular Study](#-environment-holdout-molecular-study)).

> [!NOTE]
> Equivariant models are currently only supported for atomic-level targets.

---

## ⚙️ Installation

QT-Net uses [`uv`](https://github.com/astral-sh/uv) for environment and dependency management.

> [!NOTE]
> QT-Net requires **Python 3.11 or later**. Make sure `uv` is installed on your system (`pip install uv` or via the [official installer](https://docs.astral.sh/uv/getting-started/installation/)) before proceeding.

Start by cloning the repository and entering the project directory:

```bash
git clone https://github.com/pablomcrespo/QT-Net.git
cd QT-Net
```

Then create a virtual environment with the locked dependencies and activate it:

```bash
uv sync --group envclust
source .venv/bin/activate
```

This installs the versions pinned in `uv.lock`, including JAX, Flax, ChemProp, XGBoost, RDKit, and Optuna, along with development tools like Jupyter and Matplotlib, and installs the package in editable mode so that local changes to the source are reflected immediately. The `envclust` group adds the SOAP and clustering packages (`ase`, `dscribe`, `hdbscan`) used by the environment classification, the 6-Bohr distances and the leakage analysis.

> [!WARNING]
> JAX GPU support requires a separate installation step that depends on your CUDA version. After the base install, follow the [official JAX GPU installation guide](https://jax.readthedocs.io/en/latest/installation.html) and install the `jax[cuda12]` variant at the locked version: `uv pip install "jax[cuda12]==0.6.2"`. A later `uv sync` removes the plugin, so run this step last. The default install pulls CPU-only JAX. We recommend to read the JAX documentation for taking full advantage of these models.


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
│       ├── qm9_env_labels_6b.pkl.gz    # whole-QM9 6-Bohr environment label of every atom (environment-holdout study)
│       └── qm9_env_folds_6b.json       # the five environment folds of that study
└── molecular/
    ├── aimel_clustered_molecular.pkl   # AIM-labelled molecules used for QT-Net training
    ├── qm9_filtered.pkl                # QM9 molecules outside AIMEl (98,096) with cleaned features
    ├── qm9_inferred.pkl                # qm9_filtered + per-atom AIM properties (produced by inference pipeline)
    ├── qm9_chemotypes.pkl.gz           # Morgan-fingerprint chemotype of every QM9 molecule (environment-holdout study)
    ├── qm9_classification_order.json.gz  # molecule order (+ 23 QM9 conformers) the two whole-QM9 classifications used
    └── GeoScalarGNNMolecular_hp.json   # hyperparameters of the environment-holdout models
```

The curated files derive from QM9 ([Ramakrishnan et al., *Scientific Data* 1, 140022, 2014](https://doi.org/10.1038/sdata.2014.22)): of its 133,885 molecules, the 3,054 that failed the original geometry check are removed (130,831), then the 1,923 containing fluorine (128,908). `qm9_filtered.pkl` is that set minus AIMEl, in Bohr, and `qm9_classification_order.json.gz` records the row order of the fluorine-free set.

### ⚛️ Atomic data

The atomic dataset is sourced from the .sumviz files of AIMEl, a collection of QTAIM-partitioned atomic properties: https://zenodo.org/records/11406726

The `data_curation/atomic/cluster_analysis/` directory contains the notebook (`AIMEl_csv_to_datasets.ipynb`) and helpers (`atomic_env_split.py`, `cluster_cooccurrence.py`) that turn the raw CSV into a clustered train/val/test split, with held-out cluster labels (e.g. `H_10`, `C_11`, `N_13`, `O_10`) used as out-of-distribution probes during evaluation. The notebook documents how the curated files were made; it is not a runnable step, and `scripts/setup_data.sh` provides its outputs.

Once the raw data is in place, generate the topology cache by running:

```bash
python scripts/atomic/precompute_complexes.py --cutoff 8.0 --max-neighbors 12
```

This produces `data_curation/atomic/precomputed_complexes_cutoff8.0_max12.pkl` and `precomputed_test_cutoff8.0_max12.pkl`, which contain the cell complexes along with RBF-encoded interatomic distances for each molecule in the dataset. Coordinates are stored in Bohr units internally and converted to Ångström during loading in `qtnet.data_utils`.

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

The atomic models use a repeated 5×5 scaffold-based cross-validation scheme, yielding 25 folds in total (indexed 0–24, with `repeat = fold // 5`). The typical workflow is to first run hyperparameter optimisation (HPO) on fold 0, then train all folds using the best configuration found.

The training scripts read their hyperparameters from a JSON file in `optimal_hyperparams/` by default (override the directory with `--optuna-dir`). The repository ships the files of the five models in the table below, taken from their `config.json`. Every training run also writes a `config.json` next to its checkpoints containing the full set of hyperparameters and the random seed actually used, so any individual fold can be reproduced from that file alone.

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

The analysis notebooks read the paper's model names, so move each model's outputs from `<model_type>/` to the directory named in the first column. Graph and batch flags are those of the released `config.json` files. EGFC and EGNN ran in the first round of experiments with the unweighted per-component loss, which only EGNN folds 1 and 22 record in their `config.json`:

| Paper | `--model-type` | Graph flags | Loss flags | `--batch-size` | `--epochs` | `--rotate-every` |
|---|---|---|---|---|---|---|
| SG-8-12 | `SGN2` | `--cutoff 8 --max-neighbors 12` | | 256 | 2000 | 1 |
| SG-8-5 | `SGNN_final` | `--cutoff 8 --max-neighbors 5` | | 2048 | 2000 | 1 |
| SFC2 | `SNF2` | `--fully-connected` | | 256 | 2000 | 1 |
| EGFC | `EGNF` | `--fully-connected` | `--no-element-weights --mse-loss` | 128 | 1500 | 0 |
| EGNN | `EGNN` | `--cutoff 5.25 --max-neighbors 5` | `--no-element-weights --mse-loss` | 512 | 3000 | 0 |

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
      --model-type SGN2 --ensemble-label scalar_qtnet --member-idx $i
done

# 3) Build the QM9 atomic-side topology cache
python scripts/inference/precompute_complexes_QM9.py

# 4) Run ensemble inference → qm9_inferred.pkl (per-atom N, LI, Mu, Q + ensemble stds)
python scripts/inference/infer_QTAIM_QM9.py --model-type SGN2

# 5) Build molecular complexes for QM9 (blind + informed)
python scripts/inference/precompute_molecular_QM9.py

# 6) Predict molecular properties on QM9 with trained molecular ensembles
python scripts/inference/predict_from_inferred.py \
    --variants informed blind --fractions 0.01 0.05 0.1 1.0
```

Step 6 selects 5 folds per `(variant, fraction)` — one per repeat, the best by `best_val_so_far` from `loss_history.json` — and writes ensemble-averaged predictions to `data_curation/molecular/qm9_molecular_preds.pkl` with `{variant}_{fraction}_pred_{prop}` and `{variant}_{fraction}_std_{prop}` columns.

Ensemble outputs land in `experiments/inference/<model_type>/model_<i>/` (`config.json`, `stats.json`, `checkpoints/`, `loss/`) — mirroring the atomic experiments layout but indexed by ensemble member instead of fold. Per-atom denormalisation must use **each member's own** `stats.json`; predictions are denormalised *before* averaging across members.

An auxiliary script, `scripts/inference/molecule_dipoles_from_inferred.py`, recomputes molecular dipole moments from the inferred per-atom dipoles for downstream analysis.

---

## 🧭 Environment-Holdout Molecular Study

Do molecular models generalise because they see atomic descriptors, or because they have seen the atomic environments? This study holds out **Morgan-fingerprint chemotypes** and **atomic environments** at the same time, and compares a **blind** and an **informed** (N/LI/µ/Q) `GeoScalarGNNMolecular`. Both variants see explicit geometry, so an informed advantage cannot come from geometry implicit in the atomic properties. Models are trained on AIMEl (ground-truth QTAIM) and then deployed, without retraining, on the 98,096 QM9 molecules outside AIMEl.

Both axes are whole-QM9 classifications shipped in `data_curation/`:

| axis | how | file |
|---|---|---|
| atomic environments | SOAP (6 Bohr) → per-element PCA (99 % variance) → HDBSCAN; 317 labels | `qm9_env_labels_6b.pkl.gz` |
| environment folds | 5 co-occurrence-closure holdouts: an N or O seed plus the environments co-occurring with it in ≥ 95 % of its molecules | `qm9_env_folds_6b.json` |
| chemotypes | ECFP4 (2048 bits) → PCA(50) → k-means(50) | `qm9_chemotypes.pkl.gz` |

Cross-validation is 5 × 5. Fold (repeat *r*, chemotype fold *c*) holds out chemotype group *c* and environment fold (*r* + *c*) mod 5, and **every molecule carrying a held-out environment leaves training**. The test set is the 2 × 2 of the two holdouts:

| panel | chemotype | environments |
|---|---|---|
| A | held out | seen |
| B | held out | held out |
| C | seen | held out |
| D | seen | seen (in-distribution reference) |

Because the environment fold rotates with the repeat, each molecule is scored both with its environments unseen (B) and seen (A of other folds), with its chemotype held out both times. All folds train on the same number of molecules at a given fraction: 160, 803, 1,607 and 16,070 at 1, 5, 10 and 100 %.

```bash
# 0) (optional) regenerate the classifications; the shipped files define the splits, so reruns write *_rerun files
python data_curation/atomic/cluster_analysis/qm9_environment_clustering.py
python data_curation/atomic/cluster_analysis/qm9_environment_folds.py \
    --labels data_curation/atomic/cluster_analysis/qm9_env_labels_6b_rerun.pkl.gz
python data_curation/molecular/qm9_chemotype_clustering.py

# 1) Molecular complexes (3.5 Bohr / 4 NN) for AIMEl (ground truth) and the QM9 remainder (inferred)
python scripts/molecular/jax/pregenerate_complexes_env_holdout.py

# 2) Train one (repeat, cv_fold, fraction, variant); 5 × 5 folds × 4 fractions × 2 variants = 200 runs
python scripts/molecular/jax/train_molecular_env_holdout.py \
    --repeat 0 --cv-fold 0 --fraction 0.1 --variant informed

# 3) Deploy every trained run on the QM9 remainder (no retraining)
python scripts/inference/predict_env_holdout_from_inferred.py --variant blind
python scripts/inference/predict_env_holdout_from_inferred.py --variant informed

# 4) Tables from the run outputs
python analysis/molecular/env_holdout_analysis.py

# 5) Figures and the remaining tables: run the notebooks
#    analysis/molecular/analyze_env_holdout.ipynb, analysis/molecular/atomic_envs_novelty_6b.ipynb,
#    analysis/atomic/leakage_analysis/leakage_vs_radius.ipynb
```

Runs land in `experiments/molecular/env_holdout/<variant>/fold_<n>/frac_<f>/`. Only `config.json` is committed: it holds every seed, split size, held-out chemotype and environment, and the normalisation statistics (fit on the training split only). The run writes `val_preds.pkl`, `test_preds.pkl` (with the molecule's `panel`), `remainder_preds.pkl` and `train_ids.json` next to it. The informed variant is served the QT-Net ensemble's inferred properties (`qm9_inferred.pkl`), normalised with the statistics stored at training time.

> [!NOTE]
> The molecular graphs use a 3.5 Bohr / 4-neighbour cutoff, deliberately shorter than the 6-Bohr environment radius, to limit model capacity. After the model's 4 message-passing layers, 98.3 % of the atoms inside each atom's 6-Bohr environment are within reach.

### What reproduces without training

| Paper artifact | Entry point | Needs |
|---|---|---|
| Table 14; Figs 3, 7–13, 18 | `atomic_envs_novelty_6b.ipynb` | `setup_data.sh` and about 30 min; Fig 18 also needs a Chromium browser |
| Table 5 | `data_curation/atomic/cluster_analysis/qm9_environment_clustering.ipynb`, whose PCA rows section 4 of `atomic_envs_novelty_6b.ipynb` recomputes | `setup_data.sh` |
| Table 6 | section 6 of `atomic_envs_novelty_6b.ipynb`; the per-element atom columns are `per_label_atoms` in `qm9_env_folds_6b.json` | `setup_data.sh` |
| Figs 4, 19; Table 7 | first section of `analyze_env_holdout.ipynb` | the committed tables and `config.json` files |
| Fig 14 | sections 1–2 of `leakage_vs_radius.ipynb` | recomputes SOAP (about 10 min and 8 GB of disk per radius) |
| Table 11 | section 3 of `leakage_vs_radius.ipynb` | atomic `test_preds.pkl` from `scripts/atomic/train_multitask.py` |
| The rest of `analyze_env_holdout.ipynb` | steps 1–4 above | the 200 trained runs |

### Notes on reproducing the runs

- **Splits** regenerate exactly from the shipped files and the seeds in `config.json`.
- **Software:** the 200 runs used a flax >= 0.12 container, and this repository pins flax < 0.11, the stack that runs both the atomic and the molecular models. Stored checkpoints gave near-identical predictions on both stacks.
- **Training** ran on A100 GPUs with `JAX_DEFAULT_MATMUL_PRECISION=tensorfloat32`; deployment ran without it.
- **Hyperparameters** of `GeoScalarGNNMolecular` were tuned outside this repository, and `GeoScalarGNNMolecular_hp.json` is the record. The `code_commit` values in `config.json` refer to a private tree.
- **`verdict` in `r2_informed_vs_blind_*.csv`** follows the Holm-adjusted paired t-test. `analyze_env_holdout.ipynb` recomputes it with Tukey HSD, so rerunning its last cell changes those files on purpose.
- **t-SNE maps** reproduce up to numerical noise.
- **Chemotype reruns** depend on the BLAS/OpenMP thread count: another machine may reach an adjusted Rand index near 0.68 against the shipped labels, which define the splits.
- **Ensemble retrains** get different validation splits, because `train_QTNet_ensemble.py` seeds them with Python's salted `hash()`. The released `qm9_inferred.pkl` is the reference.

---

## 📈 Evaluation and Analysis

The `analysis/` directory contains the libraries and notebooks used to aggregate fold-level results across the 25 folds, run RM-ANOVA + Tukey HSD model comparisons, and produce the figures and tables in the paper.

```
analysis/
├── atomic/
│   ├── result_analysis.py       # canonical RM-ANOVA + Tukey HSD; LaTeX tables
│   ├── tmp_plot_cell_updated.py # forest plots, parity plots, box plots
│   ├── analyze_CV_augmented.ipynb
│   ├── analyze_CV_nonaug.ipynb
│   └── leakage_analysis/        # holdout novelty vs perception radius (Fig 14, Table 11)
├── molecular/
│   ├── analysis_molecular.py
│   ├── analyze_molecular_training_fracs.ipynb
│   ├── env_holdout_analysis.py      # environment-holdout tables (step 4 above)
│   ├── env_holdout_figures.py
│   ├── analyze_env_holdout.ipynb    # environment-holdout results: informed vs blind, same-molecule effect
│   ├── compare_atomic_envs_6b.py
│   ├── atomic_envs_novelty_6b.ipynb # distance to AIMEl, t-SNEs, QT-Net dipole errors
│   └── env_holdout_results/         # committed tables behind the figures
└── inference/
    ├── analyse_inference.py
    └── inference_results.ipynb
```

The 5×5 CV convention used throughout: 25 folds → repeat-level mean (5 folds per repeat) → RM-ANOVA on the 5 repeat-level means per model. See `analysis/atomic/result_analysis.py` for the canonical Tukey HSD implementation; the molecular and inference analyses reuse it. The environment-holdout study gives every fold its own test set, and its fold metrics are not correlated within repeats, so its comparisons use the 25 folds as subjects.


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
│   ├── molecular/env_holdout/<variant>/fold_<n>/frac_<f>/
│   └── inference/<model_type>/model_<i>/
└── optimal_hyperparams/   # Hyperparameter JSONs read by the atomic and ensemble training scripts
```

---

## 🤝 Contributing

Contributions are welcome. Please follow PEP 8 style conventions and annotate all public functions and classes with type hints. Use the `logging` module rather than `print` statements for any diagnostic output. Before opening a pull request, run HPO locally on at least fold 0 to verify that a new model or feature trains without errors.

---

## 📄 License

QT-Net is released under the MIT License.

---

## 📚 Citation

If you use QT-Net in your research, please cite:

```bibtex
@inproceedings{martinezcrespo2026qtnet,
  title     = {QT-Net: Rethinking Evaluation of AI Models in Atomic Chemical Space},
  author    = {Mart{\'i}nez Crespo, Pablo and Ribes, Stefano and Beckmann, Richard and Gliege, Marisa and Miret, Santiago and Jordan, Robert S. and Narasimhan, Vijay Kris and Rahm, Martin and Mercado, Roc{\'i}o},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
