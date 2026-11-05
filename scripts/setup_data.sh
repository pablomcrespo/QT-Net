#!/usr/bin/env bash
# Reassemble and decompress all large data files.
# Run once from the repo root after cloning:
#   bash scripts/setup_data.sh

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "Setting up data files..."

decompress() {
  local gz="$1"
  local dest="${gz%.gz}"
  if [[ -f "$dest" ]]; then
    echo "  already exists, skipping: $(basename "$dest")"
    return
  fi
  echo "  decompressing: $(basename "$gz")"
  gunzip -k "$gz"
}

reassemble_and_decompress() {
  local prefix="$1"             # e.g. .../qm9_inferred.pkl.gz.part
  local gz="${prefix%.part}"    # .../qm9_inferred.pkl.gz
  local dest="${gz%.gz}"        # .../qm9_inferred.pkl

  if [[ -f "$dest" ]]; then
    echo "  already exists, skipping: $(basename "$dest")"
    return
  fi
  echo "  reassembling parts: $(basename "$gz")"
  cat "${prefix}"* > "$gz"
  echo "  decompressing: $(basename "$gz")"
  gunzip -k "$gz"
}

# Single-file gzip archives
decompress "$REPO_ROOT/data_curation/molecular/aimel_clustered_molecular.pkl.gz"
decompress "$REPO_ROOT/data_curation/atomic/data/aimel_dataset_with_components.csv.gz"
decompress "$REPO_ROOT/data_curation/molecular/qm9_filtered.pkl.gz"
decompress "$REPO_ROOT/data_curation/atomic/cluster_analysis/train_and_val.pkl.gz"
decompress "$REPO_ROOT/data_curation/atomic/cluster_analysis/test.pkl.gz"

# Split archive: qm9_inferred.pkl.gz.part{aa,ab,ac}
reassemble_and_decompress "$REPO_ROOT/data_curation/molecular/qm9_inferred.pkl.gz.part"

# The atomic scripts read these two from data_curation/atomic/
ln -sf cluster_analysis/train_and_val.pkl "$REPO_ROOT/data_curation/atomic/train_and_val.pkl"
ln -sf cluster_analysis/test.pkl "$REPO_ROOT/data_curation/atomic/test.pkl"

echo "Done. All data files are ready."
