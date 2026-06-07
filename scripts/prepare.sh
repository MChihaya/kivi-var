#!/usr/bin/env bash
# Fetch the VAR model code and the VAR-d16 checkpoints needed to run the experiments.
set -euo pipefail
cd "$(dirname "$0")/.."

# 1) VAR model code (FoundationVision/VAR) -- provides `models.build_vae_var`.
if [ ! -d VAR ]; then
  git clone --depth 1 https://github.com/FoundationVision/VAR.git
fi

# 2) Checkpoints: the VAR-d16 transformer + the shared multi-scale VQ-VAE, from the official HF repo.
mkdir -p checkpoints
base="https://huggingface.co/FoundationVision/var/resolve/main"
for f in vae_ch160v4096z32.pth var_d16.pth; do
  if [ ! -f "checkpoints/$f" ]; then
    echo "downloading $f ..."
    wget -q --show-progress -O "checkpoints/$f" "$base/$f"
  fi
done

echo "Ready: VAR code in ./VAR, checkpoints in ./checkpoints"
