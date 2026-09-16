#!/usr/bin/env bash
# scripts/install_torch.sh
# Install PyTorch cu130 nightly into the active conda environment.
# Run AFTER: conda env create -f environment.yml && conda activate hiver_sde
#
# Why separate: conda's pip install applies --index-url globally, which breaks
# all other PyPI packages if torch is listed alongside them in environment.yml.

set -euo pipefail

echo "=== Installing PyTorch cu130 nightly for sm_120 (RTX 5050 Blackwell) ==="

pip install --index-url https://download.pytorch.org/whl/nightly/cu130 \
    "torch>=2.14.0.dev0"

echo "=== Verifying CUDA capability ==="
python -c "
import torch
version = torch.__version__
cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else 'N/A (no CUDA)'
print(f'torch: {version}')
print(f'cuda available: {torch.cuda.is_available()}')
print(f'device capability: {cap}')
if torch.cuda.is_available():
    assert cap == (12, 0), f'Expected (12, 0) for RTX 5050 sm_120, got {cap}'
    print('PASS: sm_120 verified')
else:
    print('INFO: No CUDA — will run on CPU (acceptable for data-wrangling phases)')
"
