#!/usr/bin/env bash
set -euo pipefail

echo "Python:"
python --version

echo "GPU:"
nvidia-smi || true

pip install -r requirements-colab.txt

echo "Setup complete."
