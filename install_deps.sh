#!/bin/bash
# DiffCG Dependency Installation Script
# Tested on macOS (Darwin) with Python 3.10
# Note: JAX runs on CPU only due to jax-md compatibility constraints

set -e  # Exit on error

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install Miniconda or Anaconda first."
    exit 1
fi

# Configuration
ENV_NAME="diffcg"
PYTHON_VERSION="3.10"

echo "=== DiffCG Installation Script ==="
echo ""

# Create conda environment if it doesn't exist
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Conda environment '${ENV_NAME}' already exists."
    echo "Activating existing environment..."
else
    echo "Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
    conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y
fi

# Initialize conda for script
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

echo ""
echo "=== Installing JAX ecosystem (compatible versions) ==="
# JAX 0.4.23 is required for jax-md 0.2.8 compatibility
# jax-metal (Apple Silicon GPU) is NOT compatible with this JAX version
pip install "jax==0.4.23" "jaxlib==0.4.23" --no-deps

echo ""
echo "=== Installing JAX-MD ==="
pip install "jax-md==0.2.8"

echo ""
echo "=== Installing compatible versions of JAX ecosystem packages ==="
# These versions are compatible with JAX 0.4.23
pip install "dm-haiku==0.0.12" --no-deps
pip install "optax==0.1.9" "chex==0.1.85" "flax==0.7.5" "orbax-checkpoint==0.5.7" --no-deps

echo ""
echo "=== Installing NumPy and SciPy (compatible versions) ==="
# NumPy 2.x is incompatible with JAX 0.4.23
# SciPy 1.12+ has API changes that break JAX 0.4.23
pip install "numpy>=1.24,<2" "scipy>=1.9,<1.12"

echo ""
echo "=== Installing other dependencies ==="
pip install ase pandas chemfiles matplotlib

echo ""
echo "=== Installing remaining JAX ecosystem dependencies ==="
# These are dependencies of the packages above that may not have been installed
pip install absl-py msgpack tensorstore rich typing_extensions PyYAML \
    etils nest_asyncio protobuf jmp tabulate toolz ml_dtypes opt_einsum

echo ""
echo "=== Installing diffcg in development mode ==="
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pip install -e "${SCRIPT_DIR}" --no-deps

echo ""
echo "=== Verifying installation ==="
python -c "
import jax
import jax_md
import diffcg
print('JAX version:', jax.__version__)
print('JAX-MD version:', jax_md.__version__)
print('DiffCG installed successfully!')
print('')
print('Note: JAX is running on CPU. GPU acceleration via jax-metal')
print('is not available due to jax-md compatibility constraints.')
"

echo ""
echo "=== Installation complete! ==="
echo ""
echo "To activate the environment, run:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "To run the test script:"
echo "  python example/test_diffsim.py"
echo ""
echo "Note: The test script has hardcoded paths that may need updating."
