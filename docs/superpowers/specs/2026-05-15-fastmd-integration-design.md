# fastMD pip-installable integration design

## Goal

Make `fastmd` a required dependency of `diffcg` that installs automatically via `pip install diffcg`, removing the need for users to manually clone and compile the fastMD binary.

## Architecture

Two independently versioned packages on PyPI:

```
fastmd (PyPI)                          diffcg (PyPI)
├── fastmd/__init__.py                 ├── diffcg/
│   └── get_binary_path()             │   ├── md/fastmd_sampler.py
├── fastmd/_binary/fastmd (compiled)  │   └── ...
├── pyproject.toml                     ├── pyproject.toml
├── CMakeLists.txt                     │   dependencies: ["fastmd>=0.2"]
└── src/ (CUDA/C++ source)            └── ...
```

`pip install diffcg` → pip resolves `fastmd` dependency → CMake compiles CUDA source → binary lands in the fastmd package → `FastMDSampler` finds it via `fastmd.get_binary_path()`.

## fastMD package changes

### pyproject.toml (new)

Uses `scikit-build-core` as the build backend. It invokes CMake, compiles CUDA code, and places the binary into the wheel.

```toml
[build-system]
requires = ["scikit-build-core>=0.9"]
build-backend = "scikit_build_core.build"

[project]
name = "fastmd"
version = "0.2.0"
description = "GPU-accelerated molecular dynamics simulator"
requires-python = ">=3.8"

[tool.scikit-build]
cmake.args = ["-DCMAKE_BUILD_TYPE=Release"]
wheel.packages = ["python/fastmd"]
```

### python/fastmd/__init__.py (new)

Thin module with a single public API:

```python
from pathlib import Path

def get_binary_path() -> str:
    """Return the path to the compiled fastmd binary."""
    return str(Path(__file__).parent / "_binary" / "fastmd")
```

### CMakeLists.txt (modified)

Add an install rule that copies the compiled `fastMD` binary into the Python package's `_binary/` directory so scikit-build-core includes it in the wheel. The existing build logic stays untouched.

## diffcg changes

### pyproject.toml

Add `fastmd` to the required dependencies:

```toml
dependencies = [
    ...
    "fastmd>=0.2.0",
]
```

### FastMDSampler (diffcg/md/fastmd_sampler.py)

Replace the PATH-based default with `fastmd.get_binary_path()`:

```python
# Constructor signature change
fastmd_exe: str | None = None,  # was: str = "fastmd"
```

In the body, resolve the default at init time:

```python
if fastmd_exe is None:
    from fastmd import get_binary_path
    fastmd_exe = get_binary_path()
```

The `fastmd_exe` parameter is kept (not removed) so advanced users can point to a custom binary.

### sample.py

Update the two factory call sites (`create_sampler`, `create_sampler_from_checkpoint`) to let `fastmd_exe` default to `None`, so the sampler resolves it internally instead of hardcoding `"fastmd"`.

## Install-time failure handling

When CUDA toolkit is not available, `pip install fastmd` fails during the CMake build step. A pre-build check for `nvcc` provides a clear error message:

```
Error: CUDA Toolkit not found.
fastmd requires CUDA >= 11.0 with nvcc on PATH.
Install CUDA from: https://developer.nvidia.com/cuda-downloads
```

No runtime error handling is needed — if `pip install` succeeded, the binary is guaranteed present.

## Non-goals

- Pre-compiled GPU-architecture-specific wheels (users must have CUDA toolkit at install time)
- Conda package (pip-only for now; conda can follow the same pattern later)
- Changes to fastMD's C++ source or build logic beyond the install target
