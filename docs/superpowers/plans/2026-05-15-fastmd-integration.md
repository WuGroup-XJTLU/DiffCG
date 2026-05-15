# fastMD pip-installable integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `fastmd` a required pip dependency of `diffcg` so users get the fastMD binary automatically via `pip install diffcg`.

**Architecture:** The fastMD repo gains a `pyproject.toml` with scikit-build-core that compiles the CUDA source and packages the binary into a Python wheel. The diffcg repo gains a dependency on `fastmd` and `FastMDSampler` resolves the binary via `fastmd.get_binary_path()` instead of PATH lookup.

**Tech Stack:** scikit-build-core, CMake, CUDA C++17, setuptools

**Repos touched:** `/home/zhenghaowu/fastMD` (external), `/home/zhenghaowu/diffcg_repo` (this repo)

---

## File Map

| Action | Path | Purpose |
|--------|------|---------|
| Create | `fastMD/pyproject.toml` | scikit-build-core build config |
| Create | `fastMD/python/fastmd/__init__.py` | `get_binary_path()` API |
| Create | `fastMD/python/fastmd/_check_cuda.py` | Pre-build CUDA toolkit check |
| Modify | `fastMD/CMakeLists.txt` | Add install() target to copy binary |
| Modify | `diffcg/pyproject.toml` | Add `fastmd>=0.2.0` dependency |
| Modify | `diffcg/md/fastmd_sampler.py:145` | Change `fastmd_exe` default |
| Modify | `diffcg/md/fastmd_sampler.py:165-166` | Resolve binary path at init |
| Modify | `diffcg/md/sample.py:390` | Update factory call site 1 |
| Modify | `diffcg/md/sample.py:522` | Update factory call site 2 |

---

### Task 1: Add pyproject.toml to fastMD repo

**Repo:** `/home/zhenghaowu/fastMD`
**Files:**
- Create: `pyproject.toml`

- [ ] **Step 1: Write pyproject.toml**

```toml
[build-system]
requires = ["scikit-build-core>=0.9"]
build-backend = "scikit_build_core.build"

[project]
name = "fastmd"
version = "0.2.0"
description = "GPU-accelerated molecular dynamics simulator"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.8"
keywords = ["molecular-dynamics", "cuda", "gpu", "simulation"]
classifiers = [
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering :: Physics",
]

[project.urls]
Homepage = "https://github.com/Chenghao-Wu/fastMD"

[tool.scikit-build]
cmake.args = ["-DCMAKE_BUILD_TYPE=Release"]
wheel.packages = ["python/fastmd"]
```

- [ ] **Step 2: Commit in fastMD repo**

```bash
cd /home/zhenghaowu/fastMD && git add pyproject.toml && git commit -m "build: add pyproject.toml for pip-installable wheel"
```

---

### Task 2: Create Python package skeleton in fastMD repo

**Repo:** `/home/zhenghaowu/fastMD`
**Files:**
- Create: `python/fastmd/__init__.py`

- [ ] **Step 1: Create the package directory and __init__.py**

```bash
mkdir -p /home/zhenghaowu/fastMD/python/fastmd
```

```python
# /home/zhenghaowu/fastMD/python/fastmd/__init__.py
"""fastMD — GPU-accelerated molecular dynamics simulator.

Provides ``get_binary_path()`` to locate the compiled fastmd binary
installed alongside this package.
"""

from pathlib import Path


def get_binary_path() -> str:
    """Return the absolute path to the compiled fastmd executable.

    Returns:
        str: Path to the ``fastmd`` binary shipped with this package.

    Raises:
        FileNotFoundError: If the binary is missing (indicates a broken
            installation — reinstall the package).
    """
    binary = Path(__file__).parent / "_binary" / "fastmd"
    if not binary.is_file():
        raise FileNotFoundError(
            f"fastmd binary not found at {binary}. "
            f"Reinstall the fastmd package: pip uninstall fastmd && pip install fastmd"
        )
    return str(binary)
```

- [ ] **Step 2: Commit in fastMD repo**

```bash
cd /home/zhenghaowu/fastMD && git add python/ && git commit -m "feat: add Python package with get_binary_path()"
```

---

### Task 3: Add CMake install target to copy binary

**Repo:** `/home/zhenghaowu/fastMD`
**Files:**
- Modify: `CMakeLists.txt` (append install rule)

- [ ] **Step 1: Read current CMakeLists.txt**

The last two lines are:
```cmake
set_target_properties(fastMD PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
```

- [ ] **Step 2: Append install rule and rename output to lowercase**

scikit-build-core expects the binary to land in the wheel's package directory. We add an `install()` command that copies the built `fastMD` executable into `python/fastmd/_binary/`. Also rename the output to lowercase `fastmd` so `get_binary_path()` finds it.

Append to `CMakeLists.txt` after the `add_subdirectory(tests)` line:

```cmake
# Install the fastMD binary into the Python package for scikit-build-core wheel packaging
set_target_properties(fastMD PROPERTIES OUTPUT_NAME fastmd)
install(TARGETS fastMD DESTINATION python/fastmd/_binary)
```

- [ ] **Step 3: Verify the install target works**

```bash
cd /home/zhenghaowu/fastMD/build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/tmp/fastmd_test_install && make -j$(nproc) && make install && ls -la /tmp/fastmd_test_install/python/fastmd/_binary/fastmd
```

Expected: Binary file exists at the install path. Clean up: `rm -rf /tmp/fastmd_test_install`

- [ ] **Step 4: Commit in fastMD repo**

```bash
cd /home/zhenghaowu/fastMD && git add CMakeLists.txt && git commit -m "build: add install target to copy binary into Python package"
```

---

### Task 4: Test pip install of fastmd package

**Repo:** `/home/zhenghaowu/fastMD`

- [ ] **Step 1: Install scikit-build-core**

```bash
pip install scikit-build-core>=0.9
```

- [ ] **Step 2: Build and install fastmd in development mode**

```bash
cd /home/zhenghaowu/fastMD && pip install -e . --no-build-isolation -v
```

Expected: CMake configures and builds without errors. The `fastmd` package becomes importable.

- [ ] **Step 3: Verify get_binary_path() returns a valid executable**

```bash
python -c "from fastmd import get_binary_path; p = get_binary_path(); print('Binary at:', p); import os; print('Exists:', os.path.isfile(p)); print('Executable:', os.access(p, os.X_OK))"
```

Expected: All three checks print `True`.

- [ ] **Step 4: Verify the binary actually runs**

```bash
python -c "from fastmd import get_binary_path; import subprocess; p = get_binary_path(); result = subprocess.run([p, '--help'], capture_output=True, text=True); print('stdout:', result.stdout[:500]); print('stderr:', result.stderr[:500]); print('returncode:', result.returncode)"
```

Expected: The binary prints usage info or runs without config error (return code may be non-zero due to missing config, but should not crash with "command not found" or similar).

---

### Task 5: Add pre-build CUDA check

**Repo:** `/home/zhenghaowu/fastMD`
**Files:**
- Create: `python/fastmd/_check_cuda.py`
- Modify: `pyproject.toml` (add build hook)

- [ ] **Step 1: Write the CUDA check**

```python
# /home/zhenghaowu/fastMD/python/fastmd/_check_cuda.py
"""Pre-build check for CUDA toolkit availability.

Called by scikit-build-core before CMake configure. Provides a clear
error message when nvcc is not on PATH instead of the cryptic CMake
"CUDA compiler not found" error.
"""

import os
import shutil
import sys


def check_cuda():
    """Raise SystemExit with a helpful message if nvcc is not available."""
    if shutil.which("nvcc") is not None:
        return

    # Check common CUDA install locations for a better hint
    hints = []
    for candidate in [
        "/usr/local/cuda/bin/nvcc",
        "/usr/local/cuda-12/bin/nvcc",
        "/usr/local/cuda-11/bin/nvcc",
    ]:
        if os.path.isfile(candidate):
            hints.append(
                f"  Found {candidate} — add to PATH: "
                f"export PATH={os.path.dirname(candidate)}:$PATH"
            )

    msg = [
        "Error: CUDA Toolkit not found.",
        "fastmd requires CUDA >= 11.0 with nvcc on PATH.",
        "Install CUDA from: https://developer.nvidia.com/cuda-downloads",
    ]
    if hints:
        msg.append("")
        msg.extend(hints)

    sys.exit("\n".join(msg))
```

- [ ] **Step 2: Update pyproject.toml to wire the pre-build hook**

Replace `pyproject.toml` with the version that includes the pre-build hook in `[tool.scikit-build]`:

```toml
[build-system]
requires = ["scikit-build-core>=0.9"]
build-backend = "scikit_build_core.build"

[project]
name = "fastmd"
version = "0.2.0"
description = "GPU-accelerated molecular dynamics simulator"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.8"
keywords = ["molecular-dynamics", "cuda", "gpu", "simulation"]
classifiers = [
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering :: Physics",
]

[project.urls]
Homepage = "https://github.com/Chenghao-Wu/fastMD"

[tool.scikit-build]
cmake.args = ["-DCMAKE_BUILD_TYPE=Release"]
wheel.packages = ["python/fastmd"]
build.pre_configure = ["python -c 'from fastmd._check_cuda import check_cuda; check_cuda()'"]
```

Wait — the `build.pre_configure` hook runs *inside the build environment* where `python/fastmd/` may not yet be on `sys.path`. We need an alternative: use a standalone script at the repo root instead.

- [ ] **Step 3: Move the check to a standalone script at repo root**

Write the check as `/home/zhenghaowu/fastMD/_check_cuda.py` (at repo root, not inside the Python package):

```python
# /home/zhenghaowu/fastMD/_check_cuda.py
"""Check CUDA toolkit availability before CMake configure."""
import os
import shutil
import sys


def main():
    if shutil.which("nvcc") is not None:
        return

    hints = []
    for candidate in [
        "/usr/local/cuda/bin/nvcc",
        "/usr/local/cuda-12/bin/nvcc",
        "/usr/local/cuda-11/bin/nvcc",
    ]:
        if os.path.isfile(candidate):
            hints.append(
                f"  Found {candidate} — add to PATH: "
                f"export PATH={os.path.dirname(candidate)}:$PATH"
            )

    msg = [
        "Error: CUDA Toolkit not found.",
        "fastmd requires CUDA >= 11.0 with nvcc on PATH.",
        "Install CUDA from: https://developer.nvidia.com/cuda-downloads",
    ]
    if hints:
        msg.append("")
        msg.extend(hints)

    sys.exit("\n".join(msg))


if __name__ == "__main__":
    main()
```

Update `pyproject.toml` to reference this script:

```toml
[tool.scikit-build]
cmake.args = ["-DCMAKE_BUILD_TYPE=Release"]
wheel.packages = ["python/fastmd"]
build.pre_configure = ["python _check_cuda.py"]
```

- [ ] **Step 4: Test the check triggers when CUDA is missing (optional — only if you can safely hide nvcc)**

```bash
cd /home/zhenghaowu/fastMD && PATH=/usr/bin:/bin python _check_cuda.py
```

Expected: Prints the "CUDA Toolkit not found" error message.

- [ ] **Step 5: Commit**

```bash
cd /home/zhenghaowu/fastMD && git add _check_cuda.py pyproject.toml && git commit -m "feat: add pre-build CUDA toolkit check"
```

---

### Task 6: Add fastmd dependency to diffcg

**Repo:** `/home/zhenghaowu/diffcg_repo`
**Files:**
- Modify: `pyproject.toml:22-31`

- [ ] **Step 1: Add fastmd to dependencies**

Edit `pyproject.toml`, add `"fastmd>=0.2.0"` to the `dependencies` list:

```toml
dependencies = [
  "numpy>=1.23",
  "scipy>=1.9",
  "pandas>=1.5",
  "ase>=3.22",
  "chemfiles>=0.10",
  "optax>=0.1.6",
  "jax>=0.4",
  "jax-md>=0.2.7",
  "fastmd>=0.2.0",
]
```

- [ ] **Step 2: Commit in diffcg repo**

```bash
cd /home/zhenghaowu/diffcg_repo && git add pyproject.toml && git commit -m "feat: add fastmd as required pip dependency"
```

---

### Task 7: Update FastMDSampler to resolve binary from package

**Repo:** `/home/zhenghaowu/diffcg_repo`
**Files:**
- Modify: `diffcg/md/fastmd_sampler.py:145` (signature)
- Modify: `diffcg/md/fastmd_sampler.py:165` (default resolution)

- [ ] **Step 1: Change the default from "fastmd" to None and add resolution logic**

Change line 145 from:
```python
        fastmd_exe: str = "fastmd",
```
To:
```python
        fastmd_exe: Optional[str] = None,
```

Then replace line 165:
```python
        self.fastmd_exe = fastmd_exe
```
With:
```python
        if fastmd_exe is None:
            from fastmd import get_binary_path
            fastmd_exe = get_binary_path()
        self.fastmd_exe = fastmd_exe
```

- [ ] **Step 2: Verify the change with a quick import test**

Since fastmd is already installed (Task 4), this should work:

```bash
cd /home/zhenghaowu/diffcg_repo && python -c "
from diffcg.md.fastmd_sampler import FastMDSampler
# Just verify the class imports and the module compiles
print('FastMDSampler imported successfully')
"
```

- [ ] **Step 3: Commit in diffcg repo**

```bash
cd /home/zhenghaowu/diffcg_repo && git add diffcg/md/fastmd_sampler.py && git commit -m "feat: resolve fastmd binary via package instead of PATH"
```

---

### Task 8: Update sample.py factory call sites

**Repo:** `/home/zhenghaowu/diffcg_repo`
**Files:**
- Modify: `diffcg/md/sample.py:390` (first call site)
- Modify: `diffcg/md/sample.py:522` (second call site)

- [ ] **Step 1: Update both call sites**

Line 390 — change:
```python
            fastmd_exe=fc.get("fastmd_exe", "fastmd"),
```
To:
```python
            fastmd_exe=fc.get("fastmd_exe"),
```

Line 522 — change:
```python
            fastmd_exe=fc.get("fastmd_exe", "fastmd"),
```
To:
```python
            fastmd_exe=fc.get("fastmd_exe"),
```

This removes the hardcoded `"fastmd"` fallback. When `fc.get("fastmd_exe")` returns `None` (not in config), `FastMDSampler.__init__` will resolve the binary via `fastmd.get_binary_path()`.

- [ ] **Step 2: Verify sample.py imports and compiles**

```bash
cd /home/zhenghaowu/diffcg_repo && python -c "from diffcg.md.sample import create_sampler, create_sampler_from_checkpoint; print('sample.py imports ok')"
```

- [ ] **Step 3: Commit in diffcg repo**

```bash
cd /home/zhenghaowu/diffcg_repo && git add diffcg/md/sample.py && git commit -m "fix: remove hardcoded fastmd PATH fallback in factory functions"
```

---

### Task 9: Integration test

**Repo:** `/home/zhenghaowu/diffcg_repo`

- [ ] **Step 1: Verify end-to-end resolution with a default-constructed sampler**

This test verifies that when `fastmd_exe` is not specified, the sampler resolves it from the package:

```bash
cd /home/zhenghaowu/diffcg_repo && python -c "
import jax.numpy as jnp
from diffcg.system import AtomicSystem
from diffcg.energy import TabulatedPairEnergy

# Minimal system: 1 water molecule, 3 atoms
pos = jnp.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [-0.03, 0.09, 0.0]])
cell = jnp.eye(3) * 2.0
Z = jnp.array([0, 1, 1])
masses = jnp.array([16.0, 1.0, 1.0])
system = AtomicSystem(positions=pos, Z=Z, cell=cell, masses=masses)

# Create tabulated energy (OW-OW pair)
r = jnp.linspace(0.2, 1.0, 100)
y = 0.01 * (r - 0.3)**2  # harmonic
energy = TabulatedPairEnergy(
    x=r,
    y=y,
    interaction_type='pair',
    lammps_style='table',
    keyword='PAIR_0',
    types=(0, 0),
)

from diffcg.md.fastmd_sampler import FastMDSampler

sampler = FastMDSampler(
    system,
    energy_objects=[energy],
    topology={'pair_types': [(0, 0)]},
    ensemble='nvt',
    temperature=300.0,
    timestep=2.0,
    loginterval=10,
)
print('FastMDSampler created successfully')
print('Resolved exe:', sampler.fastmd_exe)

# Quick run
traj = sampler.run(50)
print(f'Generated {len(traj)} frames')
print('Integration test PASSED')
"
```

Expected: Sampler creates successfully, binary resolved to a path under `.../python/fastmd/_binary/fastmd`, runs 50 MD steps, produces trajectory frames.

- [ ] **Step 2: Verify custom fastmd_exe still works (backward compat)**

```bash
cd /home/zhenghaowu/diffcg_repo && python -c "
from fastmd import get_binary_path
from diffcg.md.fastmd_sampler import FastMDSampler
# Test that explicit fastmd_exe is still honored
import jax.numpy as jnp
from diffcg.system import AtomicSystem
from diffcg.energy import TabulatedPairEnergy

pos = jnp.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [-0.03, 0.09, 0.0]])
system = AtomicSystem(positions=pos, Z=jnp.array([0, 1, 1]), cell=jnp.eye(3)*2.0, masses=jnp.array([16.0, 1.0, 1.0]))
r = jnp.linspace(0.2, 1.0, 100)
energy = TabulatedPairEnergy(x=r, y=0.01*(r-0.3)**2, interaction_type='pair', lammps_style='table', keyword='PAIR_0', types=(0,0))

explicit_path = get_binary_path()
sampler = FastMDSampler(
    system, energy_objects=[energy],
    topology={'pair_types': [(0, 0)]},
    fastmd_exe=explicit_path, loginterval=10,
)
assert sampler.fastmd_exe == explicit_path, f'Expected {explicit_path}, got {sampler.fastmd_exe}'
print('Backward compat test PASSED')
"
```

- [ ] **Step 3: Commit (if needed — no code changes at this point)**

Integration tests pass. No additional commit needed unless test failure reveals a bug.

---

### Task 10: Push and publish fastmd to PyPI (manual, requires credentials)

**Repo:** `/home/zhenghaowu/fastMD`

- [ ] **Step 1: Build the source distribution**

```bash
cd /home/zhenghaowu/fastMD && pip install build && python -m build
```

Expected: Creates `dist/fastmd-0.2.0.tar.gz`.

- [ ] **Step 2: Verify the sdist contents**

```bash
tar tzf /home/zhenghaowu/fastMD/dist/fastmd-0.2.0.tar.gz | head -30
```

Expected: Should include `src/`, `python/fastmd/`, `CMakeLists.txt`, `_check_cuda.py`, `pyproject.toml`.

- [ ] **Step 3: Upload to PyPI**

```bash
pip install twine && twine upload dist/fastmd-0.2.0.tar.gz
```

> This step requires PyPI credentials for the `fastmd` project. Run it manually when ready.

---

### Task 11: Push diffcg changes

**Repo:** `/home/zhenghaowu/diffcg_repo`

- [ ] **Step 1: Ensure all diffcg commits are in order**

```bash
cd /home/zhenghaowu/diffcg_repo && git log --oneline -5
```

- [ ] **Step 2: Push (requires user authorization)**

```bash
cd /home/zhenghaowu/diffcg_repo && git push origin main
```
