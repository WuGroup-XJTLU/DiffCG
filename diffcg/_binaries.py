"""Resolve vendored binary paths installed alongside the diffcg package."""
import sysconfig
from pathlib import Path


def _find_binary(name: str) -> str:
    """Find a vendored binary by name (e.g. "gpumd", "nep", "fastmd")."""
    binary_rel = Path("diffcg") / "bin" / name

    # Non-editable install: binary is next to the package
    candidate = Path(__file__).parent / "bin" / name
    if candidate.is_file():
        return str(candidate)

    # Editable install: binary lands under platlib
    candidate = Path(sysconfig.get_path("platlib")) / binary_rel
    if candidate.is_file():
        return str(candidate)

    raise FileNotFoundError(
        f"{name} binary not found in the diffcg package. "
        f"Reinstall diffcg: pip uninstall diffcg && pip install diffcg"
    )


def get_gpumd_path() -> str:
    """Return absolute path to the vendored gpumd executable."""
    return _find_binary("gpumd")


def get_nep_path() -> str:
    """Return absolute path to the vendored nep executable."""
    return _find_binary("nep")


def get_fastmd_path() -> str:
    """Return absolute path to the vendored fastmd executable."""
    return _find_binary("fastmd")
