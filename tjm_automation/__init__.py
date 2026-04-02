"""Compatibility shim so `python -m tjm_automation` works from the repo root."""

from pathlib import Path

__all__ = ["__version__"]
__version__ = "0.1.0"

_src_package_dir = Path(__file__).resolve().parent.parent / "src" / "tjm_automation"
if _src_package_dir.exists():
    __path__.append(str(_src_package_dir))
