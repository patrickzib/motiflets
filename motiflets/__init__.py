"""motiflets package public interface."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("motiflets")
except PackageNotFoundError:  # pragma: no cover - package not installed
    __version__ = "0.0.0"

from motiflets.motiflets import Motiflets

__all__ = ["Motiflets", "__version__"]
