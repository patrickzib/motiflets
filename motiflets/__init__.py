"""motiflets package public interface."""

from __future__ import annotations

try:  # Python 3.8+
    from importlib.metadata import PackageNotFoundError, version as _pkg_version
except ImportError:  # pragma: no cover - fallback for Python 3.7
    from importlib_metadata import PackageNotFoundError, version as _pkg_version  # type: ignore

try:
    __version__ = _pkg_version("motiflets")
except PackageNotFoundError:  # pragma: no cover - package not installed
    __version__ = "0.0.0"


__all__ = ["__version__"]
