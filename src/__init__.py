# src/__init__.py
"""gluing-matrix-validation: kleines Paket-API für lokale Entwicklung."""

__all__ = ["matrix_factory", "stability", "solvers", "analytics"]
__version__ = "0.0.0"

# Optional: bequeme Kurzimporte (nicht zwingend, aber praktisch)
from . import matrix_factory, stability, solvers, analytics
