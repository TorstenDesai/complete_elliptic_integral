"""Complete elliptic integrals of the first and second kind.

Uses the arithmetic-geometric mean (AGM) iteration for K and the
companion series for E. Both are numerically stable and converge
quadratically.
"""

from .core import K, E, agm, complete_elliptic_integrals

__all__ = ["K", "E", "agm", "complete_elliptic_integrals"]
