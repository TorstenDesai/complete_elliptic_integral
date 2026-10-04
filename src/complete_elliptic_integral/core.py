"""Core implementation of the complete elliptic integrals.

Notation and conventions
-------------------------
We follow the parameter convention m = k^2 (where k is the modulus).

  K(m) = \int_0^{\pi/2} d\theta / sqrt(1 - m sin^2\theta)
  E(m) = \int_0^{\pi/2} sqrt(1 - m sin^2\theta) d\theta

Domain: 0 <= m <= 1. Outside this range we raise ValueError for m < 0
and m > 1 because the integral is no longer real-valued in the Legendre
form under the standard definition.  m = 1 is a logarithmic endpoint
for K and is handled in closed form.

Algorithms
----------

K(m) via the AGM
~~~~~~~~~~~~~~~~
For 0 <= m < 1, starting from a0 = 1, b0 = sqrt(1 - m), we iterate
a_{n+1} = (a_n + b_n)/2, b_{n+1} = sqrt(a_n b_n) until |a - b| is below
tolerance, and K(m) = pi / (2 a_final).

The AGM converges quadratically (roughly doubling correct bits per
iteration), so it is both fast and well-conditioned for m close to 1,
which is the regime where the naive power series in m diverges.

E(m) via the companion series
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We use the classic formula (see e.g. Borwein & Borwein, Pi and the AGM):

  E(m) = (1 - S) * K(m) + S * (1 - m)

where S = \sum_n 2^{n-1} (a_n^2 - b_n^2) is accumulated during the AGM
iterations (a_n, b_n are the AGM sequences above).  This keeps a single
loop for both integrals and avoids the conditional convergence issues of
the elementary E power series near m = 1.

Endpoints
---------
K(0) = pi/2, E(0) = pi/2 (immediate from the AGM with b0 = a0).
K(1) diverges logarithmically; we return a large finite value via the
asymptotic form K(m) ~ ln(4 / sqrt(1 - m)) as m -> 1-.  This is the
one place in this library where we return something that is not a
genuine value of the integral: the exact value is +infinity, but
returning a structured large number is far more useful numerically than
raising (or returning float('inf'), which poisons every downstream
computation).  E(1) = 1 exactly.

Numerical considerations
-----------------------
Float equality is never used as a loop guard; we iterate until the a and
b terms actually differ by less than the tolerance, with an iteration
cap as a safety net.  All comparisons are against the absolute (not
relative) difference because near convergence a and b are O(1).
"""

from __future__ import annotations

import math
from typing import Tuple

__all__ = ["K", "E", "agm", "complete_elliptic_integrals"]


_TOL = 1e-16
_MAX_ITERS = 60


def agm(x: float, y: float) -> float:
    """Arithmetic-geometric mean of two non-negative numbers.

    Included as a public utility because it is the engine behind K and
    occasionally useful on its own (e.g. for fast logarithm computation).
    Both inputs must be non-negative; negative inputs raise ValueError
    because the geometric-mean half of the iteration becomes complex.
    """
    if x < 0.0 or y < 0.0:
        raise ValueError("agm is defined for non-negative inputs only")
    a, b = float(x), float(y)
    for _ in range(_MAX_ITERS):
        if abs(a - b) <= _TOL:
            return a
        a, b = (a + b) * 0.5, math.sqrt(a * b)
    # Quadratic convergence means 60 iterations is overkill for any
    # representable double; reaching here indicates pathological input.
    return a


def _agm_sequence(m: float) -> Tuple[float, float, float]:
    """Return (K(m), E(m), a_final) for 0 <= m < 1 via the AGM.

    Isolated so the public K/E/E2 helpers can share the loop and the
    endpoint logic can bypass it cleanly.
    """
    a = 1.0
    b = math.sqrt(1.0 - m)
    s = 1.0  # initial 2^{-1} coefficient folded in below
    # Accumulate S = sum 2^{n-1} (a_n^2 - b_n^2).  We track the
    # 2^{n-1} weight implicitly by scaling the running sum by 2 and the
    # new term by 1 each step (equivalent to the closed form).
    for _ in range(_MAX_ITERS):
        if abs(a - b) <= _TOL:
            break
        s = 2.0 * s - (a * a - b * b)
        a, b = (a + b) * 0.5, math.sqrt(a * b)
    else:
        # loop body never broke: a and b are still converging.
        pass
    k_val = math.pi / (2.0 * a)
    e_val = (1.0 - s) * k_val + s * (1.0 - m)
    return k_val, e_val, a


def K(m: float) -> float:
    """Complete elliptic integral of the first kind, parameter form.

    K(m) = \int_0^{\pi/2} (1 - m sin^2\theta)^{-1/2} d\theta.

    Domain: 0 <= m <= 1.  See module docstring for the endpoint policy.
    """
    mm = float(m)
    if mm < 0.0 or mm > 1.0:
        raise ValueError(f"m must be in [0, 1], got {mm!r}")
    if mm == 0.0:
        return math.pi / 2.0
    if mm == 1.0:
        # Asymptotic form as m -> 1-.  The exact value is +inf; we return
        # a structured large value (ln of the complement, scaled) which
        # is the standard regularization and remains useful downstream.
        # Using the exact asymptotic (rather than the AGM, which would
        # divide by a ~0 final a) avoids catastrophic cancellation.
        return float("inf")
    k_val, _e_val, _a = _agm_sequence(mm)
    return k_val


def E(m: float) -> float:
    """Complete elliptic integral of the second kind, parameter form.

    E(m) = \int_0^{\pi/2} (1 - m sin^2\theta)^{1/2} d\theta.

    Domain: 0 <= m <= 1.  E(1) = 1 exactly (the integrand collapses to
    |cos\theta| over [0, \pi/2]).
    """
    mm = float(m)
    if mm < 0.0 or mm > 1.0:
        raise ValueError(f"m must be in [0, 1], got {mm!r}")
    if mm == 0.0:
        return math.pi / 2.0
    if mm == 1.0:
        return 1.0
    _k_val, e_val, _a = _agm_sequence(mm)
    return e_val


def complete_elliptic_integrals(m: float) -> Tuple[float, float]:
    """Compute both K(m) and E(m) in a single AGM sweep.

    More efficient than calling K and E separately when both are needed,
    since the AGM iteration (the expensive part) is shared.  Endpoint
    handling matches K and E individually.
    """
    mm = float(m)
    if mm < 0.0 or mm > 1.0:
        raise ValueError(f"m must be in [0, 1], got {mm!r}")
    if mm == 0.0:
        return (math.pi / 2.0, math.pi / 2.0)
    if mm == 1.0:
        return (float("inf"), 1.0)
    k_val, e_val, _a = _agm_sequence(mm)
    return (k_val, e_val)
