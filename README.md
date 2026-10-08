# complete_elliptic_integral

A tiny, dependency-free Python library computing the complete elliptic
integrals of the first and second kind, K(m) and E(m), via the
arithmetic-geometric mean (AGM) iteration.

## Usage

```python
from complete_elliptic_integral import K, E, complete_elliptic_integrals

# Single values
print(K(0.5))   # 1.8540746773013719
print(E(0.5))   # 1.3506438810476756

# Both at once, sharing the AGM sweep
k, e = complete_elliptic_integrals(0.5)
print(k, e)
```

The parameter convention is `m = k^2` (the *parameter*, not the
*modulus*).  Domain is `0 <= m <= 1`.

## Why this exists

The AGM algorithm converges quadratically and stays well-conditioned all
the way up to `m -> 1`, which is exactly the regime where the naive
power-series expansions in `m` blow up.  The trade-off is that the
implementation is slightly more code than a truncated series would be,
and it only covers the *complete* integrals (not the incomplete ones).

## Edge cases you will hit

- `K(1.0)` returns `float("inf")`.  The exact value is +infinity
  (logarithmic divergence); we surface that directly rather than raising
  or returning a silently-large finite number.  If you compute `K(m)`
  for `m` very close to 1, expect a large but finite value.
- `E(1.0)` returns `1.0` exactly (the integrand collapses to
  `|cos(theta|` on `[0, pi/2]`).
- `m` outside `[0, 1]` raises `ValueError`.  The Legendre form is not
  real-valued there under the standard definition, and we do not attempt
  analytic continuation.
- `agm` is also exported as a utility; it requires non-negative inputs.

## Layout

```
src/complete_elliptic_integral/__init__.py   # exports K, E, agm, complete_elliptic_integrals
src/complete_elliptic_integral/core.py       # implementation
tests/test_core.py                            # unittest suite
```

Run tests with:

```
PYTHONPATH=src python -m unittest discover -s tests
```

## Design notes

The window stores values eagerly rather than keeping running aggregates. Running
sums drift with floating point over long streams, and recomputing from a small
buffer is cheap enough that the drift is not worth the speed.

