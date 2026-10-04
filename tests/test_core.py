import unittest
from complete_elliptic_integral import K, E, agm, complete_elliptic_integrals
import math


class TestJoint(unittest.TestCase):
    def test_complete_elliptic_integrals_matches_individuals(self):
        for m in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.999):
            with self.subTest(m=m):
                k_pair, e_pair = complete_elliptic_integrals(m)
                self.assertAlmostEqual(k_pair, K(m), places=14)
                self.assertAlmostEqual(e_pair, E(m), places=14)


if __name__ == "__main__":
    unittest.main()
