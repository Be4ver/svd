"""Numerical contracts, independent of the NPU implementation."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from reference import tournament, rotation, blocked_svd


class ReferenceTests(unittest.TestCase):
    def test_tournament_covers_each_pair_once_without_conflicts(self):
        for n in (2, 4, 6, 8):
            rounds = tournament(n)
            seen = []
            for pairs in rounds:
                self.assertEqual(sorted(x for pair in pairs for x in pair), list(range(n)))
                seen.extend(tuple(sorted(pair)) for pair in pairs)
            self.assertEqual(len(set(seen)), n * (n - 1) // 2)

    def test_rotation_diagonalizes_gram_and_is_orthogonal(self):
        x = np.random.default_rng(7).normal(size=(32, 43)).astype(np.float32)
        g = x @ x.T
        r = rotation(g, sweeps=12)
        np.testing.assert_allclose(r @ r.T, np.eye(32), atol=2e-5)
        d = r @ g @ r.T
        self.assertLess(np.max(np.abs(d - np.diag(np.diag(d)))) / np.max(np.diag(d)), 2e-5)

    def test_svd_and_row_space_on_edge_inputs(self):
        rng = np.random.default_rng(17)
        for m, n in ((13, 21), (20, 20), (32, 32), (37, 19), (64, 81)):
            for kind in ("normal", "zero", "identity", "rank"):
                x = rng.normal(size=(m, n)).astype(np.float32)
                if kind == "zero": x[:] = 0
                if kind == "identity": x = np.eye(m, n, dtype=np.float32)
                if kind == "rank": x[m // 2:] = x[0]
                b = blocked_svd(x, sweeps=20)
                s = np.sort(np.linalg.norm(b.astype(np.float64), axis=1))[::-1][:min(m, n)]
                ref = np.linalg.svd(x.astype(np.float64), compute_uv=False)
                with self.subTest(shape=(m, n), kind=kind):
                    np.testing.assert_allclose(s, ref, atol=3e-4, rtol=5e-4)
                    g = b.astype(np.float64) @ b.astype(np.float64).T
                    self.assertLess(np.max(np.abs(g - np.diag(np.diag(g)))) / max(np.max(np.diag(g)), 1e-30), 1e-3)


if __name__ == "__main__":
    unittest.main()
