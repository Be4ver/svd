"""Every original pair occurs once; local layout permutations restore order."""
import unittest
import numpy as np
from run_sweep import block_rounds, local_rounds


class SweepScheduleTests(unittest.TestCase):
    def test_local_pair_sets(self):
        for block in (16, 32, 64):
            for cross in (False, True):
                seen = []
                for p, q in local_rounds(block, cross):
                    self.assertEqual(len(set(p) | set(q)), 2*block)
                    seen.extend(tuple(sorted(pair)) for pair in zip(p, q))
                expected = {(p, q) for p in range(2*block) for q in range(p+1, 2*block)
                            if not cross or (p < block <= q)}
                self.assertEqual(set(seen), expected)
                self.assertEqual(len(seen), len(expected))

    def test_full_sweep_counts_including_4096(self):
        for block in (16, 32, 64):
            for nb in (2, 4, 6, 4096//block):
                counts = np.zeros((nb*block, nb*block), np.uint8)
                for outer, blocks in enumerate(block_rounds(nb)):
                    p = np.concatenate([p for p, q in local_rounds(block, outer != 0)])
                    q = np.concatenate([q for p, q in local_rounds(block, outer != 0)])
                    for first, second in blocks:
                        ids = np.r_[first*block:(first+1)*block, second*block:(second+1)*block]
                        # A local schedule has no repeated pair, hence advanced
                        # indexing increment is safe within each local job.
                        counts[ids[p], ids[q]] += 1
                        counts[ids[q], ids[p]] += 1
                np.testing.assert_array_equal(np.diag(counts), 0)
                np.fill_diagonal(counts, 1)
                self.assertTrue(np.all(counts == 1), (block, nb, np.unique(counts)))


if __name__ == '__main__':
    unittest.main()
