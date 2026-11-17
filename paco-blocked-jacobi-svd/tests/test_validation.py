"""Check that the output validator binds each right vector to its singular value."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from run_npu import validate


class OutputValidationTest(unittest.TestCase):
    def test_large_matrix_wrong_vector_order_is_rejected(self):
        # The Frobenius norm grows with rank and can hide wrong eigenpairs.
        # At this size even reversing all right vectors passed the old check.
        n = 1536
        diagonal = np.linspace(1, 2, n, dtype=np.float32)
        a = np.diag(diagonal)[None]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            a.tofile(output / "B_rot.bin")
            diagonal[::-1].copy().tofile(output / "sigma.bin")
            np.eye(n, dtype=np.float32).tofile(output / "Vt.bin")
            with self.assertRaises(AssertionError):
                validate(a, output)

    def test_swapped_right_vectors_are_rejected(self):
        a = np.diag([3.0, 1.0]).astype(np.float32)[None]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            a.tofile(output / "B_rot.bin")
            np.array([[3.0, 1.0]], np.float32).tofile(output / "sigma.bin")
            np.eye(2, dtype=np.float32)[None].tofile(output / "Vt.bin")
            validate(a, output)
            np.eye(2, dtype=np.float32)[::-1].copy()[None].tofile(output / "Vt.bin")
            with self.assertRaises(AssertionError):
                validate(a, output)


if __name__ == "__main__":
    unittest.main()
