#!/usr/bin/env python3
"""Exercise in-kernel termination, repeated calls and partial resident grids."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
from run_npu import log_path, validate, validate_execution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, default=Path("build-fused/jacobi_svd_main"))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/fused-control"))
    args = parser.parse_args()
    rng = np.random.default_rng(17)
    cases = [
        ("two_aiv_repeat", rng.normal(size=(3, 37, 19)).astype(np.float32), 2, 30, 16, 1, 3, False),
        ("four_aiv_block32", rng.normal(size=(5, 128, 256)).astype(np.float32), 4, 30, 32, 1, 2, False),
        ("two_aiv_block64", rng.normal(size=(5, 128, 160)).astype(np.float32), 2, 30, 64, 2, 3, False),
        ("fixed_zero_block64", np.zeros((5, 129, 97), np.float32), 4, 3, 64, 1, 2, True),
        ("fixed_zero", np.zeros((5, 65, 33), np.float32), 0, 3, 16, 1, 2, True),
        ("fixed_random", rng.normal(size=(10, 32, 32)).astype(np.float32), 0, 20, 16, 2, 2, True),
    ]
    results = []
    for name, a, cores, sweeps, block, local, repeat, fixed in cases:
        root = args.artifacts / name
        inp, out = root / "input", root / "output"
        inp.mkdir(parents=True, exist_ok=True)
        a.tofile(inp / "A_batch.bin")
        batch, rows, cols = a.shape
        cmd = [str(args.exe.resolve()), str(args.device), str(rows), str(cols), str(batch), str(cores), str(sweeps),
               "--block", str(block), "--local-sweeps", str(local), "--repeat", str(repeat),
               "--input-dir", str(inp.resolve()), "--output-dir", str(out.resolve())]
        if fixed:
            cmd.append("--fixed-sweeps")
        run = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        log_path(root, "run.log").write_text(run.stdout + run.stderr)
        result = {"case": name}
        try:
            assert run.returncode == 0, run.stdout + run.stderr
            result.update(validate(a, out))
            result.update(validate_execution(out))
            assert len(result["stream_ms"]) == repeat
            assert result["converged"] is True
            assert all(f"run={i} sweeps={result['sweeps']} converged=1" in run.stdout for i in range(repeat)), run.stdout
            if fixed:
                assert result["sweeps"] == sweeps
            if cores:
                assert result["aiv"] == cores and result["aic"] == cores // 2
            result["pass"] = True
        except (AssertionError, ValueError, OSError) as error:
            result.update({"pass": False, "error": str(error)})
        results.append(result)
        print(json.dumps(result), flush=True)
        (args.artifacts / "results.json").write_text(json.dumps(results, indent=2))
    return int(any(not r["pass"] for r in results))


if __name__ == "__main__":
    raise SystemExit(main())
