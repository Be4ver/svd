#!/usr/bin/env python3
"""Time one large FP32 SVD; independently validate outside the timed region."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from run_npu import log_path, validate, validate_execution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, default=Path("build-fused/jacobi_svd_main"))
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument("--block", type=int, choices=(16, 32, 64), default=16)
    parser.add_argument("--tol", type=float, default=float(np.finfo(np.float32).eps))
    parser.add_argument("--local-sweeps", type=int, default=1)
    parser.add_argument("--max-sweeps", type=int, default=30)
    parser.add_argument("--repeat", type=int, default=4)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--no-overlap", action="store_true")
    parser.add_argument("--pipeline-jobs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261116)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/large-4096"))
    args = parser.parse_args()
    root = args.artifacts.resolve()
    inp, out = root / "input", root / "output"
    inp.mkdir(parents=True, exist_ok=True)
    a = np.random.default_rng(args.seed).standard_normal((1, args.size, args.size)).astype(np.float32)
    a.tofile(inp / "A_batch.bin")
    cmd = [str(args.exe.resolve()), str(args.device), str(args.size), str(args.size), "1", "0", str(args.max_sweeps),
           "--block", str(args.block), "--tol", str(args.tol), "--local-sweeps", str(args.local_sweeps), "--repeat", str(args.repeat),
           "--input-dir", str(inp), "--output-dir", str(out)]
    if args.no_overlap:
        cmd.append("--no-overlap")
    if args.pipeline_jobs:
        cmd.extend(["--pipeline-jobs", str(args.pipeline_jobs)])
    print("running", " ".join(cmd), flush=True)
    run = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    log_path(root, "run.log").write_text(run.stdout + run.stderr)
    print(run.stdout + run.stderr, flush=True)
    if run.returncode:
        raise AssertionError(f"CLI returned {run.returncode}: {run.stderr}")
    result = validate_execution(out)
    result["tolerance"] = args.tol
    result["stopping_criterion"] = "pairwise_relative_gram"
    assert result["converged"], "max sweeps reached: cannot report a converged decomposition time"
    assert len(result["stream_ms"]) == args.repeat
    warm = 1 if args.repeat > 1 else 0
    result["median_stream_ms"] = statistics.median(result["stream_ms"][warm:])
    result["median_operator_wall_ms"] = statistics.median(result["operator_wall_ms"][warm:])
    result["seed"] = args.seed
    (root / "timing.json").write_text(json.dumps(result, indent=2))
    print("timing", json.dumps(result), flush=True)
    print("validating with independent NumPy SVD (outside operator timing)", flush=True)
    result.update(validate(a, out))
    result["pass"] = True
    (root / "results.json").write_text(json.dumps(result, indent=2))
    print("validated", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
