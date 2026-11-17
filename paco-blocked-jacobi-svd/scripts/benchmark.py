#!/usr/bin/env python3
"""Tune block/local sweeps and compare CLI latency against the PACO baseline.

Inputs and outputs stay under --artifacts; logs are collected under project logs/.
"""
import argparse
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from run_npu import log_path, validate, validate_execution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, default=Path("build-fused/jacobi_svd_main"))
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/benchmark"))
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--blocks", type=int, nargs="+", choices=(16, 32))
    parser.add_argument("--locals", type=int, nargs="+")
    args = parser.parse_args()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261116)
    results = []
    for m, n, batch in [(20, 20, 512), (32, 32, 512), (64, 64, 256), (128, 128, 4), (128, 128, 40)]:
        a = rng.normal(size=(batch, m, n)).astype(np.float32)
        root = (args.artifacts / f"{m}x{n}_b{batch}").resolve()
        inp, work = root / "input", root / "work"
        inp.mkdir(parents=True, exist_ok=True)
        work.mkdir(exist_ok=True)
        # The PACO CLI's WriteFile does not create its output directory.
        (root / "output").mkdir(exist_ok=True)
        a.tofile(inp / "A_batch.bin")
        configurations = [(16, 8)]
        if args.tune: configurations = [(block, local) for block in (16, 32) for local in (1, 2, 4, 8)]
        if args.blocks or args.locals:
            configurations = [(block, local) for block in (args.blocks or [16]) for local in (args.locals or [8])]
        for block, local in configurations:
            output = root / f"blocked_b{block}_s{local}"
            cmd = [str(args.exe.resolve()), str(args.device), str(m), str(n), str(batch), "0", "40",
                   "--block", str(block), "--local-sweeps", str(local), "--repeat", "4",
                   "--input-dir", str(inp), "--output-dir", str(output)]
            run = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            log_path(root, f"blocked_b{block}_s{local}.log").write_text(run.stdout + run.stderr)
            record = {"implementation": "blocked", "rows": m, "cols": n, "batch": batch,
                      "block": block, "local_sweeps": local}
            try:
                if run.returncode: raise AssertionError(f"CLI returned {run.returncode}: {run.stderr}")
                record.update(validate(a, output))
                record.update(validate_execution(output))
                record["median_ms"] = statistics.median(record["stream_ms"][1:])
                record["median_wall_ms"] = statistics.median(record["operator_wall_ms"][1:])
                record["first_wall_ms"] = record["operator_wall_ms"][0]
                record["pass"] = True
            except (AssertionError, ValueError, OSError) as error:
                record.update({"pass": False, "error": str(error)})
            results.append(record)
            print(json.dumps(record), flush=True)
            (args.artifacts / "results.json").write_text(json.dumps(results, indent=2))
        if args.baseline:
            times = []
            record = {"implementation": "paco", "rows": m, "cols": n, "batch": batch}
            try:
                for repeat in range(3):
                    cmd = [str(args.baseline.resolve()), str(args.device), str(m), str(n), str(batch), "40", "40"]
                    run = subprocess.run(cmd, cwd=work, capture_output=True, text=True, timeout=180)
                    log_path(root, f"paco_{repeat}.log").write_text(run.stdout + run.stderr)
                    if run.returncode: raise AssertionError(f"CLI returned {run.returncode}: {run.stderr}")
                    times.append(float(re.search(r"耗时:\s*([\d.]+)\s*ms", run.stdout)[1]))
                record.update(validate(a, root / "output"))
                # The baseline reports host launch-to-sync time, not ACL events.
                # Each baseline sample is a fresh process; blocked repeat[1:]
                # samples are warm. Preserve both rather than equating them.
                record.update({"operator_wall_ms": times, "median_ms": statistics.median(times),
                               "timing": "fresh-process launch-to-sync", "pass": True})
            except (AssertionError, ValueError, OSError, TypeError) as error:
                record.update({"pass": False, "error": str(error)})
            results.append(record)
            print(json.dumps(record), flush=True)
            (args.artifacts / "results.json").write_text(json.dumps(results, indent=2))
    return int(any(not r["pass"] for r in results))


if __name__ == "__main__":
    raise SystemExit(main())
