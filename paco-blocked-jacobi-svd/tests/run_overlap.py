#!/usr/bin/env python3
"""Compare pipelined and serial real-device runs, including scheduling edges."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from run_npu import log_path, validate, validate_execution


def cases():
    rng = np.random.default_rng(923)
    # Literal group sizes catch omitted/repeated work or wrong local/global IDs.
    yield "explicit_even", rng.normal(size=(4, 128, 193)).astype(np.float32), 32, 0, 4, (4, 4), False
    yield "auto_small", rng.normal(size=(4, 128, 193)).astype(np.float32), 32, 0, 0, (8, 0), False
    yield "explicit_uneven", rng.normal(size=(3, 128, 129)).astype(np.float32), 16, 4, 5, (5, 7), False
    yield "odd_two_aiv", rng.normal(size=(5, 32, 48)).astype(np.float32), 16, 2, 0, (2, 3), False
    yield "singleton", rng.normal(size=(1, 7, 9)).astype(np.float32), 32, 0, 0, (1, 0), False
    yield "fixed_zero", np.zeros((3, 65, 33), np.float32), 16, 0, 5, (5, 4), True
    yield "fixed_random", rng.normal(size=(3, 128, 81)).astype(np.float32), 32, 0, 1, (1, 5), True
    yield "full_grid_auto", rng.normal(size=(8, 512, 512)).astype(np.float32), 32, 0, 0, (40, 24), False
    yield "full_grid_40", rng.normal(size=(8, 512, 512)).astype(np.float32), 32, 0, 40, (40, 24), False
    yield "not_converged", rng.normal(size=(4, 128, 193)).astype(np.float32), 32, 0, 4, (4, 4), False


def invalid_options(exe, device, root):
    checks = [
        ("negative", 3, ["--pipeline-jobs", "-1"], "unsupported shape or invalid tuning argument"),
        ("empty_second_group", 3, ["--pipeline-jobs", "3"], "pipeline-jobs must be smaller"),
        ("oversized", 3, ["--pipeline-jobs", "4"], "pipeline-jobs must be smaller"),
        ("singleton_split", 1, ["--pipeline-jobs", "1"], "pipeline-jobs must be smaller"),
        ("conflicting", 3, ["--pipeline-jobs", "1", "--no-overlap"], "unsupported shape or invalid tuning argument"),
    ]
    for name, batch, flags, message in checks:
        cmd = [str(exe.resolve()), str(device), "32", "48", str(batch), "--block", "16"] + flags
        run = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        log_path(root, name + ".log").write_text(run.stdout + run.stderr)
        assert run.returncode != 0 and message in run.stderr, name + ": " + run.stdout + run.stderr
    return {"case": "invalid_options", "pass": True, "rejected": len(checks)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, default=Path("build-fused/jacobi_svd_main"))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/overlap-tests"))
    parser.add_argument("--case", help="run just one named case")
    parser.add_argument("--legacy-exe", type=Path, help="also compare the preserved serial implementation")
    args = parser.parse_args()
    results = []
    for name, a, block, cores, split, groups, fixed in cases():
        if args.case and args.case != name:
            continue
        root = args.artifacts / name
        inp = root / "input"
        inp.mkdir(parents=True, exist_ok=True)
        a.tofile(inp / "A_batch.bin")
        batch, rows, cols = a.shape
        limit = 1 if name == "not_converged" else 30
        common = [str(args.exe.resolve()), str(args.device), str(rows), str(cols), str(batch),
                  str(cores), str(limit), "--block", str(block), "--repeat", "3",
                  "--input-dir", str(inp.resolve())]
        if fixed:
            common.append("--fixed-sweeps")
        result = {"case": name}
        try:
            metrics = {}
            modes = ("overlap", "serial", "legacy") if args.legacy_exe else ("overlap", "serial")
            for mode in modes:
                out = root / mode
                cmd = common + ["--output-dir", str(out.resolve())]
                if mode == "serial":
                    cmd.append("--no-overlap")
                elif mode == "overlap" and split:
                    cmd.extend(["--pipeline-jobs", str(split)])
                if mode == "legacy":
                    cmd[0] = str(args.legacy_exe.resolve())
                run = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
                log_path(root, mode + ".log").write_text(run.stdout + run.stderr)
                assert run.returncode == 0, f"{mode}: {run.stdout} {run.stderr}"
                m = validate_execution(out)
                want_overlap = mode == "overlap" and groups[1] > 0
                assert m.get("phase_overlap") is want_overlap, f"{mode}: no requested overlap: {m}"
                assert m["converged"] is (limit > 1), m
                assert len(m["stream_ms"]) == 3, m
                assert all(f"run={rep} sweeps={m['sweeps']} converged={int(limit > 1)}" in run.stdout for rep in range(3)), run.stdout
                if fixed:
                    assert m["sweeps"] == limit, m
                if mode == "overlap":
                    assert m["pipeline_groups"] == list(groups), m
                elif mode == "serial":
                    assert m["pipeline_groups"] == [sum(groups), 0], m
                metrics[mode] = m
            assert metrics["overlap"]["sweeps"] == metrics["serial"]["sweeps"], metrics
            if args.legacy_exe:
                assert metrics["legacy"]["sweeps"] == metrics["serial"]["sweeps"], metrics
            for filename in ("B_rot.bin", "sigma.bin", "Vt.bin"):
                assert (root / "overlap" / filename).read_bytes() == (root / "serial" / filename).read_bytes(), filename
                if args.legacy_exe:
                    assert (root / "serial" / filename).read_bytes() == (root / "legacy" / filename).read_bytes(), "legacy " + filename
            if limit > 1:
                result.update(validate(a, root / "overlap"))
            else:
                assert metrics["overlap"]["sweeps"] == 1, metrics
            result.update({"pass": True, "execution": metrics, "bitwise_equal": True})
        except (AssertionError, ValueError, OSError, subprocess.TimeoutExpired) as error:
            result.update({"pass": False, "error": str(error)})
        results.append(result)
        print(json.dumps(result), flush=True)
        (args.artifacts / "results.json").write_text(json.dumps(results, indent=2))
    if args.case in (None, "invalid_options"):
        try:
            result = invalid_options(args.exe, args.device, args.artifacts / "invalid_options")
        except (AssertionError, OSError, subprocess.TimeoutExpired) as error:
            result = {"case": "invalid_options", "pass": False, "error": str(error)}
        results.append(result)
        print(json.dumps(result), flush=True)
        args.artifacts.mkdir(parents=True, exist_ok=True)
        (args.artifacts / "results.json").write_text(json.dumps(results, indent=2))
    assert results, "no matching cases"
    return int(any(not result["pass"] for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
