#!/usr/bin/env python3
"""Run the real ACL CLI and independently validate its binary outputs with NumPy.

Run on the Ascend host after sourcing set_env.sh. Artifacts are kept per case.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np


def log_path(directory, filename):
    """Keep run logs under the project logs directory, preserving case paths."""
    project = Path(__file__).resolve().parents[1]
    directory = Path(directory).resolve()
    try:
        relative = directory.relative_to(project)
    except ValueError:
        digest = hashlib.sha256(str(directory).encode("utf-8")).hexdigest()[:16]
        relative = Path("external") / digest / directory.name
    destination = project / "logs" / relative / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def validate_execution(output):
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics.get("kernel_launches_per_run") == 1, "SVD must use one kernel launch per run"
    assert metrics.get("device_preprocess") is True, "normalization/padding must run on device"
    assert metrics.get("device_postprocess") is True, "norm/sort/output must run on device"
    assert metrics.get("device_convergence") is True, "sweep control must run on device"
    assert metrics["sweeps"] >= 1
    return metrics


def validate(a, output):
    batch, m, n = a.shape
    rank = min(m, n)
    b = np.fromfile(output / "B_rot.bin", np.float32).reshape(batch, m, n).astype(np.float64)
    s = np.fromfile(output / "sigma.bin", np.float32).reshape(batch, rank).astype(np.float64)
    v = np.fromfile(output / "Vt.bin", np.float32).reshape(batch, rank, n).astype(np.float64)
    if not all(np.isfinite(x).all() for x in (b, s, v)):
        raise AssertionError("nonfinite output")
    sr = np.linalg.svd(a.astype(np.float64), compute_uv=False)
    reference_scale = np.maximum(sr[:, :1], np.finfo(np.float64).tiny)
    np.testing.assert_allclose(s / reference_scale, sr / reference_scale, atol=1e-4, rtol=1e-4)
    orth, vtorth, recon, energy, eigen = [], [], [], [], []
    for ai, bi, si, vi, sri in zip(a.astype(np.float64), b, s, v, sr):
        g = bi @ bi.T
        orth.append(float(np.abs(g - np.diag(np.diag(g))).max() / max(np.diag(g).max(), np.finfo(np.float64).tiny)))
        active = si > 1e-6 * si[0]
        va = vi[active]
        vtorth.append(float(np.abs(va @ va.T - np.eye(len(va))).max(initial=0)))
        denom = np.linalg.norm(ai) or 1.0
        recon.append(float(np.linalg.norm(ai - ai @ va.T @ va) / denom))
        energy.append(float(abs(np.linalg.norm(ai) - np.linalg.norm(bi)) / denom))
        # Projection alone accepts arbitrary orthogonal Vt for full-rank square
        # inputs. Use the spectral norm so the threshold does not weaken with
        # rank (Frobenius normalization hid wrong eigenpairs at large sizes).
        spectral = max(float(sri[0]), np.finfo(np.float64).tiny)
        an = ai / spectral
        residual = vi @ (an.T @ an) - (si / spectral)[:, None] ** 2 * vi
        eigen.append(float(np.linalg.norm(residual, axis=1).max(initial=0)))
    result = {"sigma_max_abs": float(np.abs(s - sr).max()), "b_orth": max(orth),
              "vt_orth": max(vtorth), "reconstruction": max(recon), "energy_relative": max(energy),
              "right_eigen_residual": max(eigen),
              "right_eigen_normalization": "reference_spectral_norm_squared"}
    assert result["b_orth"] < 1e-3, result
    assert result["vt_orth"] < 1e-3, result
    assert result["reconstruction"] < 1e-3, result
    assert result["energy_relative"] < 1e-4, result
    assert result["right_eigen_residual"] < 1e-3, result
    return result


def cases():
    rng = np.random.default_rng(20261116)
    for m, n, batch in [(1, 1, 1), (1, 256, 3), (128, 1, 3), (7, 9, 5),
                        (20, 20, 44), (32, 32, 40), (37, 19, 8), (64, 81, 12),
                        (65, 33, 5), (128, 128, 10), (128, 256, 10),
                        (512, 512, 1)]:
        yield f"random_{m}x{n}_b{batch}", rng.normal(size=(batch, m, n)).astype(np.float32), 16
    for block in (16, 32):
        shape = (4, 64, 81)
        for kind in ("zero", "identity", "repeated", "rank", "condition1e4"):
            a = rng.normal(size=shape).astype(np.float32)
            if kind == "zero": a[:] = 0
            if kind in ("identity", "repeated"):
                a[:] = np.eye(64, 81, dtype=np.float32) * (3 if kind == "repeated" else 1)
            if kind == "rank": a[:, 32:] = a[:, :1]
            if kind == "condition1e4":
                for i in range(len(a)):
                    q, _ = np.linalg.qr(rng.normal(size=(64, 64)))
                    v, _ = np.linalg.qr(rng.normal(size=(81, 64)))
                    a[i] = (q * np.geomspace(1, 1e-4, 64)) @ v.T
            yield f"{kind}_block{block}", a, block
        yield f"random_128x256_block{block}", rng.normal(size=(8, 128, 256)).astype(np.float32), block
    # Per-matrix normalization must handle mixed scales in one batch.
    for scale in (1e10, 1e-16, 1e30, 1e-30, 1e-38):
        yield f"correlated_scale_{scale:g}", np.array([[[scale, 0], [scale, 0]]], np.float32), 16
    a = rng.normal(size=(5, 32, 48)).astype(np.float32)
    a *= np.array([1e-30, 1e-16, 1, 1e10, 1e30], np.float32)[:, None, None]
    yield "mixed_scale_batch", a, 16


def overflow_case(exe, device, artifacts):
    """A finite input whose FP32 singular value is not representable must fail."""
    root = artifacts / "unrepresentable_sigma"
    inp, out = root / "input", root / "output"
    inp.mkdir(parents=True, exist_ok=True)
    np.array([[[np.finfo(np.float32).max, np.finfo(np.float32).max]]], np.float32).tofile(inp / "A_batch.bin")
    cmd = [str(exe.resolve()), str(device), "1", "2", "1", "0", "30",
           "--input-dir", str(inp.resolve()), "--output-dir", str(out.resolve())]
    run = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    passed = run.returncode != 0 and "exceeds FP32 output range" in run.stderr
    log_path(root, "run.log").write_text(run.stdout + run.stderr)
    return {"case": "unrepresentable_sigma", "pass": passed,
            "error": None if passed else f"unexpected exit={run.returncode}: {run.stdout} {run.stderr}"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, default=Path("build-fused/jacobi_svd_main"))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/npu-tests"))
    args = parser.parse_args()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    results = []
    for name, a, block in cases():
        root = args.artifacts / name
        inp, out = root / "input", root / "output"
        inp.mkdir(parents=True, exist_ok=True)
        a.tofile(inp / "A_batch.bin")
        batch, m, n = a.shape
        cmd = [str(args.exe.resolve()), str(args.device), str(m), str(n), str(batch), "0", "30",
               "--block", str(block), "--input-dir", str(inp.resolve()), "--output-dir", str(out.resolve())]
        run = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        log_path(root, "run.log").write_text(run.stdout + run.stderr)
        result = {"case": name, "shape": list(a.shape), "block": block}
        try:
            if run.returncode: raise AssertionError(f"CLI exit={run.returncode}: {run.stdout} {run.stderr}")
            if "[ERROR]" in run.stdout or "RegisterAscendBinary" in run.stdout:
                raise AssertionError("CANN registration/launch error")
            result.update(validate(a, out))
            result.update(validate_execution(out))
            result["pass"] = True
        except (AssertionError, ValueError, OSError) as error:
            result.update({"pass": False, "error": str(error)})
        results.append(result)
        print(name, "PASS" if result["pass"] else "FAIL", json.dumps(result), flush=True)
        (args.artifacts / "results.json").write_text(json.dumps(results, indent=2))
    failed = sum(not r["pass"] for r in results)
    overflow = overflow_case(args.exe, args.device, args.artifacts)
    results.append(overflow)
    print("unrepresentable_sigma", "PASS" if overflow["pass"] else "FAIL", json.dumps(overflow), flush=True)
    (args.artifacts / "results.json").write_text(json.dumps(results, indent=2))
    failed += int(not overflow["pass"])
    print(f"{len(results) - failed}/{len(results)} passed")
    return bool(failed)


if __name__ == "__main__":
    raise SystemExit(main())
