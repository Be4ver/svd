#!/usr/bin/env python3
"""Check fused QDWH/SDC + Jacobi singular values against CPU leaf spectra."""
import argparse
import os
import sys
import numpy as np


def load_meta(path, cols, label):
    if not os.path.isfile(path):
        raise SystemExit(f"missing {label} metadata: {path}")
    rows = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            fields = line.split()
            if not fields or fields[0].startswith("#"):
                continue
            if len(fields) != cols:
                raise SystemExit(f"{path}:{line_no}: expected {cols} columns, got {len(fields)}")
            try:
                rows.append(tuple(int(x) for x in fields))
            except ValueError as exc:
                raise SystemExit(f"{path}:{line_no}: metadata must be integers") from exc
    return rows


def load_f32(path):
    if not os.path.isfile(path):
        raise SystemExit(f"missing required output: {path}")
    if os.path.getsize(path) % 4:
        raise SystemExit(f"{path}: byte size is not a multiple of float32")
    return np.fromfile(path, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--threshold", type=int, default=64)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args()
    if args.threshold <= 0 or args.atol < 0 or args.rtol < 0:
        raise SystemExit("threshold must be positive and tolerances nonnegative")

    out = args.output_dir
    leaf_meta = load_meta(os.path.join(out, "leaf_meta.txt"), 3, "leaf")
    sigma_meta = load_meta(os.path.join(out, "sigma_meta.txt"), 4, "sigma")
    leaf_h = load_f32(os.path.join(out, "leaf_h.bin"))
    sigma = load_f32(os.path.join(out, "sigma.bin"))
    if len(leaf_meta) != len(sigma_meta):
        raise SystemExit(f"leaf metadata count {len(leaf_meta)} != sigma metadata count {len(sigma_meta)}")
    if not leaf_meta:
        raise SystemExit("no SDC leaves found")

    leaf_ranges, sigma_ranges = [], []
    expected_total = 0
    for i, ((root, n, leaf_off), (sroot, sn, sigma_off, count)) in enumerate(zip(leaf_meta, sigma_meta)):
        if root != sroot or n != sn:
            raise SystemExit(f"leaf {i}: metadata mismatch leaf=({root},{n}) sigma=({sroot},{sn})")
        if n <= 0 or n > args.threshold:
            raise SystemExit(f"leaf {i}: invalid size {n} for threshold {args.threshold}")
        if count != n:
            raise SystemExit(f"leaf {i}: sigma count {count} != leaf dimension {n}")
        if leaf_off < 0 or leaf_off + n * n > leaf_h.size:
            raise SystemExit(f"leaf {i}: matrix range [{leaf_off},{leaf_off+n*n}) outside leaf_h ({leaf_h.size})")
        if sigma_off < 0 or sigma_off + count > sigma.size:
            raise SystemExit(f"leaf {i}: sigma range [{sigma_off},{sigma_off+count}) outside sigma ({sigma.size})")
        leaf_ranges.append((leaf_off, leaf_off + n * n, i))
        sigma_ranges.append((sigma_off, sigma_off + count, i))
        expected_total += count

    for label, ranges in (("leaf matrix", leaf_ranges), ("sigma", sigma_ranges)):
        ranges.sort()
        for prev, cur in zip(ranges, ranges[1:]):
            if prev[1] > cur[0]:
                raise SystemExit(f"overlapping {label} ranges for leaves {prev[2]} and {cur[2]}")
    if sigma.size != expected_total:
        raise SystemExit(f"sigma.bin has {sigma.size} floats; metadata describes {expected_total}")

    max_abs = max_rel = 0.0
    ok = True
    for i, ((root, n, leaf_off), (_, _, sigma_off, count)) in enumerate(zip(leaf_meta, sigma_meta)):
        raw = leaf_h[leaf_off:leaf_off + n * n].reshape(n, n).astype(np.float64)
        sym = 0.5 * (raw + raw.T)
        expected = np.sort(np.abs(np.linalg.eigvalsh(sym)))[::-1]
        got = sigma[sigma_off:sigma_off + count].astype(np.float64)
        abs_errs = np.abs(got - expected)
        scale = max(float(expected[0]) if expected.size else 0.0, 1e-30)
        rel_errs = abs_errs / scale
        abs_err = float(abs_errs.max(initial=0.0))
        rel_err = float(rel_errs.max(initial=0.0))
        sorted_ok = bool(np.all(got[:-1] >= got[1:]))
        finite_ok = bool(np.all(np.isfinite(got)))
        leaf_ok = finite_ok and sorted_ok and bool(np.allclose(got, expected, atol=args.atol, rtol=args.rtol))
        ok &= leaf_ok
        max_abs = max(max_abs, abs_err)
        max_rel = max(max_rel, rel_err)
        print(f"leaf={i} root={root} n={n} max_abs={abs_err:.3e} max_rel={rel_err:.3e} sorted={sorted_ok} {'PASS' if leaf_ok else 'FAIL'}")

    print(f"sigma_count={expected_total} leaves={len(leaf_meta)} max_abs={max_abs:.3e} max_rel={max_rel:.3e} tolerance=(atol={args.atol:g}, rtol={args.rtol:g})")
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
