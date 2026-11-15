#!/usr/bin/env python3
"""验证 QDWH 极分解 + SDC 批处理结果。

检查项：
  1. QDWH：U 的正交性、H 对称性/半正定性、A ~= U H、与参考 U/H 的误差。
  2. SDC：每个叶块 n < threshold、对称、半正定；同一 root 的叶块特征值
     并集与 H 的特征值一致；trace / Frobenius 范数守恒。
"""
import argparse
import os
import sys

import numpy as np

TOL = 5e-3          # QDWH 极分解
TOL_LEAF = 1e-2     # SDC 递归（float32，多级累积）
TOL_SYM = 1e-3
TOL_PSD = 1e-4


def read_exact(path, count, dtype=np.float32):
    if not os.path.exists(path):
        raise SystemExit(f"缺少文件: {path}")
    a = np.fromfile(path, dtype=dtype)
    if a.size < count:
        raise SystemExit(f"{path}: 只有 {a.size} 个元素，期望 {count}")
    return a[:count]


def read_shapes(path):
    shapes = []
    with open(path) as f:
        for line in f:
            line = line.split("#")[0].replace(",", " ").split()
            if len(line) >= 2:
                m, n = int(line[0]), int(line[1])
                if m > 0 and n > 0:
                    shapes.append((m, n))
    return shapes


def psd_report(a):
    w = np.linalg.eigvalsh((a + a.T) * 0.5)
    scale = max(1.0, float(np.max(np.abs(w))))
    return float(w[0]) / scale, float(w[-1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", default="input")
    p.add_argument("--output-dir", default="output")
    p.add_argument("--threshold", type=int, default=256)
    p.add_argument("--max-eig-n", type=int, default=4096,
                   help="root 超过该尺寸时跳过全谱对比（CPU eig 太慢）")
    p.add_argument('--strict', action='store_true', help='Y_ITER_AIC_TASK acceptance: eig <=3e-7, u <=1e-6, leaf sym=0')
    args = p.parse_args()
    global TOL,TOL_LEAF,TOL_SYM
    if args.strict: TOL=1e-6; TOL_LEAF=3e-7; TOL_SYM=1e-6

    shapes = read_shapes(os.path.join(args.input_dir, "shapes.txt"))
    if not shapes:
        raise SystemExit("shapes.txt 为空")
    batch = len(shapes)

    a_count = sum(m * n for m, n in shapes)
    h_count = sum(min(m, n) * min(m, n) for m, n in shapes)
    a_host = read_exact(os.path.join(args.input_dir, "matrix.bin"), a_count)
    u_ref = read_exact(os.path.join(args.input_dir, "u_ref.bin"), a_count)
    h_ref = read_exact(os.path.join(args.input_dir, "h_ref.bin"), h_count)
    u_dev = read_exact(os.path.join(args.output_dir, "u.bin"), a_count)
    h_dev = read_exact(os.path.join(args.output_dir, "h.bin"), h_count)

    max_n = max(n for _, n in shapes)
    leaf_meta_path = os.path.join(args.output_dir, "leaf_meta.txt")
    if not os.path.exists(leaf_meta_path):
        raise SystemExit(f"缺少 {leaf_meta_path}")
    leaf_meta = [tuple(int(v) for v in line.split())
                 for line in open(leaf_meta_path) if line.strip()]
    leaf_raw = np.fromfile(os.path.join(args.output_dir, "leaf_h.bin"),
                           dtype=np.float32)
    need_leaf = max((off + n * n for _, n, off in leaf_meta), default=0)
    if leaf_raw.size < need_leaf:
        raise SystemExit(f"leaf_h.bin 只有 {leaf_raw.size}，期望至少 {need_leaf}")

    ok = True
    print("=" * 96)
    print("QDWH: A = U H")
    print(f"{'#':>2} {'M':>5} {'N':>5} {'u_err':>10} {'orth':>10} "
          f"{'h_err':>10} {'h_sym':>10} {'min_eig':>11} {'rec':>10}  status")
    a_off = h_off = 0
    for i, (M, N) in enumerate(shapes):
        K = min(M, N)
        A = a_host[a_off:a_off + M * N].reshape(M, N).astype(np.float64)
        U = u_dev[a_off:a_off + M * N].reshape(M, N).astype(np.float64)
        Ur = u_ref[a_off:a_off + M * N].reshape(M, N).astype(np.float64)
        H = h_dev[h_off:h_off + K * K].reshape(K, K).astype(np.float64)
        Hr = h_ref[h_off:h_off + K * K].reshape(K, K).astype(np.float64)

        if M >= N:
            orth = np.linalg.norm(U.T @ U - np.eye(N)) / np.sqrt(N)
        else:
            orth = np.linalg.norm(U @ U.T - np.eye(M)) / np.sqrt(M)
        h_sym = np.linalg.norm(H - H.T) / max(1e-30, np.linalg.norm(H))
        min_eig, _ = psd_report(H)
        # M>=N: A=U H；M<N: A=H U（U=U_T^T, H=(AA^T)^{1/2}）
        rec = (np.linalg.norm(A - U @ H) if M >= N
               else np.linalg.norm(A - H @ U)) / max(1e-30, np.linalg.norm(A))
        u_err = np.linalg.norm(U - Ur) / max(1e-30, np.linalg.norm(Ur))
        h_err = np.linalg.norm(H - Hr) / max(1e-30, np.linalg.norm(Hr))

        good = (u_err < TOL and h_err < TOL and orth < TOL and
                h_sym < TOL_SYM and min_eig > -TOL_PSD and rec < TOL)
        ok &= good
        print(f"{i:>2} {M:>5} {N:>5} {u_err:>10.3e} {orth:>10.3e} "
              f"{h_err:>10.3e} {h_sym:>10.3e} {min_eig:>11.3e} "
              f"{rec:>10.3e}  {'PASS' if good else 'FAIL'}")
        a_off += M * N
        h_off += K * K

    print("=" * 96)
    print("SDC leaves")
    by_root = {}
    for root, n, off in leaf_meta:
        by_root.setdefault(root, []).append((n, off))

    print(f"{'root':>4} {'N':>5} {'#leaf':>5} {'max_leaf':>8} "
          f"{'eig_err':>10} {'trace_err':>10} {'frob_err':>10} "
          f"{'sym_max':>10} {'min_eig':>11}  status")
    h_off = 0
    for i, (M, N) in enumerate(shapes):
        K = min(M, N)
        H = h_dev[h_off:h_off + K * K].reshape(K, K).astype(np.float64)
        leaves = by_root.get(i, [])
        mats = []
        sym_max = 0.0
        eig_min = 0.0
        for n, off in leaves:
            L = leaf_raw[off:off + n * n].reshape(n, n).astype(np.float64)
            mats.append(L)
            sym_max = max(sym_max, np.linalg.norm(L - L.T) /
                          max(1e-30, np.linalg.norm(L)))
            eig_min = min(eig_min, psd_report(L)[0])

        trace_err = abs(sum(np.trace(L) for L in mats) - np.trace(H)) / \
            max(1e-30, abs(np.trace(H)))
        frob_err = abs(np.sqrt(sum(np.sum(L * L) for L in mats)) -
                       np.linalg.norm(H)) / max(1e-30, np.linalg.norm(H))
        if K <= args.max_eig_n:
            target = np.linalg.eigvalsh((H + H.T) * 0.5)
            got = np.sort(np.concatenate(
                [np.linalg.eigvalsh((L + L.T) * 0.5) for L in mats]))
            eig_err = float(np.max(np.abs(got - target))) / \
                max(1.0, float(np.max(np.abs(target))))
        else:
            eig_err = float("nan")

        leaf_ok = (len(leaves) > 0 and
                   all(n <= args.threshold for n, _ in leaves) and
                   sym_max < TOL_SYM and eig_min > -TOL_PSD and
                   trace_err < TOL_LEAF and frob_err < TOL_LEAF and
                   (np.isnan(eig_err) or eig_err < TOL_LEAF))
        if args.strict:
            leaf_ok &= sym_max == 0.0 and np.isfinite(eig_err)
            if args.threshold==64 and shapes==[(1024,1024)]:
                leaf_ok &= len(leaves)==23 and max(n for n,_ in leaves)==63
        ok &= leaf_ok
        max_leaf = max(n for n, _ in leaves) if leaves else 0
        print(f"{i:>4} {K:>5} {len(leaves):>5} {max_leaf:>8} "
              f"{eig_err:>10.3e} {trace_err:>10.3e} {frob_err:>10.3e} "
              f"{sym_max:>10.3e} {eig_min:>11.3e}  "
              f"{'PASS' if leaf_ok else 'FAIL'}")
        h_off += K * K

    total_leaves = len(leaf_meta)
    print("-" * 96)
    print(f"roots={batch}, leaves={total_leaves}, "
          f"threshold={args.threshold}, "
          f"all_leaves_small={all(n <= args.threshold for _, n, _ in leaf_meta)}")
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
