#!/usr/bin/env python3
"""批量单边（块状）雅可比 SVD 结果验证。

读取 NPU 输出的奇异值与"旋转后矩阵"，与 numpy 参考结果逐矩阵比对，
并从三个层面检查分解是否正确:

  1) 奇异值精度   : output/sigma.bin 与 input/sigma_ref.bin（或 numpy 现场计算）
                    按降序逐个比较，给出最大绝对/相对误差。
  2) 行正交性     : 收敛的单边雅可比应使 B 的各行两两正交，
                    检查 B B^T 的非对角元素（相对尺度）。
  3) V^T 正交性   : 检查 output/Vt.bin 是否满足 Vt Vt^T = I。
  4) 重构误差     : 由 B 得到 V^T（行归一化），再算 U = A V S^-1，
                    检查 ||A - U S V^T||_F / ||A||_F。

用法:
  python3 verify_jacobi_svd.py                       # 形状从 input/shape.txt 读取
  python3 verify_jacobi_svd.py --rows 128 --cols 128 --batch 8 \
          --atol 1e-4 --rtol 1e-3 --top 5 --skip-recon

退出码 0 表示全部通过，1 表示存在失败。
"""

import argparse
import os
import sys

import numpy as np


def load_params(args):
    rows, cols, batch = args.rows, args.cols, args.batch
    if None in (rows, cols, batch):
        shape_path = os.path.join(args.input_dir, "shape.txt")
        if not os.path.exists(shape_path):
            print(f"错误: 未指定形状且找不到 {shape_path}", file=sys.stderr)
            sys.exit(1)
        with open(shape_path, "r", encoding="utf-8") as f:
            rows, cols, batch = (int(x) for x in f.read().split()[:3])
        print(f"[自动读取] rows={rows} cols={cols} batch={batch}（来自 {shape_path}）")
    return rows, cols, batch


def main():
    parser = argparse.ArgumentParser(description="验证批量单边雅可比 SVD 的结果")
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--cols", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--input-dir", default="../input")
    parser.add_argument("--output-dir", default="../output")
    parser.add_argument("--atol", type=float, default=1e-4, help="奇异值绝对容差")
    parser.add_argument("--rtol", type=float, default=1e-4, help="奇异值相对容差")
    parser.add_argument("--orth-tol", type=float, default=1e-3, help="行正交性容差")
    parser.add_argument("--recon-tol", type=float, default=1e-3, help="重构相对误差容差")
    parser.add_argument("--vt-tol", type=float, default=1e-3, help="V^T 行正交性容差")
    parser.add_argument("--top", type=int, default=5, help="每个矩阵报告的误差最大元素个数")
    parser.add_argument("--skip-recon", action="store_true", help="跳过大矩阵重构检查")
    args = parser.parse_args()

    rows, cols, batch = load_params(args)
    rank = min(rows, cols)

    a_path = os.path.join(args.input_dir, "A_batch.bin")
    s_ref_path = os.path.join(args.input_dir, "sigma_ref.bin")
    s_out_path = os.path.join(args.output_dir, "sigma.bin")
    b_out_path = os.path.join(args.output_dir, "B_rot.bin")

    for p in (a_path, s_out_path):
        if not os.path.exists(p):
            print(f"错误: 找不到文件 {p}", file=sys.stderr)
            sys.exit(1)

    a_all = np.fromfile(a_path, dtype=np.float32)
    s_out = np.fromfile(s_out_path, dtype=np.float32)
    if a_all.size != batch * rows * cols:
        print(f"错误: {a_path} 元素数 {a_all.size} != {batch}*{rows}*{cols}", file=sys.stderr)
        sys.exit(1)
    if s_out.size != batch * rank:
        print(f"错误: {s_out_path} 元素数 {s_out.size} != {batch}*{rank} = {batch * rank}",
              file=sys.stderr)
        sys.exit(1)

    a_all = a_all.reshape(batch, rows, cols)
    s_out = s_out.reshape(batch, rank)

    if os.path.exists(s_ref_path):
        s_ref = np.fromfile(s_ref_path, dtype=np.float32).reshape(batch, rank)
        ref_src = s_ref_path
    else:
        s_ref = np.empty((batch, rank), dtype=np.float32)
        for b in range(batch):
            s_ref[b] = np.linalg.svd(a_all[b].astype(np.float64), compute_uv=False)[:rank]
        ref_src = "numpy 现场计算"

    b_rot = None
    if os.path.exists(b_out_path):
        b_rot = np.fromfile(b_out_path, dtype=np.float32).reshape(batch, rows, cols)

    vt_path = os.path.join(args.output_dir, "Vt.bin")
    vt_all = None
    if os.path.exists(vt_path):
        vt_all = np.fromfile(vt_path, dtype=np.float32).reshape(batch, rank, cols)

    print(f"形状: rows={rows} cols={cols} batch={batch} rank={rank}")
    print(f"参考奇异值来源: {ref_src}")
    print(f"容差: atol={args.atol:g} rtol={args.rtol:g} "
          f"orth_tol={args.orth_tol:g} recon_tol={args.recon_tol:g}")
    print()
    header = (f"{'矩阵':>5} {'max|Δσ|':>12} {'rel|Δσ|':>10} {'B正交性':>10} "
              f"{'Vt正交性':>10} {'重构误差':>10}  结论")
    print(header)
    print("-" * len(header))

    n_fail = 0
    worst_sigma = (0, 0.0)
    worst_orth = (0, 0.0)
    worst_vt = (0, 0.0)
    worst_recon = (0, 0.0)

    for b in range(batch):
        so = s_out[b].astype(np.float64)
        sr = s_ref[b].astype(np.float64)
        abs_err = np.abs(so - sr)
        max_abs = float(abs_err.max())
        # 相对误差只在数值非零的奇异值上统计（秩亏时零奇异值的相对误差没有意义）
        sig_scale = max(float(sr[0]), 1e-30)
        rel_mask = np.abs(sr) > 1e-6 * sig_scale
        max_rel = (float((abs_err[rel_mask] / np.abs(sr[rel_mask])).max())
                   if rel_mask.any() else 0.0)
        sigma_ok = bool(np.all(abs_err <= args.atol + args.rtol * np.abs(sr)))

        orth = 0.0
        vt_err = 0.0
        recon = 0.0
        recon_ok = True
        if b_rot is not None:
            bm = b_rot[b].astype(np.float64)
            g = bm @ bm.T
            diag = np.diag(g)
            scale = max(float(diag.max()), 1e-30)
            g_off = g - np.diag(diag)
            orth = float(np.abs(g_off).max() / scale)

            if not args.skip_recon:
                # V^T 由旋转后矩阵的行归一化得到（行已近似正交）
                s_ord = np.sort(np.linalg.norm(bm, axis=1))[::-1][:rank]
                eps = 1e-6 * max(float(s_ord[0]), 1e-30)
                keep = np.linalg.norm(bm, axis=1) > eps
                rows_kept = np.argsort(np.linalg.norm(bm, axis=1))[::-1][:rank]
                rows_kept = rows_kept[keep[rows_kept]]
                v = bm[rows_kept] / np.linalg.norm(bm[rows_kept], axis=1, keepdims=True)
                s = np.linalg.norm(bm[rows_kept], axis=1)
                u = a_all[b].astype(np.float64) @ v.T @ np.diag(1.0 / s)
                a_rec = (u * s) @ v
                recon = float(np.linalg.norm(a_all[b].astype(np.float64) - a_rec)
                              / max(np.linalg.norm(a_all[b].astype(np.float64)), 1e-30))
                recon_ok = recon <= args.recon_tol

        if vt_all is not None:
            vt = vt_all[b].astype(np.float64)
            # 零奇异值对应的右奇异向量不唯一，主机侧输出为 0，只检查非零子空间
            active = so > 1e-6 * max(float(so[0]), 1e-30)
            if active.any():
                vt_sub = vt[active]
                gram = vt_sub @ vt_sub.T
                vt_err = float(np.abs(gram - np.eye(int(active.sum()))).max())
            else:
                vt_err = 0.0

        ok = sigma_ok and (orth <= args.orth_tol) and (vt_err <= args.vt_tol) and recon_ok
        if not ok:
            n_fail += 1
        if max_abs > worst_sigma[1]:
            worst_sigma = (b, max_abs)
        if orth > worst_orth[1]:
            worst_orth = (b, orth)
        if vt_err > worst_vt[1]:
            worst_vt = (b, vt_err)
        if recon > worst_recon[1]:
            worst_recon = (b, recon)

        print(f"{b:>5} {max_abs:>12.3e} {max_rel:>10.3e} {orth:>10.3e} "
              f"{vt_err:>10.3e} {recon:>10.3e}  {'PASS' if ok else 'FAIL'}")

        if not sigma_ok:
            top = np.argsort(abs_err)[::-1][:args.top]
            for idx in top:
                print(f"      sigma[{idx:>4}]: ref={sr[idx]:.6e}  out={so[idx]:.6e}  "
                      f"|Δ|={abs_err[idx]:.3e}")

    print("-" * len(header))
    print(f"总结: {batch} 个矩阵中 {n_fail} 个 FAIL")
    print(f"  最大奇异值绝对误差 : {worst_sigma[1]:.3e}  (矩阵 {worst_sigma[0]})")
    if b_rot is not None:
        print(f"  最差 B 行正交性    : {worst_orth[1]:.3e}  (矩阵 {worst_orth[0]})")
    if vt_all is not None:
        print(f"  最差 Vt 行正交性   : {worst_vt[1]:.3e}  (矩阵 {worst_vt[0]})")
    if b_rot is not None and not args.skip_recon:
        print(f"  最差重构相对误差   : {worst_recon[1]:.3e}  (矩阵 {worst_recon[0]})")
    if n_fail == 0:
        print("结论: 全部通过 ✅")
        return 0
    print("结论: 存在不一致 ❌")
    return 1


if __name__ == "__main__":
    sys.exit(main())
