#!/usr/bin/env python3
"""为批量单边（块状）雅可比 SVD 生成输入数据与参考结果。

生成 batch 个 rows x cols 的 float32 矩阵（行主序、连续存放，小端），
写入:

  input/A_batch.bin    batch * rows * cols 个 float32，按矩阵依次连续排列
  input/sigma_ref.bin  batch * rank 个 float32，rank = min(rows, cols)，
                       每个矩阵的奇异值按降序排列（numpy.linalg.svd 参考值）
  input/shape.txt      "rows cols batch" 三个整数，供验证脚本自动读取

用法:
  python3 gen_jacobi_svd_data.py --rows 128 --cols 128 --batch 8
  python3 gen_jacobi_svd_data.py --rows 128 --cols 128 --batch 8 --seed 7 \
                                 --dist normal --cond 1e3

--dist   normal（标准正态，默认）/ uniform（[-1,1] 均匀）
--cond   大于 1 时构造指定条件数的矩阵（通过给定几何衰减的奇异值谱），
         便于检验算法的数值精度；默认 0 表示直接用随机矩阵。
"""

import argparse
import os
import sys

import numpy as np


def build_matrix(rng, rows, cols, dist, cond):
    """生成一个 rows x cols 的 float32 矩阵。"""
    if cond and cond > 1.0:
        # 通过 SVD 构造具有指定条件数的矩阵：A = U diag(s) V^T
        rank = min(rows, cols)
        s = np.geomspace(1.0, 1.0 / cond, rank)
        u, _ = np.linalg.qr(rng.standard_normal((rows, rank)))
        v, _ = np.linalg.qr(rng.standard_normal((cols, rank)))
        a = (u * s) @ v.T
    elif dist == "uniform":
        a = rng.uniform(-1.0, 1.0, size=(rows, cols))
    else:
        a = rng.standard_normal((rows, cols))
    return np.ascontiguousarray(a.astype(np.float32))


def main():
    parser = argparse.ArgumentParser(description="生成批量 SVD 输入与参考奇异值")
    parser.add_argument("--rows", type=int, required=True, help="矩阵行数 M")
    parser.add_argument("--cols", type=int, required=True, help="矩阵列数 N")
    parser.add_argument("--batch", type=int, required=True, help="矩阵个数 B")
    parser.add_argument("--seed", type=int, default=2024, help="随机种子")
    parser.add_argument("--dist", choices=["normal", "uniform"], default="normal")
    parser.add_argument("--cond", type=float, default=0.0, help="目标条件数，0 表示不控制")
    parser.add_argument("--out-dir", default="../input", help="输出目录")
    args = parser.parse_args()

    rows, cols, batch = args.rows, args.cols, args.batch
    if rows <= 0 or cols <= 0 or batch <= 0:
        print("错误: rows/cols/batch 必须为正整数", file=sys.stderr)
        return 1
    if rows > 4096 or cols > 4096:
        print(f"错误: rows/cols 不能超过 4096（当前 {rows}x{cols}）", file=sys.stderr)
        return 1

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    a_path = os.path.join(out_dir, "A_batch.bin")
    s_path = os.path.join(out_dir, "sigma_ref.bin")
    shape_path = os.path.join(out_dir, "shape.txt")

    rank = min(rows, cols)
    rng = np.random.default_rng(args.seed)

    sigma_all = np.empty((batch, rank), dtype=np.float32)
    a_all = np.empty((batch, rows, cols), dtype=np.float32)

    for b in range(batch):
        a = build_matrix(rng, rows, cols, args.dist, args.cond)
        a_all[b] = a
        # 参考奇异值（降序）
        s = np.linalg.svd(a.astype(np.float64), compute_uv=False)
        sigma_all[b, :] = s[:rank].astype(np.float32)

    a_all.tofile(a_path)
    sigma_all.tofile(s_path)
    with open(shape_path, "w", encoding="utf-8") as f:
        f.write(f"{rows} {cols} {batch}\n")

    print(f"矩阵尺寸      : {rows} x {cols}  (rank={rank})")
    print(f"批大小 batch  : {batch}")
    print(f"随机种子      : {args.seed}   分布: {args.dist}   条件数: {args.cond or '未指定'}")
    print(f"输入矩阵      : {a_path}  ({a_all.nbytes} 字节)")
    print(f"参考奇异值    : {s_path}  ({sigma_all.nbytes} 字节)")
    print(f"尺寸标记      : {shape_path}")
    print("参考奇异值示例（前 3 个矩阵）:")
    for b in range(min(batch, 3)):
        print(f"  矩阵 {b}: sigma[0:4]={np.array2string(sigma_all[b, :4], precision=4)}, "
              f"sigma[-1]={sigma_all[b, -1]:.4e}, "
              f"条件数={sigma_all[b, 0] / max(sigma_all[b, -1], 1e-30):.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
