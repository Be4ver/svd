#!/usr/bin/env python3
"""生成 batch 矩阵 + QDWH 参考解。

每个矩阵用受控奇异值构造：
    A = U_k diag(sigma) V_k^T,   k = min(M, N)
    sigma = alpha * logspace(log10(sigma_max), log10(sigma_max/cond), k)
因此 sigma_max = alpha * sigma_max_ratio < alpha（默认 alpha=1.0），
满足 QDWH 对 "最大奇异值上界 < 1" 的默认假设。

输出（默认 ./input）：
    matrix.bin   batch 个 M x N float32，行主序连续
    u_ref.bin    极分解 A = U H 的 U，M x N
    h_ref.bin    PSD 参考解，边长为 min(M,N)（M>=N: H=U^T A；M<N: (AA^T)^{1/2}）
    shapes.txt   每行 "M N"
    meta.txt     谱信息

用法：
    python3 gen_data.py --shapes 512x512,640x512
    python3 gen_data.py --batch 2 --max-tiles 6
"""
import argparse
import os

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="生成 QDWH+SDC batch 测试数据")
    p.add_argument("--shapes", default="",
                   help='逗号分隔的 "MxN" 列表，例如 512x512,640x512；为空时随机生成')
    p.add_argument("--batch", type=int, default=2, help="随机模式的矩阵个数")
    p.add_argument("--max-tiles", type=int, default=5,
                   help="随机模式下每个维度 tile 数上界（不含），尺寸 = tile*128")
    p.add_argument("--tile", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="input")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--sigma-max", type=float, default=0.9,
                   help="sigma_max/alpha，默认 0.9")
    p.add_argument("--cond", type=float, default=1e3)
    return p.parse_args()


def orthonormal(rows, cols, rng):
    a = rng.standard_normal((rows, cols))
    q, _ = np.linalg.qr(a, mode="reduced")
    return q


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.shapes.strip():
        shapes = []
        for item in args.shapes.split(","):
            m_s, n_s = item.lower().split("x")
            shapes.append((int(m_s), int(n_s)))
    else:
        rng = np.random.default_rng(args.seed)
        shapes = []
        for _ in range(args.batch):
            mt, nt = (int(v) for v in rng.integers(1, args.max_tiles, size=2))
            shapes.append((args.tile * mt, args.tile * nt))

    for (m, n) in shapes:
        if not (0 < m < 8192 and 0 < n < 8192):
            raise SystemExit(f"shape {m}x{n} 越界：要求 0 < M,N < 8192")
    if not (0.0 < args.sigma_max < 1.0):
        raise SystemExit("--sigma-max 必须在 (0,1) 内")

    rng = np.random.default_rng(args.seed)
    paths = {name: os.path.join(args.out_dir, name)
             for name in ("matrix.bin", "u_ref.bin", "h_ref.bin",
                          "shapes.txt", "meta.txt")}
    total = 0
    with open(paths["matrix.bin"], "wb") as fm, \
            open(paths["u_ref.bin"], "wb") as fu, \
            open(paths["h_ref.bin"], "wb") as fh, \
            open(paths["shapes.txt"], "w") as fs, \
            open(paths["meta.txt"], "w", encoding="utf-8") as fmeta:
        fmeta.write("# idx M N sigma_min sigma_max cond\n")
        for i, (M, N) in enumerate(shapes, 1):
            k = min(M, N)
            Uk = orthonormal(M, k, rng)
            Vk = orthonormal(N, k, rng)
            smin_ratio = max(args.sigma_max / args.cond, 1e-10)
            s = args.alpha * np.logspace(np.log10(args.sigma_max),
                                         np.log10(smin_ratio), k)
            A = (Uk * s) @ Vk.T
            U_ref = Uk @ Vk.T
            # PSD 矩阵取 min(M,N) 边：M>=N 时 H=U^T A=(V S V^T)；
            # M<N 时算法先转置，得到 H=(A A^T)^{1/2}=U S U^T（M x M）。
            if M >= N:
                H_ref = (Vk * s) @ Vk.T
            else:
                H_ref = (Uk * s) @ Uk.T

            A.astype(np.float32).tofile(fm)
            U_ref.astype(np.float32).tofile(fu)
            H_ref.astype(np.float32).tofile(fh)
            fs.write(f"{M} {N}\n")
            total += A.size
            fmeta.write(f"{i} {M} {N} {s[-1]:.6e} {s[0]:.6e} "
                        f"{s[0] / s[-1]:.6e}\n")
            print(f"  [{i}/{len(shapes)}] {M} x {N}  "
                  f"sigma=[{s[-1]:.3e}, {s[0]:.3e}]  "
                  f"cond={s[0] / s[-1]:.1e}")

    print(f"完成: {len(shapes)} 个矩阵, {total} float32 -> {args.out_dir}/")
    print(f"shapes: {paths['shapes.txt']}")


if __name__ == "__main__":
    main()
