#!/usr/bin/env python3
"""模拟"列块 + 列块对"单边块状雅可比的收敛性（numpy 参考实现）。

严格按描述实现：
  * m x n 矩阵按列宽 b 分成 nb = n/b 个列块；
  * 一轮 sweep：
      1) 块内：对每个列块，用 Jacobi 扫描该块 b 列的 Gram（b x b）得到 b x b 的
         旋转矩阵 V，作用于该块（A_P <- A_P V）；
      2) 块间：对每个列块对 (P,Q)，用 Jacobi 扫描 [A_P, A_Q] 的 2b x 2b Gram 得到
         2b x 2b 的旋转矩阵 V，作用于该块对；
  * 一轮 sweep 内每个列对恰好被用到一次（Jacobi 扫描按 round-robin 匹配，两两一次）。

用法:
  python3 sim_block_jacobi.py --m 256 --n 256 --b 128 --sweeps 10
  python3 sim_block_jacobi.py --m 256 --n 384 --b 128 --sweeps 10 --pairs cross
  --pairs all   块间那一步对整个 2b x 2b Gram 做 Jacobi（含块内对，默认）
  --pairs cross 块间那一步只做 P 列与 Q 列之间的旋转（不重复块内）
"""

import argparse

import numpy as np


def rotate_gram(G, V, i, j):
    """在 Gram 空间里消去 G[i,j]：G <- J^T G J，同时累积 V <- V J。"""
    a, b, c = G[i, i], G[j, j], G[i, j]
    if abs(c) <= 1e-12 * np.sqrt(max(a * b, 1e-300)):
        return
    tau = (b - a) / (2.0 * c)
    t = 1.0 / (tau + np.sqrt(1.0 + tau * tau)) if tau >= 0 else 1.0 / (tau - np.sqrt(1.0 + tau * tau))
    cs = 1.0 / np.sqrt(1.0 + t * t)
    sn = cs * t

    gi = G[i].copy()
    gj = G[j].copy()
    G[i, :] = cs * gi - sn * gj
    G[j, :] = sn * gi + cs * gj
    gi = G[:, i].copy()
    gj = G[:, j].copy()
    G[:, i] = cs * gi - sn * gj
    G[:, j] = sn * gi + cs * gj

    vi = V[:, i].copy()
    vj = V[:, j].copy()
    V[:, i] = cs * vi - sn * vj
    V[:, j] = sn * vi + cs * vj


def jacobi_sweep(G, n, cross_split=None):
    """对 n x n 的 Gram 矩阵做一轮 round-robin Jacobi，返回旋转矩阵 V (n x n)。

    cross_split=(b,)：只旋转"前半 b 列"与"后半 b 列"之间的对（块间只做跨块旋转）。
    """
    V = np.eye(n)
    if cross_split is None:
        for r in range(n):
            for i in range(n):
                j = (r - i) % n
                if i < j:
                    rotate_gram(G, V, i, j)
    else:
        b = cross_split[0]
        for r in range(b):
            for i in range(b):
                j = b + ((r - i) % b)     # i 属于 P，j 属于 Q
                rotate_gram(G, V, i, j)
    return V


def one_sweep(A, b, pairs):
    m, n = A.shape
    nb = n // b

    # 1) 块内旋转
    for p in range(nb):
        s = p * b
        X = A[:, s:s + b]
        G = X.T @ X
        V = jacobi_sweep(G, b)
        A[:, s:s + b] = X @ V

    # 2) 块间旋转
    for p in range(nb):
        for q in range(p + 1, nb):
            sp, sq = p * b, q * b
            W = np.concatenate([A[:, sp:sp + b], A[:, sq:sq + b]], axis=1)
            G = W.T @ W
            V = jacobi_sweep(G, 2 * b, cross_split=(b,) if pairs == "cross" else None)
            AW = W @ V
            A[:, sp:sp + b] = AW[:, :b]
            A[:, sq:sq + b] = AW[:, b:]
    return A


def report(A, s_ref, tag):
    G = A.T @ A
    d = np.diag(G)
    off = np.abs(G - np.diag(d)).max() / max(np.abs(d).max(), 1e-300)
    k = s_ref.size                      # rank = min(m, n)
    sv = np.sort(np.linalg.norm(A, axis=0))[::-1][:k]
    err = np.abs(sv - s_ref).max()
    print(f"{tag:<28} 非对角={off:.3e}  sigma 误差={err:.3e}  "
          f"||A||_F={np.linalg.norm(A):.6f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=256)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--b", type=int, default=128)
    ap.add_argument("--sweeps", type=int, default=10)
    ap.add_argument("--pairs", choices=["all", "cross"], default="all")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    m, n, b = args.m, args.n, args.b
    assert n % b == 0, "n 必须是 b 的整数倍"
    rng = np.random.default_rng(args.seed)
    A = rng.standard_normal((m, n)).astype(np.float64)
    s_ref = np.linalg.svd(A, compute_uv=False)
    nb = n // b
    npairs = nb * (nb - 1) // 2

    print(f"矩阵 {m}x{n}, 块宽 b={b}, 列块数={nb}, 列块对数={npairs}, "
          f"块间策略={args.pairs}")
    print(f"||A||_F = {np.linalg.norm(A):.6f}, 初始非对角 = "
          f"{np.abs(A.T @ A - np.diag(np.diag(A.T @ A))).max() / np.abs(np.diag(A.T @ A)).max():.3e}")
    print()
    for s in range(args.sweeps):
        A = one_sweep(A, b, args.pairs)
        report(A, s_ref, f"sweep {s + 1}")


if __name__ == "__main__":
    main()
