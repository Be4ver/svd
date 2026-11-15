#!/usr/bin/env python3
"""模拟两种优化对收敛性和开销的影响。

优化 1（批量归约）：把同一轮匹配里若干不相交对的 (a,b,c) 一次性算完，只做一次
    标量读回同步；数学上与逐对计算完全等价。
优化 2（Cube 算 Gram）：每一步先由 AIC 算 G = A A^T（128x128x128 矩阵乘），
    AIV 只从 G 里取 (G[i,i], G[p,p], G[i,p]) 算角度，不再做点积归约。

本脚本对比：
  (a) 逐对现算点积（当前实现）
  (b) 每步用 AIC 算出的 Gram（等价于 (a)，验证数值一致）
并给出指令/同步次数的开销模型。
"""

import argparse

import numpy as np


def givens(a, b, c):
    if abs(c) <= 1e-9 * np.sqrt(a * b):
        return 1.0, 0.0
    tau = (b - a) / (2.0 * c)
    t = 1.0 / (tau + np.sqrt(1.0 + tau * tau)) if tau >= 0 else 1.0 / (tau - np.sqrt(1.0 + tau * tau))
    cs = 1.0 / np.sqrt(1.0 + t * t)
    return cs, cs * t


def sweep_ondemand(A, n):
    """(a) 逐对现算点积：每次旋转都重新算 a,b,c"""
    for step in range(n):
        A2 = A.copy()
        for i in range(n):
            p = (step - i) % n
            if i >= p:
                continue
            a = float(A[i] @ A[i])
            b = float(A[p] @ A[p])
            c = float(A[i] @ A[p])
            cs, sn = givens(a, b, c)
            A2[i] = cs * A[i] - sn * A[p]
            A2[p] = sn * A[i] + cs * A[p]
        A = A2
    return A


def sweep_from_gram(A, n):
    """(b) 每步先用 AIC 算 G=A A^T，再从 G 取角度（等价于 (a)）"""
    for step in range(n):
        G = A @ A.T                      # AIC: 128x128x128 矩阵乘
        A2 = A.copy()
        for i in range(n):
            p = (step - i) % n
            if i >= p:
                continue
            cs, sn = givens(G[i, i], G[p, p], G[i, p])
            A2[i] = cs * A[i] - sn * A[p]
            A2[p] = sn * A[i] + cs * A[p]
        A = A2
    return A


def report(A, s_ref, tag):
    G = A @ A.T
    d = np.diag(G)
    off = np.abs(G - np.diag(d)).max() / np.abs(d).max()
    err = np.abs(np.sort(np.linalg.norm(A, axis=1))[::-1][:s_ref.size] - s_ref).max()
    print(f"{tag:<26} 非对角={off:.3e}  sigma 误差={err:.3e}  ||A||_F={np.linalg.norm(A):.6f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--sweeps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    n = args.n
    rng = np.random.default_rng(args.seed)
    base = rng.standard_normal((n, n)).astype(np.float64)
    s_ref = np.linalg.svd(base, compute_uv=False)

    A1 = base.copy()
    A2 = base.copy()
    print(f"矩阵 {n}x{n}，每 sweep {n} 步，每步 {n // 2} 对")
    for s in range(args.sweeps):
        A1 = sweep_ondemand(A1, n)
        A2 = sweep_from_gram(A2, n)
        report(A1, s_ref, f"sweep {s + 1} 逐对现算")
        report(A2, s_ref, f"         Gram(AIC)")
        print(f"         两者最大差 = {np.abs(A1 - A2).max():.3e}")

    # 开销模型（每 sweep，n=128）
    npair = n * (n - 1) // 2
    print("\n=== 每 sweep 开销模型（n=128，8128 对）===")
    print(f"当前：每对 3 次 ReduceSum + 1 次 V->S 同步  ->  {3 * npair} 次归约, {npair} 次同步")
    print(f"优化1：每 8 对 1 次同步                     ->  {3 * npair} 次归约, {npair // 8} 次同步")
    print(f"优化2：每步 1 次 Gram 矩阵乘 + 读回        ->  {n} 次矩阵乘(128^3), 0 次归约, {n} 次同步")


if __name__ == "__main__":
    main()
