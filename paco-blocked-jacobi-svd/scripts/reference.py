"""FP32 CPU model of the packed-panel one-sided blocked algorithm."""
import numpy as np


def tournament(n):
    if n < 2 or n % 2:
        raise ValueError("block count must be even and >= 2")
    order = list(range(n))
    result = []
    for _ in range(n - 1):
        result.append([(order[i], order[n - 1 - i]) for i in range(n // 2)])
        order = [order[0], order[-1], *order[1:-1]]
    return result


def rotation(gram, sweeps=8, tol=1e-6):
    g = np.asarray(gram, dtype=np.float32).copy()
    n = len(g)
    r = np.eye(n, dtype=np.float32)
    for _ in range(sweeps):
        active = False
        for p in range(n):
            for q in range(p + 1, n):
                a, b, c = float(g[p, p]), float(g[q, q]), float(g[p, q])
                if abs(c) <= tol * np.sqrt(abs(a * b)) + 1e-30:
                    continue
                delta = .5 * (b - a)
                t = c / (delta + np.copysign(np.hypot(delta, c), delta))
                cs = 1 / np.sqrt(1 + t * t)
                sn = cs * t
                gp, gq = g[p].copy(), g[q].copy()
                g[p] = cs * gp - sn * gq
                g[q] = sn * gp + cs * gq
                g[:, p], g[:, q] = g[p], g[q]
                g[p, p], g[q, q] = a - t * c, b + t * c
                g[p, q] = g[q, p] = 0
                rp, rq = r[p].copy(), r[q].copy()
                r[p], r[q] = cs * rp - sn * rq, sn * rp + cs * rq
                active = True
        if not active:
            break
    return r


def blocked_svd(a, block=16, sweeps=20, local_sweeps=8, tol=1e-6):
    m, n = a.shape
    nb = max(2, ((m + block - 1) // block + 1) // 2 * 2)
    b = np.zeros((nb * block, n), dtype=np.float32)
    b[:m] = a
    for _ in range(sweeps):
        active = False
        for pairs in tournament(nb):
            for p, q in pairs:
                ids = np.r_[p * block:(p + 1) * block, q * block:(q + 1) * block]
                panel = b[ids]
                g = panel @ panel.T
                d = np.diag(g).astype(np.float64)
                threshold = tol * np.sqrt(np.abs(d[:, None] * d[None, :])) + 1e-30
                active |= bool(np.any(np.abs(g - np.diag(np.diag(g))) > threshold))
                b[ids] = rotation(g, local_sweeps, tol) @ panel
        if not active:
            break
    return b[:m]
