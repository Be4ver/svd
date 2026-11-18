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


def scheduled_rotation(gram, block, cross_only, tol):
    """One full or bipartite set of disjoint pivots, with bilateral updates."""
    g = np.asarray(gram, dtype=np.float64).copy()
    d = 2 * block
    r = np.eye(d)
    order = np.arange(d)
    permutation = (np.r_[np.arange(block), np.arange(block + 1, d), block]
                   if cross_only else
                   np.r_[0, block, np.arange(1, block - 1), np.arange(block + 1, d), block - 1])
    active = False
    for _ in range(block if cross_only else d - 1):
        p, q = order[:block], order[block:]
        aa, bb, cc = g[p, p], g[q, q], g[p, q]
        delta = (bb - aa) * .5
        denom = np.abs(delta) + np.hypot(delta, cc)
        t = np.divide(cc, denom, out=np.zeros_like(cc), where=denom != 0)
        t = np.where(delta >= 0, t, -t)
        mask = np.abs(cc) > tol * np.sqrt(np.abs(aa * bb)) + 1e-30
        active |= bool(np.any(mask))
        t = np.where(mask, t, 0)
        cs = 1 / np.sqrt(1 + t * t)
        sn = cs * t
        gp, gq = g[p].copy(), g[q].copy()
        g[p], g[q] = cs[:, None] * gp - sn[:, None] * gq, sn[:, None] * gp + cs[:, None] * gq
        gp, gq = g[:, p].copy(), g[:, q].copy()
        g[:, p], g[:, q] = gp * cs - gq * sn, gp * sn + gq * cs
        rp, rq = r[p].copy(), r[q].copy()
        r[p], r[q] = cs[:, None] * rp - sn[:, None] * rq, sn[:, None] * rp + cs[:, None] * rq
        order = order[permutation]
    return r, active


def blocked_svd(a, block=16, sweeps=20, local_sweeps=1, tol=1e-6):
    if local_sweeps != 1:
        raise ValueError("local_sweeps must be 1 to visit each pair once")
    m, n = a.shape
    nb = max(2, ((m + block - 1) // block + 1) // 2 * 2)
    b = np.zeros((nb * block, n), dtype=np.float32)
    b[:m] = a
    for _ in range(sweeps):
        active = False
        for outer_round, pairs in enumerate(tournament(nb)):
            for p, q in pairs:
                ids = np.r_[p * block:(p + 1) * block, q * block:(q + 1) * block]
                panel = b[ids]
                g = panel @ panel.T
                r, local_active = scheduled_rotation(g, block, outer_round != 0, tol)
                active |= local_active
                b[ids] = r @ panel
        if not active:
            break
    return b[:m]
