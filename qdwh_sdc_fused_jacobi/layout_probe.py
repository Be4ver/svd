#!/usr/bin/env python3
"""Replica of FusedLayout::Build (fused_layout.hpp).

Used to quantify the workspace and show what the Phase-2 batch-bound rewrite
saved. `legacy=True` reproduces the old `B*nPad^2` reservation.

The model is validated against the real binary: 2048x2048 reports 1641.9 MB and
this script gives 1642.0 MB.
"""
B = 8                     # kSdcBatchMax
W_VEC = 2048 * B          # kQdwhWVecTotal
W_GAP = 16384             # kQdwhWGap
T_STRIDE = 128 * 256      # kQdwhTStride
T_PANEL = 128 * 128       # kQdwhTPanel
T_SLACK = 8 * 1024        # kQdwhTSlack
MAXNODES = 16384
MAXLEAVES = 16384


def build(mPad, nPad, legacy=False):
    m, n = mPad, nPad
    qd = m + n
    np2 = nPad * nPad
    mn = m * n
    arenaFloats = 6 * np2
    qd2 = qd * qd

    batchF = nPad * (nPad + 127 * B)
    scale = (lambda k: B * k * np2) if legacy else (lambda k: k * batchF)
    x2 = B * np2 if legacy else 2 * batchF          # per-matrix x/a0/pt/xt slice

    hw = max(n * qd, 2 * x2 // 2 if legacy else 2 * batchF) if not legacy \
        else max(n * qd, B * 2 * np2)
    qb = max(qd2, B * 4 * np2) if legacy else max(qd2, 4 * batchF)

    wAic = max(128 * qd, 128 * 2 * nPad, 3 * n * 128)
    wTotal = W_VEC + W_GAP + B * wAic
    yNeed = max(2 * 64 * qd, 2 * 64 * 2 * nPad * B) + 4096

    leafCap = 2 * nPad * nPad
    sigmaCap = max(MAXLEAVES * 64, leafCap) if legacy else MAXLEAVES * 64

    mnOrPad = max(mn, np2)
    mnBatch = max(B * np2, mnOrPad) if legacy else max(mnOrPad, 2 * batchF)
    mnQdwh = (qd - n) * n
    ptNeed = max(mnQdwh, B * np2) if legacy else max(mnQdwh, 2 * batchF)

    items, o = [], 0

    def put(name, cnt):
        nonlocal o
        items.append((name, cnt))
        o += cnt

    put('a', mnOrPad)
    put('a0', mnBatch)
    put('xsdc', B * np2 if legacy else 2 * batchF)
    put('pt', ptNeed)
    put('xt', ptNeed)
    put('hwork', hw)
    put('qbuf', qb)
    put('tau', B * nPad)
    put('tmat', B * T_STRIDE + T_PANEL + T_SLACK)
    put('w', wTotal)
    put('y', yNeed)
    put('coef', 128)
    put('eye', 128 * 256)
    put('eyeN', n * n)
    put('uomega', 2 * np2)
    put('qt', np2)
    put('tmp', np2)
    put('hplus', np2)
    put('hminus', np2)
    put('om1', n * n)
    put('om2', n * n)
    put('arena', arenaFloats)
    put('leaf', leafCap)
    put('control', 32 * 40 + 4 * B + 16 + 3 * MAXNODES + 4 * MAXLEAVES
        + sigmaCap + 2 * MAXLEAVES + 2 + 32 + 128 + 8)
    return o, items


def mb(floats):
    return floats * 4 / 2**20


def main():
    print(f"{'shape':>13} {'qd':>6} {'before':>11} {'after':>11} {'saved':>8}  top after")
    for (mPad, nPad) in [(1024, 1024), (2048, 2048), (3072, 1024),
                         (4096, 4096), (8192, 8192)]:
        old, _ = build(mPad, nPad, legacy=True)
        new, items = build(mPad, nPad)
        top = sorted(items, key=lambda kv: -kv[1])[:4]
        desc = ", ".join(f"{k}={mb(v):.0f}MB" for k, v in top)
        print(f"{mPad:>6}x{nPad:<6} {mPad+nPad:>6} {mb(old):>9.0f}MB {mb(new):>9.0f}MB "
              f"{mb(old)/mb(new):>7.2f}x  {desc}")


if __name__ == '__main__':
    main()
