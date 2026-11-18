"""Check one real device sweep against an independent FP64 pair schedule.

Within-block pairs must occur once, as must cross-block pairs. Checking only
the converged singular values cannot detect an incorrect sweep definition.
"""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
from run_npu import log_path, validate_execution


def local_rounds(block, cross_only):
    d = 2 * block
    order = np.arange(d)
    if cross_only:
        permutation = np.r_[np.arange(block), np.arange(block + 1, d), block]
        count = block
    else:
        permutation = np.r_[0, block, np.arange(1, block - 1),
                            np.arange(block + 1, d), block - 1]
        count = d - 1
    result = []
    for _ in range(count):
        result.append((order[:block].copy(), order[block:].copy()))
        order = order[permutation]
    np.testing.assert_array_equal(order, np.arange(d))
    return result


def block_rounds(nb):
    # Direct tournament construction, independent of device PairBlock formula.
    order = list(range(nb))
    for _ in range(nb - 1):
        yield [(order[i], order[-1-i]) for i in range(nb // 2)]
        order = [order[0], order[-1], *order[1:-1]]


def one_sweep(a, block, tol):
    batch, rows, cols = a.shape
    nb = max(2, ((rows + block - 1) // block + 1) // 2 * 2)
    b = np.zeros((batch, nb * block, cols), np.float64)
    b[:, :rows] = a
    for outer_round, pairs in enumerate(block_rounds(nb)):
        schedule = local_rounds(block, outer_round != 0)
        for first, second in pairs:
            ids = np.r_[first * block:(first+1) * block,
                        second * block:(second+1) * block]
            for m in range(batch):
                x = b[m, ids].copy()
                g = x @ x.T
                r = np.eye(2 * block)
                for p, q in schedule:
                    aa, bb, cc = g[p, p], g[q, q], g[p, q]
                    delta = .5 * (bb-aa)
                    denominator = np.abs(delta) + np.hypot(delta, cc)
                    t = np.divide(cc, denominator, out=np.zeros_like(cc),
                                  where=denominator != 0)
                    t = np.where(delta >= 0, t, -t)
                    t = np.where(np.abs(cc) > tol*np.sqrt(np.abs(aa*bb)) + 1e-30, t, 0)
                    cs = 1 / np.sqrt(1 + t*t)
                    sn = cs*t
                    gp, gq = g[p].copy(), g[q].copy()
                    g[p], g[q] = cs[:, None]*gp-sn[:, None]*gq, sn[:, None]*gp+cs[:, None]*gq
                    gp, gq = g[:, p].copy(), g[:, q].copy()
                    g[:, p], g[:, q] = gp*cs-gq*sn, gp*sn+gq*cs
                    rp, rq = r[p].copy(), r[q].copy()
                    r[p], r[q] = cs[:, None]*rp-sn[:, None]*rq, sn[:, None]*rp+cs[:, None]*rq
                b[m, ids] = r @ x
    return b[:, :rows]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe', type=Path, default=Path('build-fused/jacobi_svd_main'))
    p.add_argument('--artifacts', type=Path, default=Path('artifacts/sweep-once'))
    args = p.parse_args()
    rng = np.random.default_rng(20261120)
    results = []
    for block in (16, 32, 64):
        for nb, odd, batch, aiv in ((2, False, 1, 0), (4, False, 2, 0), (6, True, 1, 2),
                                    (8, True, 5, 6), (10, False, 3, 8)):
            rows, cols = nb*block-int(odd), nb*block+17
            # Separated row norms avoid unstable pivot signs near repeated
            # diagonal values: this test checks pair visitation, not those
            # legitimate FP32/FP64 differences in the Jacobi trajectory.
            a = rng.normal(scale=.001, size=(batch, rows, cols))
            a[:, np.arange(rows), np.arange(rows)] += 1
            # More rows need wider norm separation to keep this trajectory
            # comparison away from almost equal diagonal Gram entries.
            lo, hi = (.05, 20) if nb > 6 else (.25, 4)
            a *= np.geomspace(lo, hi, rows)[None, :, None]
            a = a.astype(np.float32)
            root = args.artifacts / f'block{block}_nb{nb}_batch{batch}_aiv{aiv}'
            inp, out = root/'input', root/'output'
            inp.mkdir(parents=True, exist_ok=True)
            a.tofile(inp/'A_batch.bin')
            cmd = [str(args.exe.resolve()), '0', str(rows), str(cols), str(batch), str(aiv), '1',
                   '--block', str(block), '--repeat', '2', '--fixed-sweeps', '--input-dir', str(inp.resolve()),
                   '--output-dir', str(out.resolve())]
            run = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            log_path(root, 'run.log').write_text(run.stdout+run.stderr)
            assert run.returncode == 0, run.stdout+run.stderr
            metrics = validate_execution(out)
            assert metrics['sweeps'] == 1, metrics
            expected = one_sweep(a, block, float(np.finfo(np.float32).eps))
            actual = np.fromfile(out/'B_rot.bin', np.float32).reshape(a.shape)
            error = float(np.linalg.norm(actual-expected) / np.linalg.norm(expected))
            record = {'block': block, 'nb': nb, 'rows': rows, 'batch': batch,
                      'relative_B_error': error, 'pass': error < 2e-5}
            results.append(record)
            (args.artifacts/'results.json').write_text(json.dumps(results, indent=2))
            print(record, flush=True)
    assert all(r['pass'] for r in results), 'one-sweep pair schedule differs from FP64 reference'
    print(f'{len(results)}/{len(results)} passed')


if __name__ == '__main__':
    main()
