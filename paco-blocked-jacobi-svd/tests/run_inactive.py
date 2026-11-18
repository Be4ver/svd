"""Exercise identity panels, threshold boundaries and sparse active rounds."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from run_npu import log_path, validate_execution
from run_sweep import one_sweep


def cases(eps):
    for name, factor, rows, cols, batch, aiv in (
        ('zero_cross', 0., 256, 256, 1, 0),
        ('below', .5, 256, 256, 1, 0),
        ('threshold', 1., 256, 256, 1, 0),
        ('above', 2., 256, 256, 1, 0),
        ('padded_reuse', 2., 255, 273, 5, 2),
    ):
        a = np.zeros((batch, rows, cols), np.float32)
        diagonal = np.geomspace(.25, 4., rows).astype(np.float32)
        a[:, np.arange(rows), np.arange(rows)] = diagonal
        # Blocks 0 and 1 meet after the initial full-pair outer round.
        for m in range(batch):
            a[m, m, 64+m] = factor * eps * diagonal[m]
        yield name, a, aiv
    for rows, cols, batch, aiv in ((256, 273, 1, 0), (255, 273, 5, 2),
                                   (383, 401, 3, 6)):
        a = np.zeros((batch, rows, cols), np.float32)
        diagonal = np.geomspace(.25, 4., rows).astype(np.float32)
        a[:, np.arange(rows), np.arange(rows)] = diagonal
        # Disjoint pairs become active in separated local rounds. Most rounds
        # are identities, so omitting their permutations loses later pivots.
        # Include a below-threshold pair alongside the three active pairs.
        for m in range(batch):
            for p, q, factor in ((0, 1, .015), (3, 13, -.007),
                                 (17, 28, 2*eps), (47, 57, .25*eps)):
                a[m, p+m, 64+(q+m) % 64] = factor * diagonal[p+m]
        yield f'sparse_rounds_r{rows}_b{batch}_aiv{aiv}', a, aiv


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe', type=Path, default=Path('build-core-local/jacobi_svd_main'))
    p.add_argument('--baseline', type=Path,
                   default=Path('artifacts/scalar-pipeline-start/jacobi_svd_baseline'))
    p.add_argument('--artifacts', type=Path, default=Path('artifacts/scalar-pipeline/inactive'))
    args = p.parse_args()
    eps = float(np.finfo(np.float32).eps)
    results = []
    for name, a, aiv in cases(eps):
        batch, rows, cols = a.shape
        root = args.artifacts / name
        inp = root / 'input'
        inp.mkdir(parents=True, exist_ok=True)
        a.tofile(inp/'A_batch.bin')
        hashes, metrics = {}, {}
        for mode, exe in (('baseline', args.baseline), ('candidate', args.exe)):
            out = root / mode
            cmd = [str(exe.resolve()), '0', str(rows), str(cols), str(batch), str(aiv), '1',
                   '--block', '64', '--fixed-sweeps', '--repeat', '3',
                   '--input-dir', str(inp.resolve()), '--output-dir', str(out.resolve())]
            run = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
            log_path(root, mode+'.log').write_text(run.stdout+run.stderr)
            assert run.returncode == 0, run.stdout+run.stderr
            metrics[mode] = validate_execution(out)
            assert metrics[mode]['sweeps'] == 1 and len(metrics[mode]['stream_ms']) == 3
            hashes[mode] = {f: hashlib.sha256((out/f).read_bytes()).hexdigest()
                            for f in ('B_rot.bin', 'sigma.bin', 'Vt.bin')}
        expected = one_sweep(a, 64, eps)
        actual = np.fromfile(root/'candidate/B_rot.bin', np.float32).reshape(a.shape)
        error = float(np.linalg.norm(actual-expected) / np.linalg.norm(expected))
        assert error < 2e-5, (name, error)
        assert hashes['baseline'] == hashes['candidate'], (name, 'baseline mismatch')
        assert metrics['baseline']['converged'] == metrics['candidate']['converged']
        record = {'case': name, 'pass': True, 'relative_B_error': error,
                  'bitwise_equal': True, 'execution': metrics, 'hashes': hashes}
        results.append(record)
        (args.artifacts/'results.json').write_text(json.dumps(results, indent=2))
        print(name, 'PASS', error, flush=True)
    print(f'{len(results)}/{len(results)} passed', flush=True)


if __name__ == '__main__':
    main()
