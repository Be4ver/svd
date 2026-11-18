"""Validate Prepare changes through real SVDs and optional baseline bit equality.

Record a JSON manifest before changing the kernel, then rerun with --reference.
Random panels exercise all map entries; two-AIV repeated calls exercise immutable
map reuse and the shared preprocessing/Gram scratch lifetime.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from run_npu import log_path, validate, validate_execution


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe', type=Path, default=Path('build-fused/jacobi_svd_main'))
    p.add_argument('--artifacts', type=Path, default=Path('artifacts/prepare-vector'))
    p.add_argument('--reference', type=Path)
    args = p.parse_args()
    reference = json.loads(args.reference.read_text()) if args.reference else None
    rng = np.random.default_rng(20261118)
    results = []
    for block in (16, 32, 64):
        d = 2 * block
        cases = [
            ('random', rng.normal(size=(3, 2*d, 2*d+17)).astype(np.float32), 0),
            ('identity', np.eye(d, dtype=np.float32)[None], 0),
            ('zero', np.zeros((1, d, d), np.float32), 0),
            ('reuse', rng.normal(size=(5, d+1, d-7)).astype(np.float32), 2),
        ]
        for kind, a, aiv in cases:
            name = f'{kind}_block{block}'
            root = args.artifacts / name
            inp, out = root / 'input', root / 'output'
            inp.mkdir(parents=True, exist_ok=True)
            a.tofile(inp / 'A_batch.bin')
            b, m, n = a.shape
            cmd = [str(args.exe.resolve()), '0', str(m), str(n), str(b), str(aiv), '30',
                   '--block', str(block), '--repeat', '3', '--input-dir', str(inp.resolve()),
                   '--output-dir', str(out.resolve())]
            run = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            log_path(root, 'run.log').write_text(run.stdout + run.stderr)
            assert run.returncode == 0, run.stdout + run.stderr
            metrics = validate_execution(out)
            assert metrics['converged'], metrics
            assert len(metrics['stream_ms']) == 3, metrics
            assert all(f"run={i} sweeps={metrics['sweeps']} converged=1" in run.stdout
                       for i in range(3)), run.stdout
            result = {'case': name, **validate(a, out), 'sweeps': metrics['sweeps'],
                      'hashes': {f: hashlib.sha256((out/f).read_bytes()).hexdigest()
                                 for f in ('B_rot.bin', 'sigma.bin', 'Vt.bin')}}
            if reference:
                old = next(r for r in reference if r['case'] == name)
                assert result['hashes'] == old['hashes'], (name, 'baseline output mismatch')
                assert result['sweeps'] == old['sweeps'], (name, 'baseline sweep mismatch')
            results.append(result)
            (args.artifacts / 'results.json').write_text(json.dumps(results, indent=2))
            print(name, 'PASS', 'baseline bit equality' if reference else 'FP64 validation', flush=True)
    print(f'{len(results)}/{len(results)} passed', flush=True)


if __name__ == '__main__':
    main()
