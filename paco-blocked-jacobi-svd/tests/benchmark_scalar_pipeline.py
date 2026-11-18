#!/usr/bin/env python3
"""Serial ABBA full-SVD benchmark against the saved core-local executable.

All runs use the existing 4096 input, epsilon threshold and one mixed launch.
One output directory is reused; each run records output hashes and metrics.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess

import numpy as np
from run_npu import validate, validate_execution
from run_core_local_4096 import INPUT_SHA256


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe', type=Path, default=Path('build-core-local/jacobi_svd_main'))
    p.add_argument('--baseline', type=Path,
                   default=Path('artifacts/scalar-pipeline-start/jacobi_svd_baseline'))
    p.add_argument('--label', required=True)
    p.add_argument('--repeat', type=int, default=4)
    p.add_argument('--abba', action='store_true')
    p.add_argument('--validate', action='store_true')
    args = p.parse_args()
    assert args.repeat >= 2
    inp = Path('artifacts/exact-sweep-4096/input').resolve()
    assert sha(inp/'A_batch.bin') == INPUT_SHA256
    root = Path('logs/scalar-pipeline') / args.label
    root.mkdir(parents=True, exist_ok=True)
    out = Path('artifacts/scalar-pipeline/4096/output').resolve()
    result = {'input_sha256': INPUT_SHA256, 'tolerance': float(np.finfo(np.float32).eps),
              'historical_ms': 966.515, 'runs': []}
    order = ('baseline', 'candidate', 'candidate', 'baseline') if args.abba else ('candidate',)
    for index, name in enumerate(order):
        exe = (args.exe if name == 'candidate' else args.baseline).resolve()
        cmd = [str(exe), '0', '4096', '4096', '1', '0', '40', '--block', '64',
               '--repeat', str(args.repeat), '--input-dir', str(inp), '--output-dir', str(out)]
        run = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        (root/f'{index}-{name}.log').write_text(run.stdout+run.stderr)
        assert run.returncode == 0, run.stdout+run.stderr
        metrics = validate_execution(out)
        assert metrics['converged'], metrics
        entry = {'name': name, 'command': cmd, 'executable_sha256': sha(exe), **metrics,
                 'hashes': {f: sha(out/f) for f in ('B_rot.bin', 'sigma.bin', 'Vt.bin')}}
        if args.validate and name == 'candidate' and 'fp64_validation' not in result:
            a = np.fromfile(inp/'A_batch.bin', np.float32).reshape(1,4096,4096)
            result['fp64_validation'] = validate(a, out)
        result['runs'].append(entry)
        (root/'results.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(entry), flush=True)
    for name in set(order):
        warm = [ms for r in result['runs'] if r['name'] == name for ms in r['stream_ms'][1:]]
        result[name+'_warm_median_ms'] = statistics.median(warm)
    candidate_hashes = [r['hashes'] for r in result['runs'] if r['name'] == 'candidate']
    result['candidate_repeated_bitwise_equal'] = all(h == candidate_hashes[0] for h in candidate_hashes)
    if 'baseline' in order:
        result['bitwise_equal_to_baseline'] = all(r['hashes'] == candidate_hashes[0] for r in result['runs'])
        result['time_reduction_percent'] = 100 * (1-result['candidate_warm_median_ms']/result['baseline_warm_median_ms'])
    (root/'results.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
