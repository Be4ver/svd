#!/usr/bin/env python3
"""Serial full-SVD timings on the existing exact-sweep 4096 input."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess

import numpy as np
from run_npu import log_path, validate, validate_execution

INPUT_SHA256 = '8485da69127c68fe91b3a84082c00b678f9a1842f83d4f6aff366c66df49867d'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe', type=Path, default=Path('build-core-local/jacobi_svd_main'))
    p.add_argument('--baseline', type=Path, default=Path('artifacts/core-local-start/jacobi_svd_baseline'))
    p.add_argument('--input', type=Path, default=Path('artifacts/exact-sweep-4096/input'))
    p.add_argument('--artifacts', type=Path, default=Path('artifacts/core-local/4096'))
    args = p.parse_args()
    digest = hashlib.sha256((args.input/'A_batch.bin').read_bytes()).hexdigest()
    assert digest == INPUT_SHA256, digest
    args.artifacts.mkdir(parents=True, exist_ok=True)
    result = {'input_sha256': digest, 'seed': 20261116,
              'historical_warm_median_ms': 966.515,
              'tolerance': float(np.finfo(np.float32).eps),
              'stopping_criterion': 'pairwise_relative_gram'}
    for name, exe, repeats in [('candidate', args.exe, 6), ('baseline', args.baseline, 4)]:
        out = args.artifacts/name
        cmd = [str(exe.resolve()), '0', '4096', '4096', '1', '0', '40',
               '--block', '64', '--repeat', str(repeats),
               '--input-dir', str(args.input.resolve()), '--output-dir', str(out.resolve())]
        run = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        log_path(args.artifacts, name+'.log').write_text(run.stdout+run.stderr)
        assert run.returncode == 0, run.stdout+run.stderr
        metrics = validate_execution(out)
        assert metrics['converged'] is True, metrics
        metrics['warm_median_ms'] = statistics.median(metrics['stream_ms'][1:])
        result[name] = metrics
        (args.artifacts/'results.json').write_text(json.dumps(result, indent=2))
        print(name, json.dumps(metrics), flush=True)
    result['bitwise_equal_to_saved_baseline'] = all(
        (args.artifacts/'candidate'/f).read_bytes() == (args.artifacts/'baseline'/f).read_bytes()
        for f in ('B_rot.bin', 'sigma.bin', 'Vt.bin'))
    a = np.fromfile(args.input/'A_batch.bin', np.float32).reshape(1,4096,4096)
    result['fp64_validation'] = validate(a, args.artifacts/'candidate')
    median = result['candidate']['warm_median_ms']
    result['historical_speedup'] = 966.515/median
    result['same_session_speedup'] = result['baseline']['warm_median_ms']/median
    result['pass'] = True
    (args.artifacts/'results.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
