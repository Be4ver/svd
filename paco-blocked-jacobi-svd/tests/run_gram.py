"""Exercise tiled AIC Gram dependencies through independently verified SVDs."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
from run_npu import log_path, validate, validate_execution


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe', type=Path, default=Path('build-fused/jacobi_svd_main'))
    p.add_argument('--artifacts', type=Path, default=Path('artifacts/gram-pipeline'))
    args = p.parse_args()
    rng = np.random.default_rng(20261119)
    results = []
    for block in (16, 32, 64):
        # K tails, multiple L1/L0 chunks, 4096 columns, and queue reuse on
        # one AIC owning many Gram jobs. Full device calls repeat three times.
        for cols, aiv in ((31, 0), (129, 2), (257, 0), (4096, 2)):
            a = rng.normal(size=(3, 2*block+1, cols)).astype(np.float32)
            root = args.artifacts / f'block{block}_k{cols}'
            inp, out = root/'input', root/'output'
            inp.mkdir(parents=True, exist_ok=True)
            a.tofile(inp/'A_batch.bin')
            batch, rows, _ = a.shape
            cmd = [str(args.exe.resolve()), '0', str(rows), str(cols), str(batch), str(aiv), '30',
                   '--block', str(block), '--repeat', '3', '--input-dir', str(inp.resolve()),
                   '--output-dir', str(out.resolve())]
            run = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            log_path(root, 'run.log').write_text(run.stdout+run.stderr)
            assert run.returncode == 0, run.stdout+run.stderr
            metrics = validate_execution(out)
            assert metrics['converged'], metrics
            assert all(f"run={i} sweeps={metrics['sweeps']} converged=1" in run.stdout
                       for i in range(3)), run.stdout
            result = {'block': block, 'cols': cols, **validate(a, out), **metrics}
            results.append(result)
            (args.artifacts/'results.json').write_text(json.dumps(results, indent=2))
            print(f'block{block} k{cols} PASS', flush=True)
    print(f'{len(results)}/{len(results)} passed', flush=True)


if __name__ == '__main__':
    main()
