"""Run an isolated impossible-dependency build, then check a healthy launch."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from run_npu import log_path, validate_execution

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--fault-exe', type=Path, default=Path('build-fault/jacobi_svd_main'))
p.add_argument('--exe', type=Path, default=Path('build-core-local/jacobi_svd_main'))
p.add_argument('--input', type=Path,
               default=Path('artifacts/core-local/sweep-volatile/block16_nb2_batch1_aiv0/input'))
p.add_argument('--artifacts', type=Path, default=Path('artifacts/core-local/fault'))
args = p.parse_args()
common = ['0', '32', '49', '1', '0', '1', '--block', '16', '--fixed-sweeps',
          '--input-dir', str(args.input.resolve())]
start = time.monotonic()
run = subprocess.run([str(args.fault_exe.resolve()), *common,
                      '--output-dir', str((args.artifacts/'fault-output').resolve())],
                     capture_output=True, text=True, timeout=20)
elapsed = time.monotonic() - start
log_path(args.artifacts, 'injected.log').write_text(run.stdout+run.stderr)
assert run.returncode != 0 and 'core-local dependency watchdog expired' in run.stderr, run.stdout+run.stderr
assert 'ACL 507014' not in run.stderr, run.stderr
run = subprocess.run([str(args.exe.resolve()), *common, '--repeat', '3',
                      '--output-dir', str((args.artifacts/'recovery-output').resolve())],
                     capture_output=True, text=True, timeout=20)
log_path(args.artifacts, 'healthy.log').write_text(run.stdout+run.stderr)
assert run.returncode == 0, run.stdout+run.stderr
metrics = validate_execution(args.artifacts/'recovery-output')
assert len(metrics['stream_ms']) == 3 and metrics['sweeps'] == 1, metrics
result = {'pass': True, 'injected_fault_rejected': True, 'fault_wall_seconds': elapsed,
          'healthy_repeats': metrics['stream_ms']}
(args.artifacts/'results.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result), flush=True)
