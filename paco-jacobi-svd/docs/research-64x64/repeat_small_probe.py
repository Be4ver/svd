from pathlib import Path
import subprocess,json,re
import numpy as np
base=Path('/tmp/jacobi64-study-20261112-iOuF3K'); results=[]
for seed in [2025,2026,2027]:
    rng=np.random.default_rng(seed); a=rng.standard_normal((40,64,64)).astype(np.float32)
    ref=np.linalg.svd(a.astype(np.float64),compute_uv=False).astype(np.float32)
    a.tofile(base/'input/A_batch.bin'); ref.tofile(base/'input/sigma_ref.bin'); (base/'input/shape.txt').write_text('64 64 40'+chr(10))
    for variant in ['baseline','tournament']:
        build='build' if variant=='baseline' else 'build-tournament'
        p=subprocess.run([str(base/build/'jacobi_svd_main'),'0','64','64','40','40','30'],cwd=base/'build',capture_output=True,text=True,check=True)
        (base/f'repeat_{variant}_s{seed}.log').write_text(p.stdout+p.stderr)
        row={'seed':seed,'variant':variant,'batch':40}
        for k in ['median_ms','min_ms','max_ms']:row[k]=float(re.search(k+'=([0-9.]+)',p.stdout)[1])
        q=subprocess.run(['python3','/home/zhushengrui/project/paco-jacobi-svd/scripts/verify_jacobi_svd.py','--rows','64','--cols','64','--batch','40'],cwd=base/'build',capture_output=True,text=True)
        (base/f'repeat_verify_{variant}_s{seed}.log').write_text(q.stdout+q.stderr)
        row['full_verify_pass']=q.returncode==0
        results.append(row); print(json.dumps(row),flush=True)
(base/'repeat_results.json').write_text(json.dumps(results,indent=2))
