import os,subprocess,re,json,sys
from pathlib import Path
import numpy as np
base=Path('/tmp/jacobi64-study-20261112-iOuF3K')
sys.path.insert(0,'/home/zhushengrui/project/paco-jacobi-svd/scripts')
from gen_jacobi_svd_data import build_matrix
results=[]
cases=[('normal',b,0) for b in [1,40,41,256,281,320,3200]]+[('cond1e3',80,1e3),('cond1e6',80,1e6)]
for kind,batch,cond in cases:
    rng=np.random.default_rng(2024)
    a=np.stack([build_matrix(rng,64,64,'normal',cond) for _ in range(batch)])
    ref=np.linalg.svd(a.astype(np.float64),compute_uv=False).astype(np.float32)
    a.tofile(base/'input/A_batch.bin'); ref.tofile(base/'input/sigma_ref.bin')
    (base/'input/shape.txt').write_text(f'64 64 {batch}\n')
    variants=['vector','vector8'] if kind=='normal' else ['baseline','vector','vector8']
    for variant in variants:
        build='build' if variant=='baseline' else 'build-'+variant
        p=subprocess.run([str(base/build/'jacobi_svd_main'),'0','64','64',str(batch),'40','30'],cwd=base/'build',capture_output=True,text=True)
        (base/f'vector_{variant}_{kind}_b{batch}.log').write_text(p.stdout+p.stderr)
        match=re.search(r'DEVICE_EVENT median_ms=([\d.]+) min_ms=([\d.]+) max_ms=([\d.]+)',p.stdout)
        row={'variant':variant,'case':kind,'batch':batch,'returncode':p.returncode}
        if match: row.update(zip(['median_ms','min_ms','max_ms'],map(float,match.groups())))
        if p.returncode==0:
            verify=subprocess.run([sys.executable,'/home/zhushengrui/project/paco-jacobi-svd/scripts/verify_jacobi_svd.py','--rows','64','--cols','64','--batch',str(batch)],cwd=base/'build',capture_output=True,text=True)
            (base/f'vector_verify_{variant}_{kind}_b{batch}.log').write_text(verify.stdout+verify.stderr)
            row['full_verify_pass']=verify.returncode==0
            sigma=np.fromfile(base/'output/sigma.bin',dtype=np.float32).reshape(batch,64)
            row['sigma_max_abs']=float(np.abs(sigma-ref).max())
            row['smallest_sigma_relative_max']=float(np.max(np.abs(sigma[:,-1]-ref[:,-1])/ref[:,-1]))
            for key,pattern in [('vt_orth',r'最差 Vt 行正交性\s*:\s*([\d.eE+-]+)'),('recon',r'最差重构相对误差\s*:\s*([\d.eE+-]+)')]:
                m=re.search(pattern,verify.stdout)
                if m: row[key]=float(m[1])
        results.append(row); print(json.dumps(row),flush=True)
        (base/'vector_results.json').write_text(json.dumps(results,indent=2))
