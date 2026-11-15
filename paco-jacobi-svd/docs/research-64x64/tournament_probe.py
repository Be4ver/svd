import os,subprocess,re,json,sys
from pathlib import Path
import numpy as np
base=Path('/tmp/jacobi64-study-20261112-iOuF3K')
sys.path.insert(0,'/home/zhushengrui/project/paco-jacobi-svd/scripts')
from gen_jacobi_svd_data import build_matrix
results=[]
cases=[('normal',b,0) for b in [1,8,40]]+[('cond1e3',40,1e3),('cond1e6',40,1e6),('zero',8,0),('identity',8,0),('rank32',8,0),('repeated',8,0)]
for kind,batch,cond in cases:
    rng=np.random.default_rng(2024)
    a=np.stack([build_matrix(rng,64,64,'normal',cond) for _ in range(batch)])
    if kind=='zero': a[:]=0
    if kind=='identity': a[:]=np.eye(64,dtype=np.float32)
    if kind=='rank32': a[:,:,32:]=0
    if kind=='repeated':
        for k in range(batch):
            u=np.linalg.qr(rng.normal(size=(64,64)))[0]; v=np.linalg.qr(rng.normal(size=(64,64)))[0]
            a[k]=(u*np.repeat([1.0,0.1,0.01,0.001],16))@v.T
    ref=np.linalg.svd(a.astype(np.float64),compute_uv=False).astype(np.float32)
    a.tofile(base/'input/A_batch.bin'); ref.tofile(base/'input/sigma_ref.bin')
    (base/'input/shape.txt').write_text(f'64 64 {batch}'+chr(10))
    for variant in ['baseline','tournament']:
        build='build' if variant=='baseline' else 'build-'+variant
        p=subprocess.run([str(base/build/'jacobi_svd_main'),'0','64','64',str(batch),'40','30'],cwd=base/'build',capture_output=True,text=True)
        (base/f'tournament_{variant}_{kind}_b{batch}.log').write_text(p.stdout+p.stderr)
        match=re.search(r'DEVICE_EVENT median_ms=([0-9.]+) min_ms=([0-9.]+) max_ms=([0-9.]+)',p.stdout)
        row={'variant':variant,'case':kind,'batch':batch,'returncode':p.returncode}
        if match: row.update(zip(['median_ms','min_ms','max_ms'],map(float,match.groups())))
        if p.returncode==0:
            verify=subprocess.run([sys.executable,'/home/zhushengrui/project/paco-jacobi-svd/scripts/verify_jacobi_svd.py','--rows','64','--cols','64','--batch',str(batch)],cwd=base/'build',capture_output=True,text=True)
            (base/f'tournament_verify_{variant}_{kind}_b{batch}.log').write_text(verify.stdout+verify.stderr)
            row['full_verify_pass']=verify.returncode==0
            sigma=np.fromfile(base/'output/sigma.bin',dtype=np.float32).reshape(batch,64)
            row['sigma_max_abs']=float(np.abs(sigma-ref).max())
            if kind not in ['zero','rank32']:row['smallest_sigma_relative_max']=float(np.max(np.abs(sigma[:,-1]-ref[:,-1])/ref[:,-1]))
            for key,pattern in [('vt_orth',r'最差 Vt 行正交性 *: *([0-9.eE+-]+)'),('recon',r'最差重构相对误差 *: *([0-9.eE+-]+)')]:
                m=re.search(pattern,verify.stdout)
                if m: row[key]=float(m[1])
        results.append(row); print(json.dumps(row),flush=True)
        (base/'tournament_results.json').write_text(json.dumps(results,indent=2))
