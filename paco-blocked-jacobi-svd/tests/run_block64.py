"""NPU regression for paired 64-row blocks, checked against FP64 SVD."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
from run_npu import log_path, validate, validate_execution

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe', type=Path, default=Path('build-fused/jacobi_svd_main'))
    p.add_argument('--artifacts', type=Path, default=Path('artifacts/block64-numerical'))
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    rng = np.random.default_rng(20261117)
    cases = [('random128', rng.normal(size=(2,128,128)).astype(np.float32))]
    if not args.smoke:
        cases += [('overlap80', rng.normal(size=(80,128,128)).astype(np.float32)),
                  ('random129x97', rng.normal(size=(3,129,97)).astype(np.float32)),
                  ('random256', rng.normal(size=(1,256,256)).astype(np.float32)),
                  ('zero', np.zeros((1,128,128),np.float32)),
                  ('identity', np.eye(128,dtype=np.float32)[None]),
                  ('repeated', (3*np.eye(128,dtype=np.float32))[None])]
        a = rng.normal(size=(2,128,160)).astype(np.float32)
        a[:,64:] = a[:,:1]
        cases.append(('rank',a))
        q,_ = np.linalg.qr(rng.normal(size=(128,128)))
        v,_ = np.linalg.qr(rng.normal(size=(160,128)))
        cases.append(('condition1e4', ((q*np.geomspace(1,1e-4,128))@v.T).astype(np.float32)[None]))
    results=[]
    for name,a in cases:
        root=args.artifacts/name
        inp,out=root/'input',root/'output'
        inp.mkdir(parents=True,exist_ok=True)
        a.tofile(inp/'A_batch.bin')
        b,m,n=a.shape
        cmd=[str(args.exe.resolve()),'0',str(m),str(n),str(b),'0','30','--block','64',
             '--input-dir',str(inp.resolve()),'--output-dir',str(out.resolve())]
        run=subprocess.run(cmd,capture_output=True,text=True,timeout=180)
        log_path(root,'run.log').write_text(run.stdout+run.stderr)
        assert run.returncode==0, run.stdout+run.stderr
        result={'case':name,**validate(a,out),**validate_execution(out)}
        assert result['converged'], result
        results.append(result)
        print(name,json.dumps(result),flush=True)
        (args.artifacts/'results.json').write_text(json.dumps(results,indent=2))

if __name__=='__main__':
    main()
