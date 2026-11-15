import numpy as np, json
from pathlib import Path
rng=np.random.default_rng(20261112)
n=64
ring=list(range(n)); rounds=[]
for r in range(n-1):
    pairs=[(ring[i],ring[n-1-i]) for i in range(n//2)]
    rounds.append((np.array([p[0] for p in pairs]),np.array([p[1] for p in pairs])))
    ring=[ring[0],ring[-1]]+ring[1:-1]
flat=[tuple(sorted((int(i),int(j)))) for a,b in rounds for i,j in zip(a,b)]
assert len(flat)==2016 and len(set(flat))==2016
assert all(len(set(np.r_[a,b]))==64 for a,b in rounds)
lex=[(np.array([i]),np.array([j])) for i in range(n) for j in range(i+1,n)]
normal=rng.normal(size=(16,n,n)).astype(np.float32)
def controlled(cond):
    out=[]
    for _ in range(8):
        u=np.linalg.qr(rng.normal(size=(n,n)))[0]
        v=np.linalg.qr(rng.normal(size=(n,n)))[0]
        out.append((u*np.geomspace(1,1/cond,n))@v.T)
    return np.array(out,dtype=np.float32)
cases={'normal':normal,'cond1e3':controlled(1e3),'cond1e6':controlled(1e6)}
results=[]
for kind,source in cases.items():
 for name,schedule in [('lex',lex),('circle63',rounds)]:
    x=source.copy(); batch=len(x)
    done=np.zeros(batch,dtype=bool); sweep_counts=np.full(batch,30)
    for sweep in range(30):
        changed=np.zeros(batch,dtype=bool)
        for ids,jds in schedule:
            u=x[:,ids,:]; v=x[:,jds,:]
            a=np.sum(u*u,axis=-1); b=np.sum(v*v,axis=-1); c=np.sum(u*v,axis=-1)
            act=(np.abs(c)>np.float32(1e-6)*np.sqrt(a*b)) & ~done[:,None]
            changed|=act.any(axis=1)
            safe=np.where(act,2*c,np.float32(1))
            tau=(b-a)/safe
            t=np.where(tau>=0,1,-1).astype(np.float32)/(np.abs(tau)+np.sqrt(1+tau*tau))
            t=np.where(act,t,np.float32(0))
            inv=np.sqrt(1+t*t); cs=1/inv; sn=cs*t
            un=cs[...,None]*u-sn[...,None]*v
            vn=inv[...,None]*v+t[...,None]*un
            x[:,ids,:]=un; x[:,jds,:]=vn
        newly=~changed&~done; sweep_counts[newly]=sweep+1; done|=newly
        if done.all():break
    xd=x.astype(np.float64); norms=np.linalg.norm(xd,axis=-1)
    sigma=np.sort(norms,axis=-1)[:,::-1]
    ref=np.linalg.svd(source.astype(np.float64),compute_uv=False)
    vt=xd/norms[...,None]; gram=vt@vt.transpose(0,2,1)
    row={'case':kind,'schedule':name,'count':batch,'sweep_min':int(sweep_counts.min()),'sweep_max':int(sweep_counts.max()),'all_converged':bool(done.all()),'sigma_max_abs':float(np.abs(sigma-ref).max()),'smallest_sigma_rel_max':float(np.max(np.abs(sigma[:,-1]-ref[:,-1])/ref[:,-1])),'vt_orth_max':float(np.abs(gram-np.eye(n)).max())}
    results.append(row); print(json.dumps(row),flush=True)
Path('/tmp/jacobi64-study-20261112-iOuF3K/schedule_results.json').write_text(json.dumps(results,indent=2))
