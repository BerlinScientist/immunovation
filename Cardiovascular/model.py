"""Li et al. (2014), Eqs. 2, 4, 5: independent Python implementation.
Original physiological parameters: SI Tables S2 and S3 (subject 2).
See README for reconstruction choices and inconsistencies in the SI LQR notation.
Run: python3 doyle_model.py
"""
from pathlib import Path
import json
import numpy as np
from scipy.linalg import solve_continuous_are
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
PAR = dict(cas=0.01016, cvs=0.6500, cap=0.0361, cvp=0.1408,
           cl=0.025, cr=0.045, Rp=0.5, Rs0=6.5, A=18.0,
           Vtot=5.0580, Oa=0.2, rho=0.012, M0=0.36, VT=6.0)

def quantities(x, H, W, p=PAR):
    """Units: minutes, litres, mmHg, beats/min, watts; oxygen L O2/L blood."""
    Pas, Pvs, Pap, Ov = x
    Pvp = (p['Vtot']-p['cas']*Pas-p['cvs']*Pvs-p['cap']*Pap)/p['cvp']
    Rs = p['A']*Ov+p['Rs0']
    Ql, Qr = p['cl']*H*Pvp, p['cr']*H*Pvs
    Fs, Fp = (Pas-Pvs)/Rs, (Pap-Pvp)/p['Rp']
    M = p['rho']*W+p['M0']
    return Pvp, Rs, Ql, Qr, Fs, Fp, M

def rhs(x, H, W, p=PAR):
    Pvp, Rs, Ql, Qr, Fs, Fp, M = quantities(x, H, W, p)
    return np.array([(Ql-Fs)/p['cas'], (Fs-Qr)/p['cvs'],
                     (Qr-Fp)/p['cap'], (-M+Fs*(p['Oa']-x[3]))/p['VT']])

def equilibrium(H, W, p=PAR):
    """Solve flow balance and oxygen balance algebraically (Eqs. 2-3).
    This avoids root-solver dependence and uses the conserved total volume.
    """
    M = p['rho']*W+p['M0']
    coeff = (p['cas']*(1/(p['cr']*H)+p['Rs0']+p['A']*p['Oa'])
             +p['cvs']/(p['cr']*H)+p['cap']*(1/(p['cl']*H)+p['Rp'])
             +p['cvp']/(p['cl']*H))
    F = (p['Vtot']+p['cas']*p['A']*M)/coeff
    Ov = p['Oa']-M/F
    Pvs, Pvp = F/(p['cr']*H), F/(p['cl']*H)
    Pas, Pap = Pvs+F*(p['Rs0']+p['A']*Ov), Pvp+F*p['Rp']
    return np.array([Pas, Pvs, Pap, Ov])

def controller(H0, W0, weights):
    x0 = equilibrium(H0, W0)
    # Central finite differences of the ORIGINAL nonlinear equations.
    steps = np.array([1e-4, 1e-5, 1e-5, 1e-7])
    A = np.column_stack([(rhs(x0+np.eye(4)[i]*h,H0,W0)
                         -rhs(x0-np.eye(4)[i]*h,H0,W0))/(2*h)
                        for i,h in enumerate(steps)])
    h = 1e-3
    B = ((rhs(x0,H0+h,W0)-rhs(x0,H0-h,W0))/(2*h))[:,None]
    qp, qo, qh = weights
    # Main text Eq. 5: SUM of squared errors, not square of their sum.
    Q, R = np.diag([qp**2,0,0,qo**2]), np.array([[qh**2]])
    P = solve_continuous_are(A,B,Q,R)
    K = np.linalg.solve(R,B.T@P)
    assert np.max(np.real(np.linalg.eigvals(A-B@K))) < 0
    residual = A.T@P+P@A-P@B@np.linalg.solve(R,B.T@P)+Q
    assert np.linalg.norm(residual)/max(1,np.linalg.norm(Q)) < 1e-6
    return x0, K

def simulate(H0, W0, weights, feedback=True):
    x0, K = controller(H0, W0, weights)
    def heart(x):
        return H0-float((K@(x-x0)).item()) if feedback else H0
    # Demonstration input; original measured input/data are not available.
    # Piecewise integration prevents stepping over input discontinuities.
    segments = [(0,40,W0),(40,100,W0+50),(100,160,W0),
                (160,220,W0+50),(220,280,W0),(280,340,W0+50),(340,400,W0)]
    state=x0.copy(); ts=[]; xs=[]; ws=[]
    for j,(a,b,w) in enumerate(segments):
        grid=np.linspace(a/60,b/60,int(b-a)+1)
        sol=solve_ivp(lambda t,x:rhs(x,heart(x),w),(a/60,b/60),state,
                      t_eval=grid,method='Radau',rtol=1e-8,atol=1e-10,max_step=1/60)
        if not sol.success: raise RuntimeError(sol.message)
        state=sol.y[:,-1]
        keep=slice(None) if j==0 else slice(1,None)
        ts.extend(sol.t[keep]*60);xs.extend(sol.y.T[keep]);ws.extend([w]*len(sol.t[keep]))
    t,x,w=np.array(ts),np.array(xs),np.array(ws)
    H=np.array([heart(s) for s in x])
    # Do not silently clip: failure flags excursions outside the model's physical domain.
    assert np.all((x[:,3]>0)&(x[:,3]<PAR['Oa']))
    assert np.all(H>0)
    for s,h,ww in zip(x,H,w):
        assert quantities(s,h,ww)[0]>0 and np.all(s[:3]>0)
    return t,x,w,H,K

def main():
    fig,axs=plt.subplots(4,2,figsize=(11,9),sharex=True)
    report={}
    for col,(H0,W0,q) in enumerate([(60,0,(30,100000,1)),(120,100,(80,100000,35))]):
        x0=equilibrium(H0,W0)
        assert np.max(np.abs(rhs(x0,H0,W0)))<1e-9
        for feedback,color,label in [(True,'#137f89','Optimal controller'),(False,'#b78445','Fixed heart rate')]:
            t,x,w,H,K=simulate(H0,W0,q,feedback)
            for ax,values in zip(axs[:,col],[w,H,x[:,0],1000*x[:,3]]):
                ax.plot(t,values,color=color,label=label)
            np.savetxt(OUT/f'simulation_{W0}W_{"feedback" if feedback else "fixed"}.csv',
                       np.c_[t,w,H,x],delimiter=',',comments='',
                       header='time_s,workload_W,HR_bpm,Pas_mmHg,Pvs_mmHg,Pap_mmHg,Ov_L_O2_per_L_blood')
        axs[0,col].set_title(f'Workload {W0}–{W0+50} W')
        axs[-1,col].set_xlabel('Time (seconds)')
        report[str(W0)]={'reference_HR_bpm':H0,'reference_state':x0.tolist(),
                        'weights':q,'gain_K':K.tolist()}
    labels=['Workload (W)','Heart rate (1/min)','Mean pressure (mmHg)','Venous O₂ content\n(ml O₂/L blood)']
    for row,label in enumerate(labels): axs[row,0].set_ylabel(label)
    for ax in axs.flat: ax.grid(alpha=.2)
    axs[1,1].legend(fontsize=9)
    fig.suptitle('Li et al. / Doyle (2014): Equations 2, 4, 5 in Python\nPublished parameters; chosen initial values and demonstration workload',fontsize=12)
    fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(OUT/'doyle_simulation.png',dpi=170)
    (OUT/'model_details.json').write_text(json.dumps({'parameters':PAR,'runs':report},indent=2))
    print('Equilibria, LQR stability, Riccati equation and physical domain checked.')
    for w,r in report.items(): print(f'{w} W: Initial pressure {r["reference_state"][0]:.2f} mmHg')

if __name__=='__main__': main()
