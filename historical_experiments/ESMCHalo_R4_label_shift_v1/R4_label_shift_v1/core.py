import numpy as np
from scipy.special import expit,logit
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score,matthews_corrcoef,brier_score_loss,log_loss,roc_auc_score,average_precision_score,confusion_matrix

def fit_platt(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float)
    def fun(t):
        z=t[0]*x+t[1];return np.mean(np.logaddexp(0,z)-y*z)
    def jac(t):
        e=expit(t[0]*x+t[1])-y;return np.array([np.mean(e*x),np.mean(e)])
    r=minimize(fun,[1.,0.],jac=jac,method='L-BFGS-B',options={'maxiter':2000,'ftol':1e-15,'gtol':1e-10})
    if not r.success or not np.isfinite(r.x).all() or r.x[0]<=0:raise RuntimeError('Platt fit failed or nonpositive slope: '+str(r.message))
    return r.x

def adapt(p,source_prior,tol=1e-8,maxiter=1000):
    p=np.asarray(p,float)
    if p.ndim!=1 or len(p)==0 or not np.isfinite(p).all() or not ((p>0)&(p<1)).all():raise ValueError('Probabilities must be finite and strictly inside (0,1)')
    if not 1e-4<=source_prior<=1-1e-4:raise ValueError('Invalid source prior')
    x=logit(p)-logit(source_prior);prior=source_prior
    for it in range(1,maxiter+1):
        q=expit(x+logit(prior));new=float(np.clip(q.mean(),1e-4,1-1e-4))
        if abs(new-prior)<tol:
            q=expit(x+logit(new))
            if not ((q>0)&(q<1)).all():raise RuntimeError('Numerical probability saturation')
            order=np.argsort(p);dp=np.diff(p[order]);dq=np.diff(q[order])
            if (dq[dp>0]<=0).any() or (dq[dp==0]!=0).any():raise RuntimeError('Numerical ranking/tie change')
            return q,dict(source_prior=float(source_prior),estimated_prior=new,iterations=it,boundary=bool(new<=1e-4 or new>=1-1e-4))
        prior=new
    raise RuntimeError('EM did not converge in '+str(maxiter)+' iterations')

def metrics(y,p):
    y=np.asarray(y,int);p=np.asarray(p,float);tn,fp,fn,tp=confusion_matrix(y,p>=.5,labels=[0,1]).ravel()
    return dict(n=len(y),positive=int(y.sum()),ACC=accuracy_score(y,p>=.5),MCC=matthews_corrcoef(y,p>=.5),Brier=brier_score_loss(y,p),log_loss=log_loss(y,np.clip(p,1e-15,1-1e-15),labels=[0,1]),AP=average_precision_score(y,p),AUROC=roc_auc_score(y,p),TN=int(tn),FP=int(fp),FN=int(fn),TP=int(tp))

def gate(rows):
    keys=['ACC','MCC','Brier','log_loss'];d={k:float(np.mean([r['after_'+k]-r['before_'+k] for r in rows])) for k in keys}
    return dict(delta=d,passed=bool(d['ACC']>0 and d['log_loss']<0 and d['MCC']>=0 and d['Brier']<=0 and not any(r['boundary'] for r in rows)))
