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

def metrics(y,p):
    y=np.asarray(y,int);p=np.asarray(p,float);tn,fp,fn,tp=confusion_matrix(y,p>=.5,labels=[0,1]).ravel()
    return dict(n=len(y),positive=int(y.sum()),ACC=accuracy_score(y,p>=.5),MCC=matthews_corrcoef(y,p>=.5),Brier=brier_score_loss(y,p),log_loss=log_loss(y,np.clip(p,1e-15,1-1e-15),labels=[0,1]),AP=average_precision_score(y,p),AUROC=roc_auc_score(y,p),TN=int(tn),FP=int(fp),FN=int(fn),TP=int(tp))
