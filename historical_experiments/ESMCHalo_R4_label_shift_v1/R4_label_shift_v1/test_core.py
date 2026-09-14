import unittest,numpy as np
from core import adapt,fit_platt,gate
from scipy.special import expit
class Tests(unittest.TestCase):
 def test_known_shift(self):
  p=np.r_[np.full(380,.8),np.full(620,.2)];q,s=adapt(p,.5)
  self.assertAlmostEqual(s['estimated_prior'],.3,places=6)
  self.assertAlmostEqual(q[0],.3*.8/(.3*.8+.7*.2),places=6)
 def test_no_shift(self):
  p=np.r_[np.full(500,.8),np.full(500,.2)];q,s=adapt(p,.5);np.testing.assert_allclose(q,p)
 def test_order(self):
  p=np.linspace(.01,.99,500);q,_=adapt(p,.6);self.assertTrue((np.diff(q)>0).all())
  rng=np.random.default_rng(3);ix=rng.permutation(len(p));z,_=adapt(p[ix],.6);np.testing.assert_allclose(z,q[ix],atol=1e-12)
 def test_invalid(self):
  for p in [[],[0,.5],[np.nan],[1.1]]:
   with self.assertRaises(ValueError):adapt(p,.5)
  with self.assertRaises(RuntimeError):adapt(np.r_[np.full(380,.8),np.full(620,.2)],.5,maxiter=1)
 def test_platt(self):
  x=np.repeat([-1.,0.,1.],100);y=np.r_[np.r_[np.ones(20),np.zeros(80)],np.r_[np.ones(50),np.zeros(50)],np.r_[np.ones(80),np.zeros(20)]]
  a,b=fit_platt(x,y);np.testing.assert_allclose(expit(a*np.array([-1,0,1])+b),[.2,.5,.8],atol=1e-6)
 def test_gate(self):
  r={'boundary':False,'before_ACC':.8,'after_ACC':.81,'before_MCC':.6,'after_MCC':.6,'before_Brier':.1,'after_Brier':.1,'before_log_loss':.4,'after_log_loss':.3}
  self.assertTrue(gate([r]*15)['passed']);r['after_ACC']=.79;self.assertFalse(gate([r]*15)['passed'])
if __name__=='__main__':unittest.main(verbosity=2)
