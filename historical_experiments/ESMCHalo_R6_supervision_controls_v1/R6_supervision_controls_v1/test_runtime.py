"""CPU correctness tests before real R6 training."""
import unittest,tempfile,json
from pathlib import Path
import numpy as np,pandas as pd,torch
import torch.nn.functional as F
import training as t
class Tests(unittest.TestCase):
 def setUp(self):
  torch.manual_seed(13);self.model=t.Student(4,.2);self.model.eval();self.x=torch.randn(8,4);self.y=torch.tensor([0.,1.]*4);self.u=torch.randn(8)
 def test_b0_objective(self):
  a=t.training_loss(self.model,self.x,self.y,self.u,'B0',np.random.default_rng(1));b=F.binary_cross_entropy_with_logits(self.model(self.x),self.y);torch.testing.assert_close(a,b)
 def test_b0_teacher_unused(self):self.check_teacher('B0')
 def test_m0_teacher_unused(self):self.check_teacher('M0')
 def check_teacher(self,arm):
  grads=[];losses=[]
  for u in [self.u,self.u*100+77]:
   self.model.zero_grad(set_to_none=True);l=t.training_loss(self.model,self.x,self.y,u,arm,np.random.default_rng(1));l.backward();losses.append(l.detach());grads.append([p.grad.clone() for p in self.model.parameters()])
  torch.testing.assert_close(*losses)
  for a,b in zip(*grads):torch.testing.assert_close(a,b)
 def test_m0_targets(self):
  class Rng:
   def beta(self,a,b):return .3
   def permutation(self,n):return np.arange(n)[::-1].copy()
  p=torch.arange(7,-1,-1);loss=t.training_loss(self.model,self.x,self.y,self.u,'M0',Rng());expected=F.binary_cross_entropy_with_logits(self.model(.3*self.x+.7*self.x[p]),.3*self.y+.7*self.y[p]);torch.testing.assert_close(loss,expected)
 def test_gradients(self):
  for arm in ['B0','M0']:
   self.model.zero_grad(set_to_none=True);self.model.train();l=t.training_loss(self.model,self.x,self.y,self.u,arm,np.random.default_rng(1));l.backward();self.assertTrue(torch.isfinite(l));self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in self.model.parameters()))
 def test_unknown_arm(self):
  with self.assertRaises(ValueError):t.training_loss(self.model,self.x,self.y,self.u,'H0',np.random.default_rng(1))
 def test_tiny_fold_restore(self):
  protocol=json.loads((Path(__file__).parent/'PROTOCOL.json').read_text());protocol['student']['maximum_epochs']=2;protocol['student']['early_stopping_patience']=2;protocol['student']['batch_size']=8
  x=np.random.default_rng(2).normal(size=(40,4)).astype(np.float32);master=pd.DataFrame(dict(clean_id=['x'+str(i) for i in range(40)],label=[0,1]*20,fold=np.repeat(np.arange(5),8),homology_group=['g'+str(i) for i in range(40)],teacher_logit=np.linspace(-2,2,40)))
  for arm in ['B0','M0']:
   with tempfile.TemporaryDirectory() as tmp:
    out=Path(tmp);r=t.run_fold(arm,0,x,master,None,protocol,torch.device('cpu'),out);self.assertEqual(len(r),8)
    ck=torch.load(out/'partial'/arm/'fold_0.pt',map_location='cpu',weights_only=False);self.assertFalse(ck['teacher_supervision_used']);self.assertEqual(ck['arm'],arm);model=t.Student(4,.2);model.load_state_dict(ck['model_state_dict']);z=((x[:8].astype(float)-ck['scaler_mean'])/ck['scaler_scale']).astype(np.float32);raw,_=t.predict(model,z,torch.device('cpu'),8);np.testing.assert_allclose(raw,r.student_raw_logit,atol=1e-6)
if __name__=='__main__':
 torch.set_num_threads(1);unittest.main(verbosity=2)
