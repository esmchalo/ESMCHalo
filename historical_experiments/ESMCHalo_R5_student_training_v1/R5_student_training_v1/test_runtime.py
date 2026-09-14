"""Fast CPU tests in the same verified server interpreter, before GPU training."""
import unittest,tempfile,json
from pathlib import Path
import numpy as np,pandas as pd,torch
import torch.nn.functional as F
import training as t
class RuntimeTests(unittest.TestCase):
 def test_mix_endpoints(self):
  x=torch.arange(12,dtype=torch.float32).reshape(3,4);y=torch.tensor([0.,1.,0.]);u=torch.tensor([-4.,1.,3.]);perm=torch.tensor([2,0,1])
  for lam in [0.,1.]:
   a,b,q=t.mix_batch(x,y,u,lam,perm)
   torch.testing.assert_close(a,x if lam==1 else x[perm]);torch.testing.assert_close(b,y if lam==1 else y[perm]);torch.testing.assert_close(q,torch.sigmoid(u/2) if lam==1 else torch.sigmoid(u[perm]/2))
 def test_mix_targets(self):
  x=torch.randn(3,4);y=torch.tensor([0.,1.,0.]);u=torch.tensor([-4.,1.,3.]);perm=torch.tensor([2,0,1]);a,b,q=t.mix_batch(x,y,u,.3,perm)
  torch.testing.assert_close(q,.3*torch.sigmoid(u/2)+.7*torch.sigmoid(u[perm]/2));self.assertFalse(torch.allclose(q,torch.sigmoid((.3*u+.7*u[perm])/2)))
  self.assertTrue(((q>=0)&(q<=1)).all())
 def test_focal_gamma0(self):
  z=torch.tensor([-5.,0.,4.]);y=torch.tensor([0.,1.,0.]);torch.testing.assert_close(t.focal_hard(z,y,0),F.binary_cross_entropy_with_logits(z,y))
 def test_focal_value_gradient(self):
  z=torch.tensor([-20.,-2.,0.,3.,20.],requires_grad=True);y=torch.tensor([1.,0.,1.,1.,0.]);loss=t.focal_hard(z,y);ce=F.binary_cross_entropy_with_logits(z,y,reduction='none');torch.testing.assert_close(loss,((1-torch.exp(-ce))**2*ce).mean());loss.backward();self.assertTrue(torch.isfinite(z.grad).all())
 def test_mix_kd_gradient(self):
  for arm in ['Mixup','Focal']:
   model=t.Student(4,.2);x=torch.randn(8,4);y=torch.tensor([0.,1.]*4);u=torch.randn(8);loss=t.training_loss(model,x,y,u,arm,np.random.default_rng(1));loss.backward();self.assertTrue(torch.isfinite(loss));self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
 def test_h0_objective(self):
  model=torch.nn.Linear(4,1);x=torch.randn(8,4);y=torch.tensor([0.,1.]*4);u=torch.randn(8)
  class Flat(torch.nn.Module):
   def forward(self,x):return model(x).squeeze(-1)
  m=Flat();a=t.training_loss(m,x,y,u,'H0',np.random.default_rng(1));z=m(x);b=.5*F.binary_cross_entropy_with_logits(z,y)+2*F.binary_cross_entropy_with_logits(z/2,torch.sigmoid(u/2));torch.testing.assert_close(a,b)
 def test_tiny_fold_train_and_restore(self):
  protocol=json.loads((Path(__file__).parent/'PROTOCOL.json').read_text());protocol['student']['maximum_epochs']=2;protocol['student']['early_stopping_patience']=2;protocol['student']['batch_size']=8
  x=np.random.default_rng(2).normal(size=(40,4)).astype(np.float32);master=pd.DataFrame(dict(clean_id=['x'+str(i) for i in range(40)],label=[0,1]*20,fold=np.repeat(np.arange(5),8),homology_group=['g'+str(i) for i in range(40)],teacher_logit=np.linspace(-2,2,40)))
  for arm in ['Mixup','Focal']:
   with tempfile.TemporaryDirectory() as tmp:
    out=Path(tmp);r=t.run_fold(arm,0,x,master,None,protocol,torch.device('cpu'),out);self.assertEqual(len(r),8)
    ck=torch.load(out/'partial'/arm/'fold_0.pt',map_location='cpu',weights_only=False);self.assertEqual(ck['arm'],arm);model=t.Student(4,.2);model.load_state_dict(ck['model_state_dict']);z=((x[:8].astype(float)-ck['scaler_mean'])/ck['scaler_scale']).astype(np.float32);raw,_=t.predict(model,z,torch.device('cpu'),8);np.testing.assert_allclose(raw,r.student_raw_logit,atol=1e-6)
if __name__=='__main__':
 torch.set_num_threads(1);unittest.main(verbosity=2)
