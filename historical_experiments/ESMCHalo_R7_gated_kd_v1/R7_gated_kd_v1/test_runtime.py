"""CPU tests; must pass before G0 GPU training."""
import unittest,tempfile,json
from pathlib import Path
from unittest.mock import patch
import numpy as np,pandas as pd,torch
import torch.nn.functional as F
import training as t

class Tests(unittest.TestCase):
 def setUp(self):
  torch.manual_seed(13);self.model=t.Student(4,.2);self.model.eval();self.x=torch.randn(8,4);self.y=torch.tensor([0.,1.]*4)
 def loss(self,u):return t.training_loss(self.model,self.x,self.y,u,'G0',None)
 def compare_grads(self,a,b):
  params=list(self.model.parameters());ga=torch.autograd.grad(a,params);gb=torch.autograd.grad(b,params)
  torch.testing.assert_close(a,b)
  for x,y in zip(ga,gb):torch.testing.assert_close(x,y)
 def test_all_agree_equals_h0(self):
  u=(2*self.y-1)*3;z=self.model(self.x)
  expected=.5*F.binary_cross_entropy_with_logits(z,self.y)+2*F.binary_cross_entropy_with_logits(z/2,torch.sigmoid(u/2))
  self.compare_grads(self.loss(u),expected)
 def test_all_conflict_equals_half_bce(self):
  for scale in [1.,100.]:
   u=(1-2*self.y)*scale
   self.compare_grads(self.loss(u),.5*F.binary_cross_entropy_with_logits(self.model(self.x),self.y))
 def test_mixed_full_batch_mean(self):
  u=torch.tensor([-3.,3.,2.,-2.,-1.,1.,1.,-1.]);z=self.model(self.x);mask=((u>=0)==(self.y>=.5)).float()
  expected=(.5*F.binary_cross_entropy_with_logits(z,self.y,reduction='none')+2*mask*F.binary_cross_entropy_with_logits(z/2,torch.sigmoid(u/2),reduction='none')).mean()
  self.compare_grads(self.loss(u),expected)
 def test_zero_logit_is_positive(self):
  u=torch.zeros(8);z=self.model(self.x);mask=self.y
  expected=(.5*F.binary_cross_entropy_with_logits(z,self.y,reduction='none')+2*mask*F.binary_cross_entropy_with_logits(z/2,torch.ones(8)*.5,reduction='none')).mean()
  torch.testing.assert_close(self.loss(u),expected)
 def test_invalid_inputs(self):
  with self.assertRaises(ValueError):self.loss(torch.full((8,),float('nan')))
  with self.assertRaises(ValueError):t.training_loss(self.model,self.x,self.y,torch.zeros(8),'B0',None)
 def test_gradients_finite(self):
  self.model.train();l=self.loss(torch.randn(8));l.backward();self.assertTrue(torch.isfinite(l));self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.model.parameters()))
 def test_tiny_fold_partition_and_restore(self):
  protocol=json.loads((Path(__file__).parent/'PROTOCOL.json').read_text());protocol['student'].update(maximum_epochs=2,early_stopping_patience=2,batch_size=8)
  x=np.random.default_rng(2).normal(size=(40,4)).astype(np.float32)
  master=pd.DataFrame(dict(clean_id=['x'+str(i) for i in range(40)],label=[0,1]*20,fold=np.repeat(np.arange(5),8),homology_group=['g'+str(i) for i in range(40)],teacher_logit=np.arange(40,dtype=float)-20))
  choose=t.choose_epoch;fit=t.fit_epochs
  def checked_choose(xs,y,u,w,xv,yv,*args):
   np.testing.assert_array_equal(u,master.loc[master.fold>=2,'teacher_logit'].to_numpy(np.float32));np.testing.assert_array_equal(y,master.loc[master.fold>=2,'label'].to_numpy(np.float32));return choose(xs,y,u,w,xv,yv,*args)
  def checked_fit(xs,y,u,w,*args):
   np.testing.assert_array_equal(u,master.loc[master.fold!=0,'teacher_logit'].to_numpy(np.float32));np.testing.assert_array_equal(y,master.loc[master.fold!=0,'label'].to_numpy(np.float32));return fit(xs,y,u,w,*args)
  with tempfile.TemporaryDirectory() as tmp,patch.object(t,'choose_epoch',side_effect=checked_choose),patch.object(t,'fit_epochs',side_effect=checked_fit):
   out=Path(tmp);r=t.run_fold('G0',0,x,master,None,protocol,torch.device('cpu'),out);self.assertEqual(len(r),8)
   ck=torch.load(out/'partial/G0/fold_0.pt',map_location='cpu',weights_only=False);self.assertTrue(ck['teacher_supervision_used']);self.assertEqual(ck['module'],'R7')
   model=t.Student(4,.2);model.load_state_dict(ck['model_state_dict']);z=((x[:8].astype(float)-ck['scaler_mean'])/ck['scaler_scale']).astype(np.float32);raw,_=t.predict(model,z,torch.device('cpu'),8);np.testing.assert_allclose(raw,r.student_raw_logit,atol=1e-6)
   meta=json.loads((out/'partial/G0/fold_0.json').read_text());self.assertEqual(meta['inner_training_records'],24)
if __name__=='__main__':
 torch.set_num_threads(1);unittest.main(verbosity=2)
