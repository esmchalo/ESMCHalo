import unittest,copy
import numpy as np,torch
import averaging as av
import legacy_v205 as old
class Tests(unittest.TestCase):
 def setUp(self):
  torch.manual_seed(7);model=old.Student(4,.2)
  self.a=dict(arm='H0',outer_fold=0,canonical_mean_only=True,lora=False,model_state_dict=model.state_dict(),scaler_mean=np.zeros(4),scaler_scale=np.ones(4))
  self.b=copy.deepcopy(self.a);self.b['arm']='Mixup'
 def test_midpoint_exact(self):
  for k,v in self.b['model_state_dict'].items():self.b['model_state_dict'][k]=v+1
  m=av.merge_checkpoints(self.a,self.b,0)
  for k,v in m['model_state_dict'].items():torch.testing.assert_close(v,.5*self.a['model_state_dict'][k]+.5*self.b['model_state_dict'][k])
  self.assertEqual(m['arm'],'S0');self.assertFalse(m['trained'])
 def test_identical_weights_same_predictions(self):
  x=np.random.default_rng(1).normal(size=(20,4)).astype(np.float32)
  a,_=av.score(self.a,x,torch.device('cpu'));b,_=av.score(av.merge_checkpoints(self.a,self.b,0),x,torch.device('cpu'));np.testing.assert_array_equal(a,b)
 def test_fold_mismatch_rejected(self):
  self.b['outer_fold']=1
  with self.assertRaises(ValueError):av.merge_checkpoints(self.a,self.b,0)
 def test_scaler_mismatch_rejected(self):
  self.b['scaler_mean'][0]=1e-12
  with self.assertRaises(ValueError):av.merge_checkpoints(self.a,self.b,0)
 def test_parameter_mismatch_rejected(self):
  self.b['model_state_dict'].pop(next(iter(self.b['model_state_dict'])))
  with self.assertRaises(ValueError):av.merge_checkpoints(self.a,self.b,0)
 def test_nan_rejected(self):
  k=next(iter(self.b['model_state_dict']));self.b['model_state_dict'][k].fill_(float('nan'))
  with self.assertRaises(ValueError):av.merge_checkpoints(self.a,self.b,0)
 def test_standardization(self):
  self.a['scaler_mean']=np.arange(4,dtype=float);self.a['scaler_scale']=np.ones(4)*2;x=np.arange(8,dtype=np.float32).reshape(2,4)
  np.testing.assert_array_equal(av.standardized(x,self.a),((x-self.a['scaler_mean'])/2).astype(np.float32))
if __name__=='__main__':
 torch.set_num_threads(1);unittest.main(verbosity=2)
