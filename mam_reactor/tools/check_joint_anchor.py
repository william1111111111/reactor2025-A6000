import os
import sys
import io
from pathlib import Path
import torch
root=Path('/public/zhaowenjie/react2025_new/mam_reactor')
sys.path.insert(0,str(root))
os.environ['MAM_REACTOR_ROOT']=str(root)
from regnn.conditional_model import ConditionalREGNN, ConditionalREGNNConfig, ConditionalREGNNLoss
from regnn.conditional_data import ConditionalReactionDataset, crop_and_pad
from regnn.anchor_3dmm_loss import geometry_loss
torch.set_num_threads(1)
torch.manual_seed(1)
ds=ConditionalReactionDataset(str(root.parent/'data'),'train',training=False,load_listener_3dmm=True)
b=ds[0]
raw=ds._load_tensor(ds.coefficient_dir/ds._paired_target(ds.records[0])).squeeze()
expected=crop_and_pad((raw-ds.mean_face)/ds.std_face,int(b['start']),750)
torch.testing.assert_close(expected,b['target_3dmm'])
model=ConditionalREGNN(ConditionalREGNNConfig(listener_3dmm=True)).cuda()
inputs=[b[k][None].cuda() for k in ['speaker_audio','speaker_emotion','speaker_3dmm','length']]
with torch.autocast('cuda',dtype=torch.bfloat16):
    o=model(*inputs)
    emotion_loss=ConditionalREGNNLoss()(o['prediction'],b['target'][None].cuda(),o['valid_mask'])['loss']
g=geometry_loss(o['prediction_3dmm'],b['target_3dmm'][None].cuda(),o['valid_mask'])
geometry_grad=torch.autograd.grad(g['geometry_mse'],model.condition_fusion.fusion[0].weight,retain_graph=True)[0]
assert geometry_grad.abs().sum()>0 and torch.isfinite(geometry_grad).all()
total=emotion_loss+g['geometry_mse']+0.1*g['geometry_velocity']
total.backward()
assert all(p.requires_grad for p in model.parameters())
for name in ['to_listener_3dmm.weight','to_initial_reaction.weight','condition_fusion.fusion.0.weight']:
    grad=dict(model.named_parameters())[name].grad
    assert grad is not None and grad.abs().sum()>0 and torch.isfinite(grad).all(),name
before=model.to_listener_3dmm.weight.detach().clone()
torch.optim.AdamW(model.parameters(),lr=1e-4).step()
assert not torch.equal(before,model.to_listener_3dmm.weight)
buf=io.BytesIO(); torch.save(model.state_dict(),buf); buf.seek(0)
restored=ConditionalREGNN(ConditionalREGNNConfig(listener_3dmm=True))
restored.load_state_dict(torch.load(buf,map_location='cpu'),strict=True)
old=torch.load(root/'checkpoints/conditional_regnn_offline_anchor_epoch0050.pth',map_location='cpu')
ConditionalREGNN(ConditionalREGNNConfig(**old['model_config'])).load_state_dict(old['state_dict'],strict=True)
p=torch.randn(2,5,58,requires_grad=True); t=torch.randn_like(p); mask=torch.tensor([[1,1,1,0,0],[1,1,0,0,0]],dtype=torch.bool)
one=geometry_loss(p,t,mask); altered=t.detach().clone(); altered[~mask]=100
two=geometry_loss(p,altered,mask)
for k in one: torch.testing.assert_close(one[k],two[k])
print('PASS: paired alignment; geometry-only gradients reach shared backbone; both heads trained; optimizer update; strict checkpoint roundtrip; legacy checkpoint; padding invariance')
print('samples',len(ds),'outputs',o['prediction'].shape,o['prediction_3dmm'].shape,'loss',float(total),'geometry', {k:float(v) for k,v in g.items()})
