import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, '/public/zhaowenjie/react2025_new/mam_reactor')
from regnn.joint_3dmm import pairwise_mse, listener_3dmm_loss
from regnn.conditional_data import ConditionalReactionDataset, crop_and_pad
from regnn.conditional_model import ConditionalREGNN, ConditionalREGNNConfig
from regnn.emotion_query_mamba import EmotionQueryResidualMamba, EmotionQueryMambaConfig
from regnn.train_emotion_query_mamba import load_emotion_warmstart

root = Path('/public/zhaowenjie/react2025_new/mam_reactor')
torch.set_num_threads(1)
torch.manual_seed(1)
p = torch.randn(2, 3, 7, 58, requires_grad=True)
t = torch.randn(2, 4, 7, 58)
m = torch.rand(2, 4, 7) > 0.3
m[0, 3] = False
fast = pairwise_mse(p, t, m)
slow = ((p[:, :, None] - t[:, None]).square() * m[:, None, :, :, None]).sum((-1, -2)) / (m.sum(-1)[:, None].clamp_min(1) * 58)
torch.testing.assert_close(fast, slow, atol=1e-6, rtol=1e-5)
ep, et = torch.randn(2, 3, 7, 25), torch.randn(2, 4, 7, 25)
loss = listener_3dmm_loss(p, ep, t, et, m)
altered = t.clone(); altered[~m] = 1000
loss2 = listener_3dmm_loss(p, ep, altered, et, m)
for k in loss:
    torch.testing.assert_close(loss[k], loss2[k])
sum(loss.values()).backward()
assert torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
print('PASS masked loss, padding invariance, gradients', flush=True)

ds = ConditionalReactionDataset(str(root.parent/'data'), 'train', clip_length=750, training=False, target_mode='session_cropped', load_listener_3dmm=True)
item = ds[0]
for k, path in enumerate(item['target_paths']):
    raw = ds._load_tensor(ds.coefficient_dir / Path(path).with_suffix('.npy')).squeeze()
    expected = crop_and_pad((raw - ds.mean_face)/ds.std_face, int(item['target_starts'][k]), 750)
    torch.testing.assert_close(item['targets_3dmm'][k], expected)
print('PASS real target alignment', item['targets'].shape, item['targets_3dmm'].shape, flush=True)
ck = torch.load(root/'checkpoints/conditional_regnn_offline_anchor_epoch0050.pth', map_location='cpu')
anchor = ConditionalREGNN(ConditionalREGNNConfig(**ck['model_config']))
anchor.load_state_dict(ck['state_dict'])
cfg = EmotionQueryMambaConfig(listener_3dmm=True, static_style_adapter=True, compatibility_mode='legacy_relative_softmax', gate_floor=1.)
model = EmotionQueryResidualMamba(anchor, cfg).cuda()
load_emotion_warmstart(model, root/'checkpoints/eqr_legacy_warmstart_epoch0003.pth')
inputs = [item[k].unsqueeze(0).cuda() for k in ['speaker_audio','speaker_emotion','speaker_3dmm','length']]
with torch.autocast('cuda', dtype=torch.bfloat16):
    out = model(*inputs)
assert out['prediction'].shape == (1,10,750,25)
assert out['prediction_3dmm'].shape == (1,9,750,58)
mask = torch.arange(750)[None,:] < item['target_lengths'][:,None]
mask &= item['target_available_mask'][:,None]
ls = listener_3dmm_loss(out['prediction_3dmm'], out['prediction'][:,1:], item['targets_3dmm'][None].cuda(), item['targets'][None].cuda(), mask[None].cuda())
sum(ls.values()).backward()
assert model.to_3dmm.weight.grad.abs().sum() > 0
assert model.context_projection.weight.grad.abs().sum() > 0
assert all(p.grad is None for p in model.anchor.parameters())
assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
print('PASS real forward/backward, new head and shared trunk gradients, frozen anchor', {k: float(v) for k,v in ls.items()}, flush=True)
