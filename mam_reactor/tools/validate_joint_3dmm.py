"""Deterministic paired-listener validation, not official challenge metrics."""
import os
import sys
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader

root = Path('/public/zhaowenjie/react2025_new/mam_reactor')
sys.path.insert(0, str(root))
os.environ['MAM_REACTOR_ROOT'] = str(root)
from regnn.conditional_data import ConditionalReactionDataset
from regnn.conditional_model import ConditionalREGNN, ConditionalREGNNConfig
from regnn.emotion_query_mamba import EmotionQueryResidualMamba, EmotionQueryMambaConfig
from regnn.query_mamba import apply_residual_channel_multipliers

torch.set_num_threads(1)
torch.manual_seed(1234)
torch.backends.cuda.matmul.allow_tf32 = False
run = root/'runs/joint_3dmm_trial_20260917'
outdir = run/'validation_val'
outdir.mkdir(exist_ok=False)
ds = ConditionalReactionDataset(str(root.parent/'data'), 'val', clip_length=750,
    training=False, target_mode='session_cropped', num_targets=1,
    target_selection_mode='fixed_first', load_listener_3dmm=True)
loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=2)
paths = {'baseline25': root/'runs/offline_20260917_172106/emotion-query-mamba-epoch0003-seed1.pth'}
paths.update({f'joint_epoch{e}': run/f'emotion-query-mamba-epoch{e:04d}-seed1.pth' for e in [1,2,3]})
models = {}
for name, path in paths.items():
    ck = torch.load(path, map_location='cpu')
    model = EmotionQueryResidualMamba(ConditionalREGNN(ConditionalREGNNConfig(**ck['anchor_model_config'])),
        EmotionQueryMambaConfig.from_checkpoint_payload(ck['emotion_query_model_config']))
    model.load_state_dict(ck['state_dict'], strict=True)
    models[name] = model.cuda().eval()
stats = {name:{} for name in [*models, 'mean_face', 'copy_speaker']}
rows = []
std = ds.std_face.cuda()

def mse(p, t, mask):
    return ((p-t).square()*mask[...,None]).sum((-1,-2))/(mask.sum(-1).clamp_min(1)*p.shape[-1])

def ccc(p,t,mask):
    n = mask.sum(-1).clamp_min(1)[:,None]
    w = mask[...,None]
    mp,mt = (p*w).sum(1)/n,(t*w).sum(1)/n
    vp,vt = (((p-mp[:,None])**2)*w).sum(1)/n,(((t-mt[:,None])**2)*w).sum(1)/n
    cov = ((p-mp[:,None])*(t-mt[:,None])*w).sum(1)/n
    return (2*cov/(vp+vt+(mp-mt)**2+1e-8)).mean(-1)

def add(name, key, values):
    assert torch.isfinite(values).all(), (name,key)
    stats[name].setdefault(key,[]).extend(values.cpu().tolist())

with torch.inference_mode():
    for bi,b in enumerate(loader):
        a,e,s,l = [b[k].cuda() for k in ['speaker_audio','speaker_emotion','speaker_3dmm','length']]
        t,g = b['target'].cuda(),b['targets_3dmm'][:,0].cuda()
        mask = torch.arange(750,device='cuda')[None,:] < l[:,None]
        assert mask.any(-1).all()
        for name, prediction in [('mean_face',torch.zeros_like(g)),('copy_speaker',s)]:
            add(name,'geometry_normalized_mse',mse(prediction,g,mask))
            add(name,'geometry_raw_mse',mse(prediction*std,g*std,mask))
        for name,model in models.items():
            output = model(a,e,s,l,residual_scale=0.95,style_residual_scale=0.95)
            p = apply_residual_channel_multipliers(model,output,[1.8]*15+[0.4]*2+[1.6]*8).float()
            add(name,'emotion_q9_mse',mse(p[:,9],t,mask))
            add(name,'emotion_q9_ccc',ccc(p[:,9],t,mask))
            add(name,'emotion_anchor_mse',mse(p[:,0],t,mask))
            add(name,'emotion_best9_mse',torch.stack([mse(p[:,q],t,mask) for q in range(1,10)],-1).min(-1).values)
            if 'prediction_3dmm' in output:
                pred = output['prediction_3dmm'].float()
                q9 = pred[:,8]
                add(name,'geometry_normalized_mse',mse(q9,g,mask))
                add(name,'geometry_raw_mse',mse(q9*std,g*std,mask))
                add(name,'geometry_best9_mse',torch.stack([mse(pred[:,q],g,mask) for q in range(9)],-1).min(-1).values)
                add(name,'geometry_velocity_mse',mse(q9[:,1:]-q9[:,:-1],g[:,1:]-g[:,:-1],mask[:,1:] & mask[:,:-1]))
        rows.extend(b['clip_id'])
        if bi % 25 == 0:
            print(f'validated {len(rows)}/{len(ds)}',flush=True)
summary = {name:{key:sum(v)/len(v) for key,v in metrics.items()} for name,metrics in stats.items()}
result = {'protocol':'val paired listener, deterministic center crop <=750 frames; FP32; per-clip average; q9 is fixed mixture, best9 uses ground-truth selection and is optimistic; NOT official challenge metrics',
    'num_clips':len(rows),'models':{k:str(v) for k,v in paths.items()},'metrics':summary}
(outdir/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
(outdir/'per_clip.json').write_text(json.dumps({'clip_ids':rows,'metrics':stats})+'\n')
print(json.dumps(result,indent=2),flush=True)
