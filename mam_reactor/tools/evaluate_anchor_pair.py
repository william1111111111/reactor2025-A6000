import os, sys, json, random
from pathlib import Path
import numpy as np
import torch
root=Path('/public/zhaowenjie/react2025_new/mam_reactor')
sys.path.insert(0,str(root))
os.environ['MAM_REACTOR_ROOT']=str(root)
from regnn.conditional_model import ConditionalREGNN, ConditionalREGNNConfig
from regnn.eval_conditional_regnn_official_test import initialize_official_hydra_runtime, build_official_test_loader, generate_official_offline_batch
from regnn.eval_emotion_query_mamba_official_test import load_official_test_targets_from_results_cache
from framework.metrics import compute_FRC, compute_s_mse, compute_FRVar, compute_TLCC
torch.set_num_threads(1)
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32=True
torch.backends.cudnn.allow_tf32=True
initialize_official_hydra_runtime(root)
base=root/'runs/anchor_comparison_20260917'
base.mkdir(exist_ok=False)
paths={'old':root/'checkpoints/conditional_regnn_offline_anchor_epoch0050.pth', 'new':root/'runs/joint_anchor_scratch_20260917/conditional-epoch0050-seed1.pth'}
for name,path in paths.items():
    random.seed(1234); np.random.seed(1234); torch.manual_seed(1234); torch.cuda.manual_seed_all(1234)
    ck=torch.load(path,map_location='cpu')
    model=ConditionalREGNN(ConditionalREGNNConfig(**ck['model_config'])).cuda().eval()
    model.load_state_dict(ck['state_dict'],strict=True)
    ds,loader=build_official_test_loader(root.parent/'data',750,4,4,1234)
    predictions=[]; speakers=[]
    with torch.inference_mode():
        for i,batch in enumerate(loader):
            a,_,e,s,_,_,_,lengths,_=batch
            p=generate_official_offline_batch(model=model,speaker_audio_clips=a,speaker_emotion_clips=e,speaker_3dmm_clips=s,speaker_seq_lengths=lengths,device=torch.device('cuda'),clip_length=750,clip_batch_size=8,num_preds=10,data_clamp=True)
            predictions.extend(p); speakers.extend([x.clone() for x in e])
            if i%25==0: print(name,'generated',len(predictions),'/',len(ds),flush=True)
    targets,_=load_official_test_targets_from_results_cache(root/'assets/evaluation/generated_local_test1142_seed1234.pt',predictions,root.parent/'data',1234,750)
    assert len(predictions)==len(ds)==1142
    out=base/name; out.mkdir()
    torch.save({'PRED':predictions,'GT':targets,'metadata':{'checkpoint':str(path),'full_test':True,'num_examples':1142,'split':'test','num_preds':10,'selection_seed':1234,'output_dim':25}},out/'results_frd.pt')
    metrics={'FRC':float(compute_FRC(predictions,targets,p=8)),'FRDiv':float(compute_s_mse(predictions)), 'FRVar':float(compute_FRVar(predictions)),'TLCC':float(compute_TLCC(predictions,speakers,p=8)), 'samples':len(predictions),'checkpoint':str(path),'epoch':ck['epoch'],'note':'Deterministic anchor repeated ten times; 25D only.'}
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print(name,metrics,flush=True)
    del model,predictions,targets,speakers,loader,ds,ck
    torch.cuda.empty_cache()
