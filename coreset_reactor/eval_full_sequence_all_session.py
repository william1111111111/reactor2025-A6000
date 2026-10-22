"""Matched full-sequence, both-direction, all-session VAL evaluation.

Two frozen models share precisely the same target tensor for every context.
Offline 750-frame chunks are concatenated before any metric is computed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F

from coreset_reactor.evaluate import _deterministic_postprocess, official_prediction
from coreset_reactor.fast_frc import candidate_frc
from coreset_reactor.exact_dtw_cuda import pairwise_frd
from framework.metrics import compute_s_mse, compute_FRVar
from framework.modules.post_processor import Processor
from regnn.conditional_model import ConditionalREGNN, ConditionalREGNNConfig
from regnn.expert_student import ExpertConditionalREGNN, ExpertStudentConfig
from regnn.emotion_query_mamba import EmotionQueryResidualMamba, EmotionQueryMambaConfig
from regnn.eval_query_mamba_official_test import generate_official_offline_query_batch

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
ASSET = ROOT/'mam_reactor'
STUDENT = ROOT/'coreset_reactor/reports/mode_reference_student_seed1/basic/student.pth'
MAM = WORKSPACE/'set_matching_project/mam_reactor/checkpoints/mam_reactor_offline_evidence_epoch0003.pth'
MODELS = ('student', 'set_matching')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    os.replace(tmp,path)


def full_student(model, streams, device, chunk_length=750, batch_size=8):
    length = streams[0].shape[0]
    if length < 1 or any(v.shape[0] != length for v in streams):
        raise ValueError('Source streams must have the same positive length')
    chunks = (length + chunk_length - 1)//chunk_length
    lengths = torch.full((chunks,),chunk_length,dtype=torch.long)
    lengths[-1] = length-(chunks-1)*chunk_length
    inputs = [F.pad(v,(0,0,0,chunks*chunk_length-length)).reshape(chunks,chunk_length,-1)
              for v in streams]
    outputs=[]
    for start in range(0,chunks,batch_size):
        batch=[v[start:start+batch_size].to(device) for v in inputs]
        pred=model.generate(*batch,lengths[start:start+batch_size].to(device))['prediction']
        outputs.append(pred.detach().float().cpu())
    result=torch.cat(outputs).permute(1,0,2,3).reshape(10,-1,25)[:,:length].contiguous()
    return official_prediction(result)


def prepare(output):
    if (output/'manifest.json').exists():
        raise FileExistsError('Use the existing frozen manifest or a fresh output directory')
    facial=ROOT/'data/val/facial-attributes'
    pools={(role,p.name):sorted(p.glob('*.npy')) for role in ('speaker','listener')
           for p in (facial/role).iterdir() if p.is_dir()}
    file_hashes={str(p.resolve()):sha(p) for paths in pools.values() for p in paths}
    entries=[]
    for role in ('speaker','listener'):
        other='listener' if role=='speaker' else 'speaker'
        for source in sorted((facial/role).glob('*/*.npy')):
            rel=source.relative_to(facial)
            paths=[ROOT/'data/val'/folder/rel for folder in ('audio-features','facial-attributes','coefficients')]
            lengths=[len(np.load(p,mmap_mode='r')) for p in paths]
            if len(set(lengths))!=1:
                raise ValueError(f'Source stream length mismatch: {rel}/{lengths}')
            paired=facial/other/source.parent.name/source.name
            targets=[paired]+[p for p in pools[other,source.parent.name] if p!=paired]
            entries.append({'index':len(entries),'clip_id':rel.with_suffix('').as_posix(),
                'role':role,'length':lengths[0],
                'source_files':[(str(p.resolve()),sha(p)) for p in paths],
                'gt_files':[(str(p.resolve()),file_hashes[str(p.resolve())]) for p in targets]})
    assert len(entries)==1142 and sum(v['role']=='speaker' for v in entries)==571
    code_paths=[Path(__file__),Path(__file__).with_name('exact_dtw_cuda.py'),
                Path(__file__).with_name('evaluate.py'),Path(__file__).with_name('fast_frc.py')]
    for folder in ('regnn','framework/modules','framework/metrics'):
        code_paths.extend(sorted((ASSET/folder).rglob('*.py')))
    manifest={'protocol':'VAL both directions, full sequence via offline750 chunks, K10, all opposite-role same-session GT, official Processor full duration, official AU rounding',
              'eval_seed':1234,'checkpoints':{'student':(str(STUDENT),sha(STUDENT)),
                                            'set_matching':(str(MAM),sha(MAM))},
              'postprocessor':sha(ASSET/'pretrained_models/post_processor/checkpoint.pth'),
              'source_hashes':{str(p):sha(p) for p in code_paths},
              'entries':entries}
    write(output/'manifest.json',manifest)
    print('Frozen 1142 contexts and all GT membership',flush=True)


def aggregate(output,manifest):
    rows=[json.loads((output/'rows'/f"{e['index']:04d}.json").read_text()) for e in manifest['entries']]
    assert [r['clip_id'] for r in rows]==[e['clip_id'] for e in manifest['entries']]
    metrics={}
    for group in ('speaker','listener','both'):
        selected=[r for r in rows if group=='both' or r['role']==group]
        metrics[group]={'contexts':len(selected),'models':{name:{k:float(np.mean([r['models'][name][k] for r in selected]))
                  for k in ('FRC','exact_FRD','FRDiv','FRVar')} for name in MODELS}}
    result={'protocol':manifest['protocol'],'manifest_sha256':sha(output/'manifest.json'),
            'checkpoints':manifest['checkpoints'],'metrics':metrics,
            'limitations':['Validation data, not hidden TEST','All-session GT retained deliberately',
                           'Full output coverage using 750-frame offline chunks, not one unlimited-context forward pass'],
            'rows':rows}
    write(output/'comparison.json',result)
    print(json.dumps(metrics,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prepare',action='store_true')
    p.add_argument('--aggregate',action='store_true')
    p.add_argument('--shard',type=int,default=0)
    p.add_argument('--shards',type=int,default=6)
    p.add_argument('--max-contexts',type=int)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    if args.prepare:
        prepare(args.output);return
    manifest=json.loads((args.output/'manifest.json').read_text())
    if args.aggregate:
        aggregate(args.output,manifest);return
    for path,digest in manifest['source_hashes'].items():
        assert sha(path)==digest, f'Source changed: {path}'
    for _,(path,digest) in manifest['checkpoints'].items():
        assert sha(path)==digest
    assert sha(ASSET/'pretrained_models/post_processor/checkpoint.pth')==manifest['postprocessor']
    assert 0<=args.shard<args.shards
    device=torch.device('cuda:0');torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(1234);np.random.seed(1234)
    state=torch.load(STUDENT,map_location='cpu');cfg=state['student_config']
    student=ExpertConditionalREGNN(ExpertStudentConfig(backbone=ConditionalREGNNConfig(**cfg['backbone']),
        num_experts=cfg['num_experts'],expert_embedding_dim=cfg['expert_embedding_dim'])).to(device).eval()
    student.load_state_dict(state['state_dict'],strict=True)
    state=torch.load(MAM,map_location='cpu')
    mam=EmotionQueryResidualMamba(ConditionalREGNN(ConditionalREGNNConfig(**state['anchor_model_config'])),
        EmotionQueryMambaConfig.from_checkpoint_payload(state['emotion_query_model_config'])).to(device).eval()
    mam.load_state_dict(state['state_dict'],strict=True)
    processor=Processor(cfg_dir=str(ASSET),ckpt_dir=str(ASSET/'pretrained_models/post_processor'),
                        clip_len_test=1000,device=device,num_preds=10)
    mean=torch.from_numpy(np.load(ASSET/'external/FaceVerse/mean_face.npy').astype('float32')).reshape(58)
    std=torch.from_numpy(np.load(ASSET/'external/FaceVerse/std_face.npy').astype('float32')).reshape(58).clamp_min(1e-8)
    for folder in ('rows','arrays'):
        (args.output/folder).mkdir(exist_ok=True)
    provenance={'manifest_sha256':sha(args.output/'manifest.json'),'shard':args.shard,'shards':args.shards,
                'visible_gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'torch':torch.__version__,
                'tf32':False,'seed':1234}
    pp=args.output/f'provenance_{args.shard}.json'
    if pp.exists():assert json.loads(pp.read_text())==provenance
    else:write(pp,provenance)
    # Largest sequences first to prevent a single late long-context tail.
    schedule=sorted(manifest['entries'],key=lambda e:(-e['length'],e['clip_id']))[args.shard::args.shards]
    started=time.monotonic();completed=0
    with torch.inference_mode():
        for entry in schedule:
            index=entry['index'];result_path=args.output/'rows'/f'{index:04d}.json'
            if result_path.exists():continue
            array_path=args.output/'arrays'/f'{index:04d}.pt'
            tick=time.monotonic()
            if array_path.exists():
                data=torch.load(array_path,map_location='cpu')
                assert data['manifest_sha256']==provenance['manifest_sha256']
                target=data['targets'];preds=data['predictions']
            else:
                for path,digest in entry['source_files']+entry['gt_files']:
                    assert sha(path)==digest,f'Data changed: {path}'
                streams=[torch.from_numpy(np.load(path).astype('float32')).reshape(entry['length'],-1)
                         for path,_ in entry['source_files']]
                streams[2]=(streams[2]-mean)/std
                preds={'student':full_student(student,streams,device)}
                preds['set_matching']=generate_official_offline_query_batch(mam,[streams[0]],[streams[1]],
                    [streams[2]],[entry['length']],device,750,8,10,0.95,True,
                    style_residual_scale=0.95,residual_channel_multipliers=([1.8]*15+[0.4]*2+[1.6]*8))[0]
                for prediction in preds.values():
                    assert prediction.shape==(10,entry['length'],25) and torch.isfinite(prediction).all()
                raw=[torch.from_numpy(np.load(path).astype('float32')) for path,_ in entry['gt_files']]
                target=_deterministic_postprocess(processor,preds['student'],raw,device,1234,entry['clip_id']).cpu()
                assert target.shape==(len(raw),entry['length'],25) and torch.isfinite(target).all()
                tmp=array_path.with_suffix('.tmp')
                torch.save({'manifest_sha256':provenance['manifest_sha256'],'clip_id':entry['clip_id'],
                            'targets':target,'predictions':preds},tmp);os.replace(tmp,array_path)
            # Both models are scored in one batch against the same full GT tensor.
            distances=pairwise_frd(torch.cat([preds[n] for n in MODELS]).numpy(),target.numpy())
            values={}
            for offset,name in enumerate(MODELS):
                prediction=preds[name]
                frc=candidate_frc(target.numpy(),prediction.numpy())
                frd=distances[offset*10:(offset+1)*10].min(axis=1)
                values[name]={'FRC':float(frc.sum()),'exact_FRD':float(frd.sum()),
                              'FRDiv':float(compute_s_mse([prediction])),
                              'FRVar':float(compute_FRVar([prediction])),
                              'FRC_per_prediction':frc.tolist(),'FRD_per_prediction':frd.tolist()}
            row={'index':index,'clip_id':entry['clip_id'],'role':entry['role'],'length':entry['length'],
                 'gt_count':target.shape[0],'models':values,'seconds':time.monotonic()-tick}
            write(result_path,row);completed+=1
            print(json.dumps({'index':index,'role':entry['role'],'frames':entry['length'],
                'completed':completed,'total_shard':len(schedule),'elapsed_s':time.monotonic()-started,
                'seconds':row['seconds']}),flush=True)
            if args.max_contexts and completed>=args.max_contexts:break
    write(args.output/f'shard_status_{args.shard}.json',{'newly_completed':completed,
        'all_done':all((args.output/'rows'/f"{e['index']:04d}.json").exists() for e in schedule),
        'seconds':time.monotonic()-started})


if __name__=='__main__':main()
