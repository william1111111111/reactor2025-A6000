"""Real-GT targets for a matched no-distillation control.

Reuse only source-view metadata from the frozen KD cache, never its output
values. Rank 0 is native paired; ranks 1..9 follow the current teachers'
relative_time_masked preprocessing. The shared cache tensor key is retained
for trainer compatibility, but provenance explicitly identifies real GT.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import torch

from coreset_reactor.teacher_cache import atomic_torch_save, sha256_file
from regnn.conditional_data import ConditionalReactionDataset, crop_and_pad, relative_time_resample_reaction

VIEW_KEYS = ('sample_index','sample_id','view_id','view_epoch','start','length','generated_length')


def real_target(raw, rank, start, valid_length, clip_length):
    if rank == 0:
        if len(raw) - start < valid_length:
            raise ValueError('Paired target is shorter than frozen input view')
        return crop_and_pad(raw,start,clip_length)
    return crop_and_pad(relative_time_resample_reaction(raw,valid_length),0,clip_length)


def build(data_root, source_manifest, fixed_manifests, output):
    source_manifest=Path(source_manifest);data_root=Path(data_root)
    source=json.loads(source_manifest.read_text())
    if source['split']!='train' or source['num_experts']!=10 or source['contexts']!=3320:
        raise ValueError('Expected full REACT2025 ten-target TRAIN cache')
    if len(fixed_manifests)!=10:raise ValueError('Ten rank manifests required')
    mapping=[]
    for path in fixed_manifests:
        m=json.loads(Path(path).read_text())
        if m['split']!='train':raise ValueError('GT selection must be TRAIN-only')
        mapping.append(m['mapping'])
    dataset=ConditionalReactionDataset(str(data_root),'train',clip_length=source['clip_length'],
        training=True,target_mode='paired',seed=source['source_seed'])
    dataset.set_epoch(source['view_epoch'])
    # Independent rank loaders audit the helper against production teacher preprocessing.
    rank_loaders=[ConditionalReactionDataset(str(data_root),'train',clip_length=source['clip_length'],
        training=True,target_mode='paired',seed=source['source_seed'],fixed_pair_rank=r,
        fixed_pair_count=10,fixed_pair_seed=1,fixed_pair_manifest=str(fixed_manifests[r]),
        fixed_pair_alignment_policy='relative_time_masked') for r in range(10)]
    for d in rank_loaders:d.set_epoch(source['view_epoch'])
    audit_indices=set(torch.linspace(0,source['contexts']-1,30).long().tolist())
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    raw_cache={};raw_hashes={};view_digest=hashlib.sha256();completed=0;parity=0
    for shard_name in source['shards']:
        old=torch.load(source_manifest.parent/shard_name,map_location='cpu')
        new=[]
        for row in old:
            idx=row['sample_index'];sample=dataset[(idx,row['view_id'])]
            if (sample['clip_id'],int(sample['start']),int(sample['length']),int(sample['crop_view_id'])) != (
                    row['sample_id'],row['start'],row['length'],row['view_id']):
                raise ValueError(f'Frozen source-view mismatch: {idx}')
            source_key=sample['clip_id']+'.npy'
            selected=[mapping[r][source_key][r] for r in range(10)]
            if len(set(selected))!=10:raise ValueError('GT identities must be distinct')
            # Every rank manifest must represent the same ordered target set.
            if any(mapping[r][source_key]!=selected for r in range(10)):
                raise ValueError('Rank manifests disagree on target identities/order')
            targets=[]
            for rank,relative in enumerate(selected):
                path=dataset.emotion_dir/relative
                if relative not in raw_cache:
                    raw_cache[relative]=dataset._load_tensor(path)
                    raw_hashes[relative]=sha256_file(path)
                target=real_target(raw_cache[relative],rank,row['start'],row['length'],source['clip_length'])
                if idx in audit_indices:
                    check=rank_loaders[rank][(idx,row['view_id'])]
                    if int(check['length'])!=row['length']:raise ValueError('Teacher/source valid length mismatch')
                    for key in ('speaker_audio','speaker_emotion','speaker_3dmm'):
                        torch.testing.assert_close(check[key],sample[key],rtol=0,atol=0)
                    torch.testing.assert_close(check['target'],target,rtol=0,atol=0)
                    parity+=1
                targets.append(target)
            value=torch.stack(targets)
            if not torch.isfinite(value).all():raise ValueError('Nonfinite real target')
            meta={k:row[k] for k in VIEW_KEYS}
            view_digest.update(json.dumps([meta[k] for k in VIEW_KEYS],separators=(',',':')).encode())
            # Match the existing KD cache's FP16 target storage, and convert back
            # to FP32 in the unmodified student trainer.
            new.append({**meta,'teacher_seed':None,'teacher_predictions':value.half(),
                        'target_kind':'real_gt','fixed_gt_ids':selected})
            completed+=1
        atomic_torch_save(new,output/shard_name)
        print(f'real-GT cache {completed}/{source["contexts"]}',flush=True)
    if completed!=source['contexts'] or parity!=300:raise ValueError('Incomplete view/parity audit')
    manifest={**source,'teacher_checkpoint_hashes':[],
        'teacher_generation':'NONE: real GT only; legacy tensor key for trainer compatibility',
        'target_kind':'real_gt','target_alignment':'relative_time_masked; paired rank0 native crop',
        'source_view_manifest_sha256':sha256_file(source_manifest),
        'fixed_gt_manifest_sha256':[sha256_file(Path(p)) for p in fixed_manifests],
        'real_gt_files_sha256':raw_hashes,'view_audit_sha256':view_digest.hexdigest(),
        'target_precision':'FP16 storage, FP32 training like matched KD cache',
        'production_target_parity_examples':parity}
    manifest.pop('cache_manifest_hash',None)
    manifest['cache_manifest_hash']=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('data-root','source-manifest','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--fixed-manifests',type=Path,nargs=10,required=True)
    a=p.parse_args();build(a.data_root,a.source_manifest,a.fixed_manifests,a.output)


if __name__=='__main__':main()
