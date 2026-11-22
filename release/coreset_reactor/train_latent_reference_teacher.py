"""Keep fixed mode MSE; marginalize the TRAIN-session reference for CCC.

No GT at inference and no change to the model, sampler or optimizer. The packed
26th target channel carries a TRAIN reference-group index only inside the loss.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

from coreset_reactor.teacher_cache import sha256_file
from regnn.conditional_data import ConditionalReactionDataset, relative_time_resample_reaction
from regnn.conditional_model import ConditionalREGNNLoss
from regnn import train_conditional_regnn as trainer

BANK = None


def reference_ccc(prediction, references):
    """Per prediction/reference mean-channel CCC; [B,T,25] x [N,T,25]."""
    if prediction.ndim!=3 or references.ndim!=3 or prediction.shape[1:]!=references.shape[1:]:
        raise ValueError('Expected matched temporal and channel shapes')
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        p, y = prediction.float(), references.float()
        pm, ym = p.mean(1), y.mean(1)
        pc, yc = p-pm[:,None], y-ym[:,None]
        pv, yv = pc.square().mean(1), yc.square().mean(1)
        covariance = torch.einsum('btd,ntd->bnd',pc,yc)/p.shape[1]
        ccc = 2*covariance/(pv[:,None]+yv[None]+(pm[:,None]-ym[None]).square()+1e-8)
        # Constant references have identically zero correlation. A constant
        # prediction against a varying reference has CCC=0 but a NONZERO valid
        # derivative through covariance; do not mask that learning signal.
        ccc = torch.where(yv[None]>0,ccc,torch.zeros_like(ccc))
        return ccc.mean(-1)


def latent_ccc(scores, temperature=.05):
    if not math.isfinite(temperature) or temperature<=0:
        raise ValueError('Temperature must be positive and finite')
    return temperature*(torch.logsumexp(scores/temperature,dim=-1)-math.log(scores.shape[-1]))


class ReferenceBank:
    def __init__(self,data_root):
        self.raw, self.groups, self.files = [], [], {}
        grouped=defaultdict(list)
        facial=data_root/'train/facial-attributes'
        for path in sorted(facial.glob('*/*/*.npy')):
            role,session,_=path.relative_to(facial).parts
            grouped[f'{role}/{session}'].append(path)
        for key,paths in sorted(grouped.items()):
            values=[]
            for path in paths:
                value=torch.from_numpy(np.load(path,allow_pickle=False).astype(np.float32))
                if value.ndim!=2 or value.shape[1]!=25 or not torch.isfinite(value).all():
                    raise ValueError(f'Invalid TRAIN reference {path}')
                values.append(value)
                self.files[path.relative_to(facial).as_posix()]=sha256_file(path)
            self.groups.append(key)
            self.raw.append(values)
        self.lookup={key:i for i,key in enumerate(self.groups)}
        self.cache=OrderedDict()

    def at(self,index,length,device):
        key=(index,length,str(device))
        if key not in self.cache:
            value=torch.stack([relative_time_resample_reaction(v,length) for v in self.raw[index]])
            self.cache[key]=value.to(device)
            # Each group typically <10MB; cap extra GPU target-cache storage.
            while len(self.cache)>48:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key]


class LatentReferenceDataset(ConditionalReactionDataset):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.split!='train' or self.fixed_pair_rank not in range(1,10):
            raise ValueError('TRAIN specialists only; reuse native paired q0')
        if self.fixed_pair_alignment_policy!='relative_time_masked' or BANK is None:
            raise ValueError('Requires frozen relative-time protocol and TRAIN bank')

    def __getitem__(self,index):
        value=super().__getitem__(index)
        role,session,_=value['clip_id'].split('/')
        opposite='listener' if role=='speaker' else 'speaker'
        group=BANK.lookup[f'{opposite}/{session}']
        target=value['target']
        value['target']=torch.cat((target,target.new_full((len(target),1),group)),dim=-1)
        return value


class LatentReferenceLoss(ConditionalREGNNLoss):
    def __init__(self,ccc_weight=1.,velocity_weight=.05,temperature=.05,bank=None,audit_every=100):
        super().__init__(ccc_weight,velocity_weight)
        self.temperature,self.bank,self.audit_every,self.calls=temperature,bank,audit_every,0

    def forward(self,prediction,target,valid_mask,velocity_supervision=None):
        if target.shape[-1]!=26:
            raise ValueError('Expected raw 25D target + reference-group tag')
        losses=super().forward(prediction,target[...,:25],valid_mask,velocity_supervision)
        indices=target[:,0,25].long()
        lengths=valid_mask.sum(1).long()
        expected=torch.arange(prediction.shape[1],device=prediction.device)[None]<lengths[:,None]
        if not torch.equal(valid_mask,expected) or (lengths<1).any():
            raise ValueError('Only nonempty valid prefixes are supported')
        scores=[]
        counts=[]
        for group,length in sorted(set(zip(indices.tolist(),lengths.tolist()))):
            choose=(indices==group)&(lengths==length)
            refs=self.bank.at(group,length,prediction.device)
            scores.append(latent_ccc(reference_ccc(prediction[choose,:length],refs),self.temperature))
            counts.extend([len(refs)]*int(choose.sum()))
        score=torch.cat(scores).mean()
        fixed_ccc=losses['ccc']
        losses['ccc']=score
        losses['ccc_loss']=1-score
        losses['loss']=losses['mse']+self.ccc_weight*losses['ccc_loss']+self.velocity_weight*losses['velocity']
        self.calls+=1
        if self.audit_every and self.calls%self.audit_every==0:
            print(json.dumps({'latent_reference_step':self.calls,'fixed_ccc':float(fixed_ccc.detach()),
                              'latent_ccc':float(score.detach()),'temperature':self.temperature,
                              'CCC_reference_count_mean':float(np.mean(counts)),
                              'CCC_reference_count_max':int(max(counts)),
                              'MSE_reference_count':1,'training_DTW_calls':0}),flush=True)
        return losses


def main():
    global BANK
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--reference-temperature',type=float,default=.05)
    known,remaining=parser.parse_known_args()
    original=trainer.parse_args

    def parse():
        global BANK
        args=original()
        if args.listener_3dmm_weight or args.supervision_tail_frames or args.online_rollout_blocks!=1:
            raise ValueError('Only frozen full-context 25-D pilot supported')
        if known.reference_temperature<=0 or not math.isfinite(known.reference_temperature):
            raise ValueError('Invalid temperature')
        BANK=ReferenceBank(args.data_dir)
        args.latent_reference_temperature=known.reference_temperature
        args.latent_reference_inputs_sha256=hashlib.sha256(json.dumps(BANK.files,sort_keys=True).encode()).hexdigest()
        args.latent_reference_source_sha256=sha256_file(Path(__file__))
        args.latent_reference_counts={key:len(BANK.raw[i]) for i,key in enumerate(BANK.groups)}
        args.latent_reference_protocol='raw TRAIN opposite-role all-session; relative-time resampling; fixed T1 MSE unchanged'
        return args

    def loss(**kwargs):
        return LatentReferenceLoss(**kwargs,temperature=known.reference_temperature,bank=BANK)

    trainer.parse_args=parse
    trainer.ConditionalReactionDataset=LatentReferenceDataset
    trainer.ConditionalREGNNLoss=loss
    sys.argv=[sys.argv[0],*remaining]
    trainer.main()


if __name__=='__main__':
    main()
