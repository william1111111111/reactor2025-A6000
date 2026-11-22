"""Mode-preserving reference marginalization: replace uniform by a TRAIN prior.

The prior is centered on the fixed teacher GT's time-order-invariant mean/std
descriptor. Its bandwidth has a fixed effective-reference count, not a VAL-fit
temperature. Everything else uses the latent-reference pilot's loss/protocol.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.special import logsumexp
import torch

from coreset_reactor import train_latent_reference_teacher as base
from coreset_reactor.teacher_cache import sha256_file
from regnn.conditional_model import ConditionalREGNNLoss


def entropy_matched_log_prior(cost, effective_count=8):
    """Row-wise Gibbs prior with entropy log(effective_count), when attainable.

Duplicate zero-distance modes can impose a larger minimal effective support;
in that case keep the symmetry instead of breaking ties arbitrarily.
"""
    cost=np.asarray(cost,dtype=np.float64)
    if cost.ndim!=2 or cost.shape[1]<1 or not np.isfinite(cost).all() or np.any(cost < -1e-8):
        raise ValueError('Expected nonnegative finite [anchors,references] costs')
    if not math.isfinite(effective_count) or effective_count<1:
        raise ValueError('Effective count must be >=1')
    target=math.log(min(effective_count,cost.shape[1]))
    cost=np.maximum(cost-cost.min(1,keepdims=True),0)
    scale=np.maximum(cost.mean(1,keepdims=True),1e-8)
    normalized=cost/scale
    lo=np.full(len(cost),-30.)
    hi=np.full(len(cost),30.)
    for _ in range(64):
        mid=(lo+hi)/2
        logits=-normalized/np.exp(mid[:,None])
        logp=logits-logsumexp(logits,axis=1,keepdims=True)
        entropy=-(np.exp(logp)*logp).sum(1)
        lo=np.where(entropy<target,mid,lo)
        hi=np.where(entropy>=target,mid,hi)
    logits=-normalized/np.exp(((lo+hi)/2)[:,None])
    return logits-logsumexp(logits,axis=1,keepdims=True)


def mode_latent_ccc(scores,log_prior,temperature=.05):
    if scores.shape!=log_prior.shape:
        raise ValueError('Reference scores and log prior must have identical shape')
    return temperature*torch.logsumexp(scores/temperature+log_prior,dim=-1)


class ModeReferenceBank(base.ReferenceBank):
    def __init__(self,data_root,effective_count=8):
        super().__init__(data_root)
        self.effective_count=effective_count
        values=torch.stack([torch.cat((v.mean(0),v.std(0,unbiased=False))) for group in self.raw for v in group])
        self.descriptor_scale=values.std(0,unbiased=False).clamp_min(.03)
        self.target_indices={}
        for group in self.groups:
            ids=sorted(k for k in self.files if k.startswith(group+'/'))
            self.target_indices.update({k:i for i,k in enumerate(ids)})
        self.priors=OrderedDict()

    def prior(self,group,length,device):
        key=(group,length,str(device))
        if key not in self.priors:
            refs=self.at(group,length,device)
            d=torch.cat((refs.mean(1),refs.std(1,unbiased=False)),dim=-1).cpu()
            d=d/self.descriptor_scale
            cost=((d[:,None]-d[None])**2).mean(-1).numpy()
            logp=entropy_matched_log_prior(cost,self.effective_count)
            self.priors[key]=torch.from_numpy(logp.astype(np.float32)).to(device)
            while len(self.priors)>48: self.priors.popitem(last=False)
        self.priors.move_to_end(key)
        return self.priors[key]


class ModeReferenceDataset(base.LatentReferenceDataset):
    def __getitem__(self,index):
        result=super().__getitem__(index)
        source=result['clip_id']+'.npy'
        target_id=self.fixed_pair_mapping[source][self.fixed_pair_rank]
        chosen=base.BANK.target_indices[target_id]
        value=result['target']
        result['target']=torch.cat((value,value.new_full((len(value),1),chosen)),dim=-1)
        return result


class ModeReferenceLoss(ConditionalREGNNLoss):
    def __init__(self,ccc_weight=1.,velocity_weight=.05,temperature=.05,bank=None,audit_every=100):
        super().__init__(ccc_weight,velocity_weight)
        self.temperature,self.bank,self.audit_every,self.calls=temperature,bank,audit_every,0

    def forward(self,prediction,target,valid_mask,velocity_supervision=None):
        if target.shape[-1]!=27: raise ValueError('Expected raw target + group + anchor ID')
        losses=super().forward(prediction,target[...,:25],valid_mask,velocity_supervision)
        indices=target[:,0,25].long()
        anchors=target[:,0,26].long()
        lengths=valid_mask.sum(1).long()
        expected=torch.arange(prediction.shape[1],device=prediction.device)[None]<lengths[:,None]
        if not torch.equal(valid_mask,expected) or (lengths<1).any(): raise ValueError('Invalid prefix')
        scores,effs,counts=[],[],[]
        audit=self.audit_every and (self.calls+1)%self.audit_every==0
        for group,length in sorted(set(zip(indices.tolist(),lengths.tolist()))):
            choose=(indices==group)&(lengths==length)
            refs=self.bank.at(group,length,prediction.device)
            logp=self.bank.prior(group,length,prediction.device)[anchors[choose]]
            scores.append(mode_latent_ccc(base.reference_ccc(prediction[choose,:length],refs),logp,self.temperature))
            if audit:
                effs.extend(torch.exp(-(logp.exp()*logp).sum(-1)).tolist())
                counts.extend([len(refs)]*int(choose.sum()))
                # The fixed MSE target must be exactly the prior's anchor.
                torch.testing.assert_close(refs[anchors[choose]],target[choose,:length,:25],rtol=0,atol=0)
        score=torch.cat(scores).mean()
        fixed=losses['ccc']
        losses['ccc']=score
        losses['ccc_loss']=1-score
        losses['loss']=losses['mse']+self.ccc_weight*losses['ccc_loss']+self.velocity_weight*losses['velocity']
        self.calls+=1
        if audit:
            print(json.dumps({'mode_reference_step':self.calls,'fixed_ccc':float(fixed.detach()),
                              'mode_latent_ccc':float(score.detach()),'effective_reference_prior_mean':float(np.mean(effs)),
                              'all_CCC_reference_count_mean':float(np.mean(counts)),'MSE_targets':1}),flush=True)
        return losses


def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--reference-temperature',type=float,default=.05)
    p.add_argument('--reference-effective-count',type=float,default=8)
    known,remaining=p.parse_known_args()
    original=base.trainer.parse_args
    def parse():
        args=original()
        if args.listener_3dmm_weight or args.supervision_tail_frames or args.online_rollout_blocks!=1:
            raise ValueError('Only frozen full-context 25-D protocol')
        if not math.isfinite(known.reference_temperature) or known.reference_temperature<=0:
            raise ValueError('Invalid temperature')
        if not math.isfinite(known.reference_effective_count) or known.reference_effective_count<1:
            raise ValueError('Invalid effective reference count')
        base.BANK=ModeReferenceBank(args.data_dir,known.reference_effective_count)
        args.latent_reference_temperature=known.reference_temperature
        args.reference_effective_count=known.reference_effective_count
        args.mode_reference_source_sha256=sha256_file(Path(__file__))
        args.latent_reference_source_sha256=sha256_file(Path(base.__file__))
        args.reference_inputs_sha256=hashlib.sha256(json.dumps(base.BANK.files,sort_keys=True).encode()).hexdigest()
        args.reference_descriptor_scale=base.BANK.descriptor_scale.tolist()
        args.reference_prior='TRAIN-only mean/std 50D; entropy-matched Gibbs kernel around unchanged fixed target; effective count8'
        return args
    base.trainer.parse_args=parse
    base.trainer.ConditionalReactionDataset=ModeReferenceDataset
    base.trainer.ConditionalREGNNLoss=lambda **kwargs: ModeReferenceLoss(**kwargs,temperature=known.reference_temperature,bank=base.BANK)
    sys.argv=[sys.argv[0],*remaining]
    base.trainer.main()


if __name__=='__main__': main()
