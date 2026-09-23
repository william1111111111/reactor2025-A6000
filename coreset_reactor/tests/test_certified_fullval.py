import copy
import json
from pathlib import Path

import pytest
import torch

from coreset_reactor.run_certified_fullval import merge_shard_records
from coreset_reactor.run_certified_oracle import validate_reuse_config,validate_cached_record,build_pool
from coreset_reactor.certified_select import solve_select
import numpy as np


def config():
    return {'checkpoint_sources':[],'source_hashes':{'numerical.py':'frozen'},
            'eval_seed':1234,'metric_workers':8,'mip_seconds':5.,'baseline':'basic',
            'primary_pool':'raw30','analysis_pool':'expanded60','quality_policy':'strict',
            'population_contexts':571,'reuse_dir':'pilot'}


def test_reuse_allows_orchestration_not_numerical_changes():
    old=config(); new=copy.deepcopy(old)
    old['source_hashes']['coreset_reactor/run_certified_oracle.py']='old'
    new['source_hashes']['coreset_reactor/run_certified_oracle.py']='new'
    validate_reuse_config(old,new)
    new['source_hashes']['numerical.py']='changed'
    with pytest.raises(ValueError):
        validate_reuse_config(old,new)


def test_shard_merge_rejects_overlap_and_mismatched_contract():
    def shard(indices):
        return {'config':{**config(),'indices':indices,'clip_ids':[str(i) for i in indices]},
                'per_context':[{'val_index':i,'clip_id':str(i)} for i in indices]}
    a,b=shard([0,1]),shard([2,3])
    assert [r['val_index'] for r in merge_shard_records([b,a])]==[0,1,2,3]
    with pytest.raises(ValueError):
        merge_shard_records([a,a])
    b['config']['quality_policy']='slack'
    with pytest.raises(ValueError):
        merge_shard_records([a,b])


def test_reuse_rechecks_feasibility_and_upper_bound():
    torch.manual_seed(17)
    x=torch.rand(10,4,25)
    p,origins,raw_count,_=build_pool([x],['basic'])
    frc=np.ones(len(p)); frd=np.ones(len(p))
    record={'candidate_origins':origins}
    for name,size in (('raw30',raw_count),('expanded60',len(p))):
        vectors=p[:size].flatten(1).double().numpy()/np.sqrt(100)
        record[name]=solve_select(vectors,frc[:size],frd[:size],list(range(10)),mip_seconds=0)
        record[name]['pool_size']=size
    validate_cached_record(record,p,origins,raw_count,frc,frd)
    corrupt=copy.deepcopy(record)
    corrupt['expanded60']['upper_bound']+=.1
    with pytest.raises(AssertionError):
        validate_cached_record(corrupt,p,origins,raw_count,frc,frd)
