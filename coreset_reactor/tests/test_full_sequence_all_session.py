import numpy as np
import pytest
import torch

from coreset_reactor.eval_full_sequence_all_session import full_student


class Echo:
    def generate(self,audio,emotion,face,lengths):
        assert (lengths>0).all()
        return {'prediction':emotion[:,None].expand(-1,10,-1,-1)}


@pytest.mark.parametrize('length',[1,749,750,751,1500,1501])
def test_full_sequence_preserves_every_frame_and_slot(length):
    emotion=torch.arange(length).float()[:,None].expand(-1,25)
    result=full_student(Echo(),[emotion,emotion,emotion],torch.device('cpu'),batch_size=1)
    assert result.shape==(10,length,25)
    torch.testing.assert_close(result,emotion[None].expand(10,-1,-1))


@pytest.mark.parametrize('n,m',[(1,1),(3,7),(19,13),(64,64)])
def test_cuda_dtw_matches_official_tslearn(n,m):
    from numba import cuda
    if not cuda.is_available():pytest.skip('GPU unavailable')
    from tslearn.metrics import dtw
    from coreset_reactor.exact_dtw_cuda import pairwise_frd
    rng=np.random.default_rng(123)
    pred=rng.normal(size=(2,n,25)).astype('float32')
    target=rng.normal(size=(3,m,25)).astype('float32')
    pred[:,:,0]=target[:,:,0]=0
    result=pairwise_frd(pred,target)
    expected=np.array([[sum(w*dtw(p[:,a:b],t[:,a:b])
        for a,b,w in [(0,15,1/15),(15,17,1),(17,25,1/8)]) for t in target] for p in pred])
    np.testing.assert_allclose(result,expected,rtol=1e-12,atol=1e-10)
