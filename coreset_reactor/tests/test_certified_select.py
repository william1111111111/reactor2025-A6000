import itertools
import json
import numpy as np
import pytest

from coreset_reactor.certified_select import (geometry,extension,selection_value,
    quality_feasible,solve_select,tangent_certificate)


def test_squared_distance_concave_identity_and_gradient():
    rng=np.random.default_rng(29)
    points=rng.normal(size=(9,6)); k=3
    gram,norms,d=geometry(points)
    z=np.full(9,k/9)
    value,g=extension(z,gram,norms,k)
    assert value==pytest.approx(z@d@z/(k*(k-1)),abs=1e-12)
    direction=rng.normal(size=9); direction-=direction.mean()
    difference=(extension(z+1e-5*direction,gram,norms,k)[0]-extension(z-1e-5*direction,gram,norms,k)[0])/2e-5
    assert difference==pytest.approx(g@direction,abs=1e-9)
    for subset in itertools.combinations(range(9),k):
        indicator=np.zeros(9); indicator[list(subset)]=1
        assert extension(indicator,gram,norms,k)[0]==pytest.approx(selection_value(d,list(subset)))


@pytest.mark.parametrize('seed',range(5))
def test_bounds_bracket_bruteforce_optimum(seed):
    rng=np.random.default_rng(seed)
    points=rng.normal(size=(8,5))
    frc=rng.uniform(.05,.2,8); frd=rng.uniform(1,5,8); baseline=[0,1,2]
    _,_,d=geometry(points)
    exact=max(selection_value(d,list(s)) for s in itertools.combinations(range(8),3)
              if quality_feasible(list(s),frc,frd,baseline))
    result=solve_select(points,frc,frd,baseline,mip_seconds=2)
    json.dumps(result,allow_nan=False)
    assert result['lower_bound']<=exact+1e-9
    assert result['upper_bound']>=exact-1e-9
    assert quality_feasible(result['selected'],frc,frd,baseline)
    assert result['lower_bound']==pytest.approx(exact,abs=1e-7)


def test_bounds_valid_from_infeasible_tangent_and_degenerate_pool():
    rng=np.random.default_rng(99)
    p=rng.normal(size=(7,3)); k=3
    gram,norms,d=geometry(p)
    frc=rng.random(7); frd=rng.random(7); base=[0,1,2]
    matrix=np.stack([-frc,frd]); bound=np.array([-frc[base].sum(),frd[base].sum()])
    cert=tangent_certificate(rng.normal(size=7),gram,norms,k,matrix,bound)
    exact=max(selection_value(d,list(s)) for s in itertools.combinations(range(7),k)
              if quality_feasible(list(s),frc,frd,base))
    assert cert['upper']>=exact
    result=solve_select(np.zeros((7,3)),np.ones(7),np.ones(7),base,mip_seconds=0)
    assert result['lower_bound']==0 and result['upper_bound']<1e-6


def test_infeasible_high_diversity_candidates_are_not_admitted():
    p=np.array([[0],[1],[2],[100]],dtype=float)
    result=solve_select(p,np.array([1,1,1,-20]),np.array([1,1,1,50]),[0,1,2],mip_seconds=2)
    assert result['selected']==[0,1,2]
