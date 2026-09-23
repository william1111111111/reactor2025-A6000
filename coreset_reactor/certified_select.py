"""Finite-pool, fixed-K FRDiv optimization with independently checked bounds.

The concave extension and LP-dual tangent certificate are valid specifically
for squared Euclidean distances. Numeric certificates are float64, with an
explicit safety allowance; this is not an interval-arithmetic proof system.
"""
from __future__ import annotations

import itertools
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp, minimize
from scipy.sparse import coo_matrix


def geometry(points):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or not np.isfinite(points).all():
        raise ValueError('Need a finite [candidates, features] matrix')
    # Translation leaves distances unchanged and improves conditioning.
    centered = points-points.mean(0, keepdims=True)
    gram = centered@centered.T
    norms = np.diag(gram).copy()
    distance = norms[:, None]+norms[None, :]-2*gram
    if distance.min() < -1e-10:
        raise ValueError('Numerically invalid squared-distance matrix')
    distance = np.maximum(distance, 0)
    np.fill_diagonal(distance, 0)
    return gram, norms, distance


def extension(z, gram, norms, count):
    denominator = count*(count-1)
    value = (2*count*(norms@z)-2*(z@gram@z))/denominator
    gradient = (2*count*norms-4*(gram@z))/denominator
    return float(value), gradient


def selection_value(distance, selected):
    k = len(selected)
    return float(distance[np.ix_(selected, selected)].sum()/(k*(k-1)))


def quality_feasible(selected, frc, frd, baseline, frc_atol=1e-10, frd_atol=1e-8):
    return bool(len(selected)==len(baseline) and len(set(selected))==len(selected)
            and np.all(np.array(selected)>=0) and np.all(np.array(selected)<len(frc))
            and frc[selected].sum() >= frc[baseline].sum()-frc_atol
            and frd[selected].sum() <= frd[baseline].sum()+frd_atol)


def tangent_certificate(z, gram, norms, count, matrix, bound, allowance=1e-8):
    """Upper bound from concavity plus a box-supported LP Lagrange dual.

    For ANY lambda>=0, nu, the affine maximization over Az<=b, sum(z)=K,
    0<=z<=1 is bounded by lambda.b + nu.K + sum(max(g-A.T lambda-nu,0)).
    Thus no LP primal iterate is ever mistaken for an upper bound.
    """
    value, gradient = extension(z, gram, norms, count)
    intercept = value-gradient@z
    # Always-valid cardinality-only support, even if LP fails.
    candidates = [{'kind':'cardinality_tangent',
                   'raw_upper':float(intercept+np.sort(gradient)[-count:].sum())}]
    lp = linprog(-gradient, A_ub=matrix, b_ub=bound,
                 A_eq=np.ones((1,len(z))), b_eq=[count], bounds=(0,1), method='highs')
    if lp.success:
        lam = np.maximum(0, -np.asarray(lp.ineqlin.marginals))
        nu = -float(lp.eqlin.marginals[0])
        residual = gradient-matrix.T@lam-nu
        raw = intercept+lam@bound+nu*count+np.maximum(residual,0).sum()
        if np.isfinite(raw):
            candidates.append({'kind':'quality_lp_dual_tangent', 'raw_upper':float(raw),
                               'lambda':lam.tolist(), 'nu':nu,
                               'box_support':float(np.maximum(residual,0).sum()),
                               'lp_primal_support':float(-lp.fun)})
    best = min(candidates, key=lambda x:x['raw_upper'])
    numerical = allowance*max(1.,abs(best['raw_upper']))
    return {**best, 'upper':best['raw_upper']+numerical,
            'numeric_allowance':numerical, 'tangent_point':z.tolist(),
            'tangent_value':value, 'lp_success':bool(lp.success)}


def improve_swaps(distance, frc, frd, baseline):
    selected = list(baseline)
    count = len(selected)
    for _ in range(100):
        current = selection_value(distance, selected)
        best = None
        outside = [i for i in range(len(frc)) if i not in selected]
        for old in selected:
            for new in outside:
                trial = sorted([i for i in selected if i!=old]+[new])
                if quality_feasible(trial,frc,frd,baseline):
                    score = selection_value(distance,trial)
                    if score > current+1e-12 and (best is None or score > best[0]+1e-12):
                        best = score, trial
        if best is not None:
            selected = best[1]
            continue
        # Bounded two-swap is a lower-bound heuristic, not an optimality claim.
        outside = sorted(outside, key=lambda i:(-distance[i,selected].sum(),i))[:16]
        for old in itertools.combinations(selected,2):
            for new in itertools.combinations(outside,2):
                trial = sorted([i for i in selected if i not in old]+list(new))
                if quality_feasible(trial,frc,frd,baseline):
                    score = selection_value(distance,trial)
                    if score > current+1e-12 and (best is None or score > best[0]+1e-12):
                        best = score,trial
        if best is None:
            break
        selected = best[1]
    assert len(selected)==count
    return selected


def integer_search(distance, matrix, bound, count, seconds):
    """Time-limited exact MILP formulation. Only validated incumbents are used."""
    n = len(distance)
    pairs = list(itertools.combinations(range(n),2))
    width = n+len(pairs)
    objective = np.zeros(width)
    objective[n:] = [-2*distance[i,j]/(count*(count-1)) for i,j in pairs]
    rr,cc,vv = [],[],[]
    low,high = [count]+[-np.inf]*len(bound), [count]+bound.tolist()
    for col in range(n):
        rr.append(0); cc.append(col); vv.append(1.)
    for row in range(len(bound)):
        for col in range(n):
            rr.append(row+1); cc.append(col); vv.append(matrix[row,col])
    row = len(low)
    for offset,(i,j) in enumerate(pairs):
        y = n+offset
        for columns, values, upper in (((y,i),(1.,-1.),0.),
                                        ((y,j),(1.,-1.),0.),
                                        ((i,j,y),(1.,1.,-1.),1.)):
            for col,val in zip(columns,values):
                rr.append(row); cc.append(col); vv.append(val)
            low.append(-np.inf); high.append(upper); row+=1
    constraints = LinearConstraint(coo_matrix((vv,(rr,cc)),shape=(row,width)).tocsc(),low,high)
    result = milp(objective, integrality=np.r_[np.ones(n),np.zeros(len(pairs))],
                  bounds=Bounds(0,1), constraints=constraints,
                  options={'time_limit':seconds,'mip_rel_gap':1e-6})
    selected = None
    if result.x is not None and np.max(np.abs(result.x[:n]-np.round(result.x[:n]))) < 1e-6:
        selected = np.flatnonzero(result.x[:n]>.5).tolist()
    dual = getattr(result,'mip_dual_bound',None)
    return selected, {'status':int(result.status), 'message':str(result.message),
                      'seconds_limit':seconds,
                      'solver_dual_bound':float(dual) if dual is not None and np.isfinite(dual) else None,
                      'note':'MIP upper bound is logged only; reported certificate uses the independent tangent bound'}


def solve_select(points, frc, frd, baseline, mip_seconds=5.):
    started = time.perf_counter()
    frc,frd = np.asarray(frc,dtype=np.float64),np.asarray(frd,dtype=np.float64)
    baseline = list(map(int,baseline))
    gram,norms,distance = geometry(points)
    n,k = len(points),len(baseline)
    if k<2 or k>n or len(set(baseline))!=k or min(baseline)<0 or max(baseline)>=n:
        raise ValueError('Invalid fixed-K baseline')
    if frc.shape!=(n,) or frd.shape!=(n,) or not np.isfinite(frc).all() or not np.isfinite(frd).all():
        raise ValueError('Invalid quality contributions')
    thresholds = [float(frc[baseline].sum()),float(frd[baseline].sum())]
    normalization = np.maximum(1,np.abs(thresholds))
    matrix = np.stack([-frc,frd])/normalization[:,None]
    bound = np.array([-thresholds[0],thresholds[1]])/normalization
    initial = np.zeros(n); initial[baseline]=1
    optimized = minimize(lambda z:-extension(z,gram,norms,k)[0],initial,
                         jac=lambda z:-extension(z,gram,norms,k)[1], method='SLSQP',
                         bounds=Bounds(0,1), constraints=[
                             LinearConstraint(np.ones((1,n)),k,k),
                             LinearConstraint(matrix,-np.inf,bound)],
                         options={'ftol':1e-12,'maxiter':1000})
    candidates = [initial]
    if np.isfinite(optimized.x).all():
        candidates.append(optimized.x)
    certificates = [tangent_certificate(z,gram,norms,k,matrix,bound) for z in candidates]
    certificate = min(certificates,key=lambda x:x['upper'])
    selected = improve_swaps(distance,frc,frd,baseline)
    heuristic = list(selected)
    integer = {'status':'disabled'}
    if mip_seconds>0:
        trial,integer = integer_search(distance,matrix,bound,k,mip_seconds)
        integer['incumbent_quality_valid'] = trial is not None and quality_feasible(trial,frc,frd,baseline)
        if integer['incumbent_quality_valid'] and selection_value(distance,trial)>selection_value(distance,selected):
            selected = trial
    if not quality_feasible(selected,frc,frd,baseline):
        raise RuntimeError('Final integer solution violates strict quality')
    lower = selection_value(distance,selected)
    if certificate['upper'] < lower-1e-10:
        raise RuntimeError('Upper bound below feasible lower bound')
    return {'selected':selected,'heuristic_selected':heuristic,
            'baseline':{'FRC':thresholds[0],'exact_FRD':thresholds[1],
                        'FRDiv':selection_value(distance,baseline)},
            'selected_metrics':{'FRC':float(frc[selected].sum()),'exact_FRD':float(frd[selected].sum()),'FRDiv':lower},
            'lower_bound':lower,'upper_bound':certificate['upper'],
            'absolute_gap':certificate['upper']-lower,
            'relative_gap_to_upper':(certificate['upper']-lower)/max(certificate['upper'],1e-12),
            'certificate':certificate, 'relaxation':{'success':bool(optimized.success),
                'message':str(optimized.message),'iterations':int(optimized.nit),
                'primal_value_not_an_upper_bound':float(-optimized.fun),
                'max_violation':float(max(0,np.max(matrix@optimized.x-bound),abs(optimized.x.sum()-k),
                                           -optimized.x.min(),optimized.x.max()-1))},
            'integer_search':integer,'seconds':time.perf_counter()-started,
            'quality_policy':'zero policy slack; float64 check atol FRC=1e-10, FRD=1e-8',
            'bound_scope':'this finite pool, fixed K and this context/reference GT only',
            'bound_arithmetic':'analytic tangent/LP dual bound, float64 plus explicit allowance; not interval arithmetic'}
