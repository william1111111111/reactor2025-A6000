"""Complete one matched direction-KD trial; never launch further training."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from coreset_reactor.run_relation_evaluation import evaluate, ROOT, RUNS, BASELINE, RELATION
from coreset_reactor.teacher_cache import sha256_file

DIRECTION = RUNS/'t10_student_direction_pilot10e'


def write_comparison(new, complete):
    paths = [BASELINE/'all_session_val571_cached.json', RELATION/'all_session_val571_full.json']
    values = [json.loads(p.read_text()) for p in paths]+[new]
    legacy_path = BASELINE/'all_session_val571_full.json'
    legacy = json.loads(legacy_path.read_text())
    if legacy['checkpoint'] != values[0]['checkpoint'] or [r['clip_id'] for r in legacy['rows']] != [r['clip_id'] for r in values[0]['rows']]:
        raise ValueError('Legacy baseline FRD checkpoint/context mismatch')
    for key in ('FRC','FRDiv','FRVar'):
        np.testing.assert_allclose(legacy['metrics'][key],values[0]['metrics'][key],rtol=0,atol=1e-4)
    np.testing.assert_allclose([r['frc'] for r in legacy['rows']],
                               [r['frc'] for r in values[0]['rows']],rtol=0,atol=1e-4)
    values[0]['metrics']['exact_FRD'] = legacy['metrics']['exact_FRD']
    configs = [json.loads((p/'config.json').read_text()) for p in (BASELINE, RELATION, DIRECTION)]
    for value in values[1:]:
        for key in ('clip_id','target_cache_key'):
            if [r[key] for r in value['rows']] != [r[key] for r in values[0]['rows']]:
                raise ValueError(f'Unmatched evaluation {key}')
    for key in ('epochs','batch_size','workers','lr','seed','expert_embedding_dim',
                'cache_manifest','initialize_backbone','parameter_count'):
        if any(c[key] != configs[0][key] for c in configs[1:]):
            raise ValueError(f'Training mismatch: {key}')
    logs = [json.loads((p/'train_metrics.json').read_text()) for p in (BASELINE, RELATION, DIRECTION)]
    first_kd = [h[0].get('kd_loss', h[0]['loss']) for h in logs]
    np.testing.assert_allclose(first_kd, first_kd[0], rtol=0, atol=0)
    if any(h[-1]['step'] != 8300 for h in logs):
        raise ValueError('Expected 8300 matched updates')
    names = ['Basic KD','Distance KD','Centered-direction KD']
    result = {'scope':new['protocol'], 'models':{}, 'exact_FRD_complete':complete,
              'matched_initial_kd_loss':first_kd,
              'basic_exact_FRD_source': {'path':str(legacy_path), 'sha256':sha256_file(legacy_path),
                  'caveat':'Earlier full FRD lacks GT cache fingerprints; same checkpoint/path/context and per-slot FRC parity checked'},
              'limitations':['single seed', 'legacy prototype teacher cache', 'VAL571, not official TEST',
                             'direction-only residual supervision is a hypothesis, not a novelty claim']}
    lines = ['# Matched direction-versus-distance distillation', '', new['protocol'], '',
             '| Model | FRC | exact FRD | FRDiv | FRVar | Train seconds | Parameters |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name, value, config in zip(names, values, configs):
        m = value['metrics']
        result['models'][name] = {'metrics':m, 'checkpoint_sha256':value['checkpoint_sha256'],
                                 'config':config, 'speed':value['speed']}
        frd = m['exact_FRD']
        frd_text = f'{frd:.6f}' if isinstance(frd,(int,float)) else 'pending'
        seconds = config.get('training_seconds')
        seconds_text = f'{seconds:.1f}' if isinstance(seconds, (int, float)) else 'not recorded'
        lines.append(f"| {name} | {m['FRC']:.6f} | {frd_text} | {m['FRDiv']:.6f} | "
                     f"{m['FRVar']:.6f} | {seconds_text} | {config['parameter_count']} |")
    for ref in values[:2]:
        label = 'basic' if ref is values[0] else 'distance'
        result[f'delta_vs_{label}'] = {}
        for key in ('FRC','FRDiv'):
            def rowval(r):
                return sum(r['frc']) if key=='FRC' else r[key]
            delta = np.array([rowval(n)-rowval(b) for n,b in zip(new['rows'],ref['rows'])])
            result[f'delta_vs_{label}'][key] = {'mean':float(delta.mean()),
                'context_delta_quantiles':np.quantile(delta,[0,.1,.5,.9,1]).tolist()}
    lines += ['', 'Same backbone, seed=1, initialization, cache, source schedule, batch4, 10 epochs / 8300 updates.',
              'First-batch KD loss matches exactly across arms. Auxiliary gradients calibrated on TRAIN only.',
              'No added parameters, teacher calls or expensive exact temporal-GT computations during training.',
              'Historical teacher label alignment is unchanged; this isolates the student objective only.',
              'Basic-KD FRD is reused from its earlier full evaluation (no cache fingerprints); checkpoint path, context and per-slot FRC parity checked.',
              'Report full quality-diversity curves before attributing a gain to mode preservation.',
              'Measured evaluation inference timing is diagnostic, not a controlled efficiency benchmark.']
    (DIRECTION/'matched_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    (ROOT/'reports/student_direction_comparison.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v['metrics'] for k,v in result['models'].items()},indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:5')
    args = parser.parse_args()
    deadline = time.monotonic()+3*3600
    while not (DIRECTION/'train_metrics.json').exists():
        if time.monotonic()>deadline:
            raise TimeoutError('Direction training not finished after three hours')
        time.sleep(20)
    # Configuration is finalized immediately after training metrics are saved.
    while 'training_seconds' not in json.loads((DIRECTION/'config.json').read_text()):
        if time.monotonic()>deadline:
            raise TimeoutError('Training config not finalized')
        time.sleep(1)
    write_comparison(evaluate(DIRECTION, 'all_session_val571_fast.json', args.device), False)
    write_comparison(evaluate(DIRECTION, 'all_session_val571_full.json', args.device, exact=True), True)


if __name__ == '__main__':
    main()
