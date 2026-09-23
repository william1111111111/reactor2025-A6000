"""Full VAL checkpoint evaluation using FRC only."""
import json
import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from coreset_reactor.anchor_pair_ensemble import load_models, ensemble_forward
from coreset_reactor.evaluate import official_prediction, _stable_rng, _deterministic_postprocess
from coreset_reactor.paired_data import PairedReactionDataset
from coreset_reactor.fast_frc import candidate_frc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:5')
    parser.add_argument('--epochs', type=int, nargs='+', default=[10,20,30,40,50])
    parser.add_argument('--all-session-gt', action='store_true')
    parser.add_argument('--run', type=Path)
    args = parser.parse_args()
    assets = Path('/home/zhaowenjie/react2025_new/mam_reactor')
    sys.path.append(str(assets))
    from framework.modules.post_processor import Processor
    from framework.metrics.FRC import _func
    torch.set_num_threads(4)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    root = Path(__file__).resolve().parents[1]
    run = root / 'coreset_reactor/reports/anchor_behavioral_pairs_full50_seed1'
    if args.run is not None:
        run = args.run.resolve()
    data_root = Path('/public/zhaowenjie/react2025_new/data')
    data = PairedReactionDataset(data_root, 'val', 750, crop_mode='center',
                                normalization_dir=assets / 'external/FaceVerse')
    selected = list(range(len(data)))
    facial = data_root / 'val/facial-attributes'
    pool = {}
    for path in sorted((facial / 'listener').glob('*/*.npy')):
        pool.setdefault(path.parent.name, []).append(path)
    processor = Processor(cfg_dir=str(assets),
                          ckpt_dir=str(assets / 'pretrained_models/post_processor'),
                          clip_len_test=1000, device=device, num_preds=10)
    result = {'contexts': len(selected), 'eval_seed': 1234, 'protocol': 'center750; paired+9; official FRC; no FRD', 'epochs': {}}
    if args.all_session_gt:
        result['protocol'] = 'full VAL571 speaker sources; center750; all opposite-role session GT; official FRC; no FRD'
    target_cache = {}
    with torch.no_grad():
        for epoch in args.epochs:
            started = time.monotonic()
            models, _ = load_models(run, epoch, device)
            rows = []
            for number, index in enumerate(selected, 1):
                sample = data[index]
                length = int(sample['source_lengths'])
                inputs = tuple(sample[key][None, :length].to(device) for key in
                               ('speaker_audio', 'speaker_emotion', 'speaker_3dmm'))
                pred = official_prediction(ensemble_forward(models, inputs,
                                            torch.tensor([length], device=device), device))
                paired = (facial / sample['clip_id'].replace('speaker/', 'listener/', 1)).with_suffix('.npy')
                alternatives = [p for p in pool[sample['session_id']] if p != paired]
                rng = _stable_rng(1234, sample['clip_id'])
                paths = [paired] + (alternatives if args.all_session_gt else rng.sample(alternatives, 9))
                key = (sample['clip_id'], length, tuple(map(str, paths)))
                if key not in target_cache:
                    raw = [torch.from_numpy(np.load(p).astype(np.float32)) for p in paths]
                    target_cache[key] = _deterministic_postprocess(processor, pred, raw, device, 1234, sample['clip_id']).cpu()
                target = target_cache[key]
                values = candidate_frc(target.numpy(), pred.cpu().numpy()).tolist()
                if number == 1:
                    reference = [float(_func(target, pred[k:k+1].cpu())) for k in range(10)]
                    np.testing.assert_allclose(values, reference, rtol=1e-5, atol=1e-6)
                    print(f'epoch={epoch} reference parity max_error={np.max(np.abs(np.array(values)-reference)):.3g}', flush=True)
                if not np.isfinite(values).all():
                    raise ValueError(f'Nonfinite FRC: {sample["clip_id"]}')
                rows.append({'clip_id': sample['clip_id'], 'gt_count': len(paths), 'frc': values})
                if number % 25 == 0:
                    print(f'epoch={epoch} contexts={number}/{len(selected)}', flush=True)
            means = np.mean([row['frc'] for row in rows], axis=0)
            result['epochs'][str(epoch)] = {'per_rank': means.tolist(), 'ensemble': float(means.sum()), 'seconds': time.monotonic()-started, 'rows': rows}
            prefix = 'frc_all_session_full_val' if args.all_session_gt else 'frc_curve_full_val'
            (run / f'{prefix}_epoch{epoch}.json').write_text(json.dumps(result, indent=2, allow_nan=False))
            print(f'epoch={epoch} FRC={means.sum():.6f} per_rank={means.tolist()}', flush=True)
            del models
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
