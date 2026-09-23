"""Build TRAIN E60 candidate-quality labels for the inference-safe selector."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from coreset_reactor.evaluate import _deterministic_postprocess, official_prediction
from coreset_reactor.oracle_select import Candidate, SharedQualityPool
from coreset_reactor.paired_data import PairedReactionDataset
from coreset_reactor.run_certified_oracle import build_pool
from coreset_reactor.teacher_cache import sha256_file

ROOT = Path(__file__).resolve().parents[1]
MAM_ROOT = ROOT / 'mam_reactor'
RUNS = ROOT / 'coreset_reactor/reports/fixed_teacher_distillation'
STUDENTS = (
    ('basic', RUNS/'t10_student_pilot10e/student.pth'),
    ('distance', RUNS/'t10_student_rel_pilot10e/student.pth'),
    ('direction', RUNS/'t10_student_direction_pilot10e/student.pth'),
)


def load_student(path: Path, device: torch.device):
    from regnn.conditional_model import ConditionalREGNNConfig
    from regnn.expert_student import ExpertConditionalREGNN, ExpertStudentConfig
    state = torch.load(path, map_location='cpu', weights_only=False)
    cfg = state['student_config']
    model = ExpertConditionalREGNN(ExpertStudentConfig(
        backbone=ConditionalREGNNConfig(**cfg['backbone']),
        num_experts=cfg['num_experts'],
        expert_embedding_dim=cfg['expert_embedding_dim']))
    model.load_state_dict(state['state_dict'], strict=True)
    return model.to(device).eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=Path('/public/zhaowenjie/react2025_new/data'))
    parser.add_argument('--asset-mam-root', type=Path, default=Path('/home/zhaowenjie/react2025_new/mam_reactor'))
    parser.add_argument('--device', default='cuda:5')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--data-workers', type=int, default=4)
    parser.add_argument('--metric-workers', type=int, default=32)
    parser.add_argument('--context-start', type=int, default=0,
                        help='inclusive TRAIN dataset index for this shard')
    parser.add_argument('--context-stop', type=int, default=None,
                        help='exclusive TRAIN dataset index for this shard')
    parser.add_argument('--max-contexts', type=int, default=0)
    parser.add_argument('--eval-seed', type=int, default=1234)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    torch.set_num_threads(2)
    device = torch.device(args.device)
    if device.type != 'cuda':
        raise ValueError('TRAIN label generation requires a CUDA device')
    torch.cuda.set_device(device)

    dataset = PairedReactionDataset(args.data_root, 'train', 750, crop_mode='center',
                                    normalization_dir=args.asset_mam_root/'external/FaceVerse')
    dataset_total = len(dataset)
    start = max(0, int(args.context_start))
    stop = dataset_total if args.context_stop is None else min(int(args.context_stop), dataset_total)
    if args.max_contexts > 0:
        stop = min(stop, start + int(args.max_contexts))
    if not 0 <= start < stop <= dataset_total:
        raise ValueError(f'invalid TRAIN shard [{start}, {stop}) of {dataset_total}')
    total = stop - start
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.data_workers, pin_memory=True)
    pools = defaultdict(list)
    facial = args.data_root/'train/facial-attributes'
    for path in sorted((facial/'listener').glob('*/*.npy')):
        pools[path.parent.name].append(path)
    if len(pools) < 1 or max(map(len, pools.values())) < 90:
        raise ValueError('TRAIN listener session pools are incomplete')
    models = [load_student(path, device) for _, path in STUDENTS]
    if any(m.config.num_experts != 10 for m in models):
        raise ValueError('Expected ten outputs per student')
    from framework.modules.post_processor import Processor
    processor = Processor(cfg_dir=str(args.asset_mam_root),
                          ckpt_dir=str(args.asset_mam_root/'pretrained_models/post_processor'),
                          device=device, clip_len_test=1000, num_preds=10)
    max_targets = max(map(len, pools.values()))
    quality_pool = SharedQualityPool(args.metric_workers, 60, 750, max_targets=max_targets)
    soft_by_model = [[] for _ in models]
    clip_ids = []
    started = time.monotonic()
    completed = 0
    seen = 0
    try:
        with torch.inference_mode():
            for batch in loader:
                batch_start, batch_stop = seen, seen + len(batch['clip_id'])
                seen = batch_stop
                if batch_stop <= start:
                    continue
                if batch_start >= stop:
                    break
                local_start = max(0, start - batch_start)
                local_stop = min(len(batch['clip_id']), stop - batch_start)
                keep = local_stop - local_start
                lengths = batch['source_lengths'][local_start:local_stop].to(device, non_blocking=True)
                inputs = [batch[key][local_start:local_stop].to(device, non_blocking=True)
                          for key in ('speaker_audio','speaker_emotion','speaker_3dmm')]
                outputs = []
                for model in models:
                    outputs.append(model.generate(*inputs, lengths)['prediction'].cpu())
                for row in range(keep):
                    batch_row = local_start + row
                    clip_id = batch['clip_id'][batch_row]
                    session = batch['session_id'][batch_row]
                    clip_ids.append(clip_id)
                    soft = [outputs[m][row].float() for m in range(len(models))]
                    for m in range(len(models)):
                        soft_by_model[m].append(soft[m].half())
                    trajectories, origins, raw_count, unique = build_pool(soft, [x[0] for x in STUDENTS])
                    if raw_count != 30 or len(trajectories) != 60 or unique != 60:
                        raise ValueError(f'Unexpected TRAIN E60 pool at {clip_id}')
                    # clip_id is speaker/session/name; the paired target lives
                    # under the opposite-role listener/session directory.
                    paired = facial/'listener'/session/(Path(clip_id).name + '.npy')
                    paths = [paired] + [p for p in pools[session] if p != paired]
                    if not paired.exists() or len(paths) != len(pools[session]):
                        raise ValueError(f'GT pool mismatch at {clip_id}')
                    raw_targets = [torch.from_numpy(np.load(path, allow_pickle=False).astype(np.float32))
                                   for path in paths]
                    official = official_prediction(trajectories)
                    targets = _deterministic_postprocess(
                        processor, official, raw_targets, device, args.eval_seed, clip_id).cpu()
                    if targets.shape != (len(paths), 750, 25):
                        raise ValueError(f'Post-processed GT shape mismatch at {clip_id}: {targets.shape}')
                    candidates = [Candidate(official[k], str(k), 'train_e60', [('train_e60', k)])
                                  for k in range(60)]
                    frc, frd = quality_pool.score(candidates, targets)
                    np.savez_compressed(args.output/f'quality_{completed:04d}.npz',
                                        frc=frc, frd=frd, clip_id=clip_id,
                                        session_id=session, gt_count=len(paths))
                    completed += 1
                    if completed % 10 == 0 or completed == total:
                        progress = {'completed': completed, 'total': total,
                                    'elapsed_seconds': time.monotonic()-started,
                                    'last_clip_id': clip_id}
                        (args.output/'progress.json').write_text(json.dumps(progress, indent=2))
                        print(json.dumps(progress), flush=True)
    finally:
        quality_pool.close()
    if completed != total:
        raise RuntimeError(f'Only completed {completed}/{total}')
    prediction_path = args.output/'train_predictions.pt'
    torch.save({'soft_predictions':[torch.stack(v) for v in soft_by_model],
                'clip_ids':clip_ids,
                'checkpoint_sha256':{name:sha256_file(path) for name,path in STUDENTS},
                'split':'train','crop_mode':'center','frames':750,
                'context_start':start,'context_stop':stop,
                'eval_seed':args.eval_seed}, prediction_path)
    manifest = {'split':'train','contexts':total,'dataset_total':dataset_total,
                'context_start':start,'context_stop':stop,
                'candidate_pool':'E60 raw30 + scale1.5',
                'gt_policy':'all listener GT in the same session, paired first',
                'frames':750,'crop_mode':'center','eval_seed':args.eval_seed,
                'metric_workers':args.metric_workers,'data_workers':args.data_workers,
                'student_checkpoints':{name:{'path':str(path),'sha256':sha256_file(path)} for name,path in STUDENTS},
                'postprocessor_sha256':sha256_file(args.asset_mam_root/'framework/modules/post_processor.py'),
                'postprocessor_checkpoint_sha256':sha256_file(args.asset_mam_root/'pretrained_models/post_processor/checkpoint.pth'),
                'script_sha256':sha256_file(Path(__file__)),'prediction_file':str(prediction_path),
                'prediction_sha256':sha256_file(prediction_path),
                'elapsed_seconds':time.monotonic()-started,
                'git_commit':__import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
