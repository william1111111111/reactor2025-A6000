"""Standalone inference for the REACT2025 main-result CoReSet student."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from regnn.conditional_model import ConditionalREGNNConfig
from regnn.expert_student import ExpertConditionalREGNN, ExpertStudentConfig
from coreset_reactor.query0_au_transition import stabilize_query0_au_transitions

DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[1] / 'checkpoints/coreset_react2025_a0_seed1.pth'
WINDOW = 750
STRIDE = 375


def load_student(checkpoint=DEFAULT_CHECKPOINT, device='cpu'):
    """Return (student, checkpoint metadata), with strict state-dict loading."""
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    config = state['student_config']
    model = ExpertConditionalREGNN(ExpertStudentConfig(
        backbone=ConditionalREGNNConfig(**config['backbone']),
        num_experts=config['num_experts'],
        expert_embedding_dim=config['expert_embedding_dim'],
    ))
    model.load_state_dict(state['state_dict'], strict=True)
    return model.to(device).eval(), {k: v for k, v in state.items() if k != 'state_dict'}


def window_starts(length, window=WINDOW, stride=STRIDE):
    if length < 1 or window < 1 or not 1 <= stride <= window:
        raise ValueError('Need positive length/window and 1 <= stride <= window')
    if length <= window:
        return [0]
    final = length - window
    starts = list(range(0, final + 1, stride))
    if starts[-1] != final:
        starts.append(final)
    return starts


def center_weights(length):
    positions = torch.arange(length, dtype=torch.float32)
    midpoint = (length - 1) / 2.0
    return 1.0 - (positions - midpoint).abs() / ((length + 1) / 2.0)


def prepare_conditions(audio, facial_attributes, coefficients, mean, std, *, normalized=False):
    """Align pre-extracted source streams and apply the paper's 3DMM scaling."""
    streams = [torch.as_tensor(v, dtype=torch.float32).detach().cpu()
               for v in (audio, facial_attributes, coefficients)]
    if streams[2].ndim == 3 and streams[2].shape[1] == 1:
        streams[2] = streams[2][:, 0]
    for value, width in zip(streams, (768, 25, 58)):
        if value.ndim != 2 or value.shape[1] != width or len(value) < 1:
            raise ValueError(f'Expected nonempty [T,{width}], got {tuple(value.shape)}')
        if not torch.isfinite(value).all():
            raise ValueError('Input streams must be finite')
    length = min(map(len, streams))
    streams = [value[:length].contiguous() for value in streams]
    if not normalized:
        mean, std = [torch.as_tensor(v, dtype=torch.float32).reshape(-1) for v in (mean, std)]
        if mean.shape != (58,) or std.shape != (58,) or not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError('Need finite 58-D normalization statistics')
        streams[2] = (streams[2] - mean) / std.clamp_min(1e-8)
    return streams


@torch.inference_mode()
def predict_sequence(model, streams, batch_size=1):
    """Generate [10,T,25] soft outputs with the paper's normalized w^4 fusion.

    Streams are CPU tensors with normalized 3DMM coefficients. Window inference
    uses FP32, no teacher, reference targets, output permutation, or noise.
    """
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    if len(streams) != 3 or any(v.ndim != 2 for v in streams):
        raise ValueError('Expected three [T,D] input streams')
    length = len(streams[0])
    if length < 1 or any(len(v) != length for v in streams):
        raise ValueError('Input streams must have a common nonempty timeline')
    device = next(model.parameters()).device
    model.eval()
    starts = window_starts(length)
    predictions = []
    for offset in range(0, len(starts), batch_size):
        current = starts[offset:offset + batch_size]
        lengths = torch.tensor([min(WINDOW, length - start) for start in current], device=device)
        batch = [torch.stack([F.pad(v[start:start + WINDOW],
                                   (0, 0, 0, max(0, WINDOW - len(v[start:start + WINDOW]))))
                              for start in current]).to(device) for v in streams]
        result = model.generate(*batch, lengths)['prediction']
        predictions.extend(result.detach().float().cpu())
    accumulator = torch.zeros((model.config.num_experts, length, 25), dtype=torch.float32)
    denominator = torch.zeros(length, dtype=torch.float32)
    weights = center_weights(WINDOW).pow(4)
    for prediction, start in zip(predictions, starts):
        valid = min(WINDOW, length - start)
        weight = weights[:valid]
        accumulator[:, start:start + valid] += prediction[:, :valid] * weight[None, :, None]
        denominator[start:start + valid] += weight
    if not (denominator > 0).all():
        raise ValueError('Uncovered output frames')
    return accumulator / denominator[None, :, None]


def paper_postprocess(prediction, source_facial_attributes):
    """Apply official AU rounding and the fixed query-0 transition convention.

    The transition step is defined only when T > 98. It changes one frame in
    each constant query-0 AU for eligible source clips. Keep soft outputs for
    rendering and for analysis of within-set structure.
    """
    result = prediction.clone()
    result[..., :15] = result[..., :15].round()
    if result.shape[1] > 98:
        result = stabilize_query0_au_transitions(result, source_facial_attributes)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--audio', type=Path)
    parser.add_argument('--facial-attributes', type=Path)
    parser.add_argument('--coefficients', type=Path)
    parser.add_argument('--normalized-3dmm', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--demo', action='store_true', help='Use synthetic features for an installation check')
    parser.add_argument('--demo-frames', type=int, default=32)
    args = parser.parse_args()
    if args.demo and any((args.audio, args.facial_attributes, args.coefficients)):
        parser.error('Choose --demo or the three input files')
    if not args.demo and not all((args.audio, args.facial_attributes, args.coefficients)):
        parser.error('Provide --audio, --facial-attributes and --coefficients')
    if args.demo_frames < 1:
        parser.error('--demo-frames must be positive')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, metadata = load_student(args.checkpoint, args.device)
    if args.demo:
        generator = torch.Generator().manual_seed(1)
        values = [torch.randn(args.demo_frames, d, generator=generator) for d in (768, 25, 58)]
    else:
        values = [np.load(p, allow_pickle=False) for p in (args.audio, args.facial_attributes, args.coefficients)]
    normalization = metadata['normalization']
    streams = prepare_conditions(*values, normalization['mean'], normalization['std'],
                                 normalized=args.normalized_3dmm or args.demo)
    prediction = predict_sequence(model, streams, args.batch_size)
    paper_prediction = paper_postprocess(prediction, streams[1])
    if not torch.isfinite(prediction).all():
        raise ValueError('Nonfinite output')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('wb') as handle:
        np.savez_compressed(handle, prediction=prediction.numpy(),
                            paper_prediction=paper_prediction.numpy(),
                            expert_ids=np.arange(model.config.num_experts))
    print(json.dumps({'output': str(args.output), 'shape': list(prediction.shape),
                      'synthetic_demo': args.demo, 'parameters': sum(p.numel() for p in model.parameters())}))


if __name__ == '__main__':
    main()
