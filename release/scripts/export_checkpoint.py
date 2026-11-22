"""Export a weights-only student without optimizer state or local path metadata."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state_hash(state):
    digest = hashlib.sha256()
    for key, value in state.items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--normalization-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = torch.load(args.source, map_location='cpu', weights_only=True)
    state = {key: value.detach().cpu().contiguous() for key, value in source['state_dict'].items()}
    payload = {'format_version': 1, 'state_dict': state,
               'student_config': source['student_config'],
               'global_step': source['global_step'],
               'source_checkpoint_sha256': sha256(args.source),
               'parameter_state_sha256': state_hash(state),
               'normalization': {name: torch.from_numpy(np.load(args.normalization_dir / f'{name}_face.npy',
                                                                allow_pickle=False).astype(np.float32)).reshape(-1)
                                 for name in ['mean', 'std']},
               'inference': {'window': 750, 'stride': 375, 'overlap_power': 4.0,
                             'paper_output': 'AU rounding, then fixed query-0 constant-AU transition'}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    loaded = torch.load(args.output, map_location='cpu', weights_only=True)
    assert list(loaded['state_dict']) == list(source['state_dict'])
    assert all(torch.equal(loaded['state_dict'][key], value) for key, value in source['state_dict'].items())
    print(json.dumps({'output': str(args.output), 'bytes': args.output.stat().st_size,
                      'sha256': sha256(args.output), 'source_sha256': sha256(args.source),
                      'parameter_state_sha256': state_hash(state),
                      'tensor_values_identical': True}))


if __name__ == '__main__':
    main()
