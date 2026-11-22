"""TRAIN-only farthest-sum selection and consistent expert assignment.

Core selection/descriptor functions are unchanged from the paper runtime.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from coreset_reactor.descriptor import reaction_descriptor
from regnn.conditional_data import relative_time_resample_reaction


def full_reaction_descriptor(raw: torch.Tensor, segments: int = 8) -> torch.Tensor:
    """Describe every frame of one raw GT, independently of exact-loss cropping."""
    if raw.ndim != 2 or raw.shape[1] != 25 or len(raw) < 2 * segments:
        raise ValueError("raw GT must be [T,25] with at least two frames per segment")
    if not torch.isfinite(raw).all():
        raise ValueError("raw GT contains nonfinite values")
    return reaction_descriptor(raw, segments)


def official_distance_matrix(reactions: torch.Tensor) -> np.ndarray:
    """Return the symmetric pairwise terms used by official FRDiv."""
    if reactions.ndim != 3 or reactions.shape[-1] != 25:
        raise ValueError("reactions must have shape [N,T,25]")
    flat = reactions.detach().cpu().numpy().astype(np.float64).reshape(
        reactions.shape[0], -1
    )
    gram = flat @ flat.T
    squared = np.maximum(
        np.diag(gram)[:, None] + np.diag(gram)[None, :] - 2.0 * gram,
        0.0,
    )
    np.fill_diagonal(squared, 0.0)
    return squared / float(flat.shape[1])


def _sha256_file_set(paths: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _load_raw(path: Path) -> torch.Tensor:
    value = torch.from_numpy(np.load(path, allow_pickle=False).astype(np.float32))
    if value.ndim != 2 or value.shape[1] != 25 or len(value) == 0:
        raise ValueError(f"invalid reaction array: {path}: {tuple(value.shape)}")
    if not torch.isfinite(value).all():
        raise ValueError(f"nonfinite reaction array: {path}")
    return value


def _pairwise_sum(distance: np.ndarray, selected: list[int]) -> float:
    ids = np.asarray(selected, dtype=np.int64)
    return float(distance[np.ix_(ids, ids)].sum() / 2.0)


def paired_farthest_indices(
    distance: np.ndarray,
    paired_index: int,
    count: int = 10,
    max_swaps: int = 20,
) -> list[int]:
    """Approximate paired-constrained max-sum dispersion.

    The paired item is immutable.  Greedy additions and one-swap refinement
    are deterministic and operate on the same official distance matrix as the
    exact MILP implementation.
    """
    n = distance.shape[0]
    if distance.shape != (n, n) or not 0 <= paired_index < n or count < 2:
        raise ValueError("invalid distance matrix or paired index")
    if count > n:
        raise ValueError("selection count exceeds candidate count")
    selected = [paired_index]
    while len(selected) < count:
        candidates = [i for i in range(n) if i not in selected]
        gains = [float(distance[i, selected].sum()) for i in candidates]
        selected.append(candidates[int(np.argmax(gains))])
    for _ in range(max_swaps):
        current = _pairwise_sum(distance, selected)
        best = current
        best_swap = None
        outside = [i for i in range(n) if i not in selected]
        for position in range(1, count):  # rank zero is the paired target
            kept = selected[:position] + selected[position + 1:]
            for candidate in outside:
                value = _pairwise_sum(distance, kept + [candidate])
                if value > best + 1e-10:
                    best = value
                    best_swap = position, candidate
        if best_swap is None:
            break
        selected[best_swap[0]] = best_swap[1]
    return sorted(selected, key=lambda i: (i != paired_index, i))


def _fit_global_prototypes(
    data_root: Path, n_modes: int = 9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    root = data_root / "train" / "facial-attributes"
    paths = sorted(root.glob("*/*/*.npy"))
    if not paths:
        raise FileNotFoundError(f"no TRAIN facial attributes under {root}")
    raw = np.stack([full_reaction_descriptor(_load_raw(path)).numpy()
                    for path in paths])
    mean = raw.mean(0)
    std = np.maximum(raw.std(0), .03)
    values = (raw - mean) / std
    model = KMeans(n_clusters=n_modes, random_state=1, n_init=20).fit(values)
    nearest = np.mean((values[:, None] - model.cluster_centers_[None]) ** 2, 2)
    order = np.argsort(nearest.argmin(0))
    centers = model.cluster_centers_[order]
    return mean, std, centers, _sha256_file_set(paths, root)


def _assign_canonical_slots(
    selected: list[int], paired_index: int, values: np.ndarray,
    centers: np.ndarray,
) -> list[int]:
    """Keep rank zero paired and Hungarian-match the nine alternatives."""
    if paired_index not in selected or len(selected) != 10:
        raise ValueError("T1 canonical assignment requires paired target")
    alternatives = [index for index in selected if index != paired_index]
    cost = np.mean((values[alternatives, None] - centers[None]) ** 2, axis=-1)
    modes, columns = linear_sum_assignment(cost.T)
    assigned = [alternatives[int(columns[np.where(modes == mode)[0][0]])]
                for mode in range(centers.shape[0])]
    return [paired_index, *assigned]


def build_assignment(data_root: Path) -> dict:
    """Fit TRAIN prototypes and select ten opposite-role references per context."""
    root = data_root / 'train/facial-attributes'
    mean, std, centers, data_hash = _fit_global_prototypes(data_root)
    mapping = {}
    for role, opposite in [('speaker', 'listener'), ('listener', 'speaker')]:
        for session_dir in sorted((root / role).iterdir()):
            if not session_dir.is_dir():
                continue
            paths = sorted((root / opposite / session_dir.name).glob('*.npy'))
            if len(paths) < 10:
                raise ValueError(f'Need ten opposite-role references: {session_dir.name}')
            raw = [_load_raw(path) for path in paths]
            aligned = torch.stack([relative_time_resample_reaction(x, 750) for x in raw])
            distance = official_distance_matrix(aligned)
            values = (np.stack([full_reaction_descriptor(x).numpy() for x in raw]) - mean) / std
            ids = [path.relative_to(root).as_posix() for path in paths]
            lookup = {name: index for index, name in enumerate(ids)}
            for source in sorted(session_dir.glob('*.npy')):
                pair = lookup[f'{opposite}/{session_dir.name}/{source.name}']
                chosen = paired_farthest_indices(distance, pair)
                chosen = _assign_canonical_slots(chosen, pair, values, centers)
                mapping[source.relative_to(root).as_posix()] = [ids[i] for i in chosen]
    return {'schema_version': 1, 'split': 'train', 'k': 10,
            'assignment_method': 'farthest_sum_hungarian', 'mapping': mapping,
            'contexts': len(mapping), 'descriptor_mean': mean.tolist(),
            'descriptor_std': std.tolist(), 'prototype_centers': centers.tolist(),
            'train_facial_attributes_sha256': data_hash}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    value = build_assignment(args.data_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps({'contexts': value['contexts'], 'output': str(args.output)}))


if __name__ == '__main__':
    main()
