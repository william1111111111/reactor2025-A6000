"""TRAIN-only, globally consistent representative targets for ten anchors."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

from coreset_reactor.dataset import full_reaction_descriptor


def assign_targets(values, centers):
    """One distinct local representative per global mode; no capacity balancing.

    A small local-density cost discourages isolated trajectories. Hungarian
    assignment resolves conflicts when multiple modes prefer the same target.
    """
    distance = np.mean((values[:, None] - centers[None]) ** 2, axis=-1)
    local = np.mean((values[:, None] - values[None]) ** 2, axis=-1)
    neighbors = min(5, len(values) - 1)
    density = np.sort(local, axis=1)[:, 1:neighbors + 1].mean(axis=1)
    cost = distance + 0.1 * density[:, None]
    rows, cols = linear_sum_assignment(cost.T)
    assert np.array_equal(rows, np.arange(9))
    return cols, distance.argmin(axis=1)[cols]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--farthest', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    root = args.data_root / 'train' / 'facial-attributes'
    paths = sorted(root.glob('*/*/*.npy'))
    ids = [p.relative_to(root).as_posix() for p in paths]
    descriptors = []
    digest = hashlib.sha256()
    for p in paths:
        raw = np.load(p, allow_pickle=False).astype(np.float32)
        digest.update(p.relative_to(root).as_posix().encode())
        digest.update(raw.tobytes())
        descriptors.append(full_reaction_descriptor(torch.from_numpy(raw)).numpy())
    raw = np.stack(descriptors)
    mean, std = raw.mean(0), np.maximum(raw.std(0), .03)
    values = (raw - mean) / std
    km = KMeans(n_clusters=9, random_state=1, n_init=20).fit(values)
    # Canonical mode numbering by nearest real TRAIN trajectory index.
    order = np.argsort(((values[:, None] - km.cluster_centers_[None]) ** 2).mean(2).argmin(0))
    centers = km.cluster_centers_[order]
    lookup = {p: i for i, p in enumerate(ids)}
    pools = {}
    for i, name in enumerate(ids):
        key = tuple(Path(name).parts[:2])
        pools.setdefault(key, []).append(i)
    mapping, rows = {}, []
    for name in ids:
        role, session, basename = Path(name).parts
        opposite = 'listener' if role == 'speaker' else 'speaker'
        paired = f'{opposite}/{session}/{basename}'
        pair_index = lookup[paired]
        alternatives = [i for i in pools[(opposite, session)] if i != pair_index]
        if len(alternatives) < 9:
            raise ValueError(f'Fewer than nine legal alternatives for {name}')
        selected, nearest = assign_targets(values[alternatives], centers)
        baseline_indices = [pair_index] + [alternatives[i] for i in selected]
        if args.farthest:
            local_values = values[alternatives]
            local = np.mean((local_values[:, None] - local_values[None]) ** 2, axis=-1)
            density = np.sort(local, axis=1)[:, 1:6].mean(1)
            eligible = density <= np.quantile(density, .95)
            if eligible.sum() < 9:
                eligible[:] = True
            nearest_distance = np.mean((local_values - values[pair_index]) ** 2, axis=1)
            chosen = []
            for _ in range(9):
                scores = np.where(eligible, nearest_distance, -np.inf)
                choice = int(scores.argmax())
                chosen.append(choice)
                eligible[choice] = False
                nearest_distance = np.minimum(nearest_distance, local[:, choice])
            # Assign the chosen nine to global roles without changing membership.
            cost = np.mean((local_values[chosen, None] - centers[None]) ** 2, axis=-1)
            modes, members = linear_sum_assignment(cost.T)
            selected = np.array(chosen)[members]
            nearest = cost[members].argmin(1)
        indices = [pair_index] + [alternatives[i] for i in selected]
        mapping[name] = [ids[i] for i in indices]
        v = values[indices]
        distances = np.sqrt(((v[:, None] - v[None]) ** 2).mean(2))
        rows.append({'source': name, 'mode_nearest_agreement': float((nearest == np.arange(9)).mean()),
                     'mean_pairwise_descriptor_distance': float(distances[np.triu_indices(10, 1)].mean()),
                     'baseline_pairwise_descriptor_distance': float(np.sqrt(((values[baseline_indices,None]-values[None,baseline_indices])**2).mean(2))[np.triu_indices(10,1)].mean())})
    result = {'split': 'train', 'seed': 1, 'policy': 'paired + nine global KMeans modes; distinct local representatives; density weight 0.1',
              'data_sha256': digest.hexdigest(), 'mapping': mapping,
              'descriptor_mean': mean.tolist(), 'descriptor_std': std.tolist(),
              'centers': centers.tolist(), 'diagnostics': rows,
              'contexts': len(mapping), 'unique_gt': len(ids),
              'mean_mode_nearest_agreement': float(np.mean([r['mode_nearest_agreement'] for r in rows]))}
    if args.farthest:
        result['policy'] = 'paired + descriptor farthest-first nine; exclude top5pct local isolation; Hungarian global role ordering'
    result['mean_pairwise_descriptor_distance'] = float(np.mean([r['mean_pairwise_descriptor_distance'] for r in rows]))
    result['baseline_pairwise_descriptor_distance'] = float(np.mean([r['baseline_pairwise_descriptor_distance'] for r in rows]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('contexts', 'unique_gt', 'mean_mode_nearest_agreement', 'data_sha256')}, indent=2))


if __name__ == '__main__':
    main()
