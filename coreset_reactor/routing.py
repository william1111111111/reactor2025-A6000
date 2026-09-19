"""TRAIN-only behavioral mode discovery and slot-routed descriptor loss."""

from __future__ import annotations

import hashlib

import torch

from coreset_reactor.losses import softmin


def _balanced_assignment(distance: torch.Tensor,
                         capacities: torch.Tensor) -> torch.Tensor:
    """Assign every row to a column while respecting deterministic capacities.

    Start from the nearest medoid, then move the least-cost examples out of
    overfull modes until every mode reaches its requested capacity.
    """
    if distance.ndim != 2 or capacities.shape != (distance.shape[1],):
        raise ValueError("distance/capacity shape mismatch")
    if int(capacities.sum()) != distance.shape[0] or (capacities < 1).any():
        raise ValueError("capacities must be positive and sum to the sample count")
    labels = distance.argmin(1)
    sizes = torch.bincount(labels, minlength=distance.shape[1])
    row = torch.arange(distance.shape[0])
    while bool((sizes > capacities).any()):
        movable = sizes[labels] > capacities[labels]
        destinations = sizes < capacities
        delta = distance - distance[row, labels, None]
        allowed = movable[:, None] & destinations[None]
        candidate = delta.masked_fill(~allowed, torch.inf)
        flat = int(candidate.argmin())
        if not torch.isfinite(candidate.flatten()[flat]):
            raise RuntimeError("balanced assignment reached an infeasible state")
        sample = flat // distance.shape[1]
        target = flat % distance.shape[1]
        source = int(labels[sample])
        labels[sample] = target
        sizes[source] -= 1
        sizes[target] += 1
    if not torch.equal(torch.bincount(labels, minlength=distance.shape[1]), capacities):
        raise AssertionError("balanced assignment did not meet capacities")
    return labels


def balanced_medoid_modes(descriptors: torch.Tensor, clusters: int = 10,
                          seed: int = 1234,
                          max_iterations: int = 50) -> dict:
    """Deterministic capacity-balanced medoid clustering in descriptor space.

    Medoids are real TRAIN examples. Cluster capacities differ by at most one,
    and final mode labels are canonicalized by ascending medoid index.
    """
    if descriptors.ndim != 2 or not torch.isfinite(descriptors).all():
        raise ValueError("descriptors must be a finite [N,D] tensor")
    count, width = descriptors.shape
    if not 1 < clusters <= count or max_iterations < 1:
        raise ValueError("invalid cluster count or iteration limit")
    values = descriptors.detach().cpu().float()
    generator = torch.Generator().manual_seed(seed)
    medoids = [int(torch.randint(count, (), generator=generator))]
    nearest = ((values - values[medoids[0]]) ** 2).mean(1)
    while len(medoids) < clusters:
        candidate = int(nearest.argmax())
        if candidate in medoids:
            raise RuntimeError("farthest-first initialization selected a duplicate")
        medoids.append(candidate)
        distance = ((values - values[candidate]) ** 2).mean(1)
        nearest = torch.minimum(nearest, distance)

    base, extra = divmod(count, clusters)
    capacities = torch.tensor([base + (index < extra)
                               for index in range(clusters)], dtype=torch.long)
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        distance = torch.cdist(values, values[medoids]).square() / width
        labels = _balanced_assignment(distance, capacities)
        updated = []
        for mode in range(clusters):
            members = torch.where(labels == mode)[0]
            centroid = values[members].mean(0)
            score = ((values[members] - centroid) ** 2).mean(1)
            updated.append(int(members[score.argmin()]))
        if updated == medoids:
            break
        medoids = updated
    distance = torch.cdist(values, values[medoids]).square() / width
    labels = _balanced_assignment(distance, capacities)

    order = sorted(range(clusters), key=lambda mode: medoids[mode])
    remap = torch.empty(clusters, dtype=torch.long)
    remap[torch.tensor(order)] = torch.arange(clusters)
    labels = remap[labels]
    medoids = [medoids[mode] for mode in order]
    sizes = torch.bincount(labels, minlength=clusters)
    assigned = torch.cdist(values, values[medoids]).square() / width
    inertia = float(assigned[torch.arange(count), labels].mean())
    digest = hashlib.sha256(labels.numpy().tobytes()).hexdigest()
    return {"labels": labels, "medoid_indices": medoids,
            "cluster_sizes": sizes.tolist(), "iterations": iterations,
            "mean_squared_descriptor_distance": inertia,
            "assignment_sha256": digest}


def routed_descriptor_loss(distance: torch.Tensor, labels: torch.Tensor,
                           temperature: float) -> tuple[torch.Tensor, int]:
    """Route slot k only toward same-session GTs assigned to global mode k."""
    if distance.ndim != 3:
        raise ValueError("distance must have shape [B,K,N]")
    if labels.shape != (distance.shape[2],) or labels.dtype != torch.long:
        raise ValueError("labels must be int64 with shape [N]")
    if labels.numel() and (int(labels.min()) < 0 or
                           int(labels.max()) >= distance.shape[1]):
        raise ValueError("mode label is outside the slot range")
    terms = []
    for slot in range(distance.shape[1]):
        members = labels == slot
        if bool(members.any()):
            terms.append(softmin(distance[:, slot, members], 1, temperature))
    if not terms:
        raise ValueError("the current session supports no routing modes")
    return torch.stack(terms, 1).mean(), len(terms)
