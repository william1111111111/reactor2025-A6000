import json

import torch

from coreset_reactor.oracle_select import (
    Candidate,
    _deduplicate,
    _feasible,
    discover_checkpoints,
    refine_quality_constrained,
)


def test_discovery_records_missing_checkpoints_without_failing(tmp_path):
    run = tmp_path / "m2_mode_routing_w0015_full2500_seed20260918"
    nested = run / "m1_B3_seed20260918"
    nested.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({
        "seed": 20260918,
        "routing": {"weight": 0.015},
    }))
    (nested / "final_checkpoint.pt").write_bytes(b"test")
    specs = discover_checkpoints([tmp_path])
    assert [spec.label for spec in specs] == ["route_0.015"]
    assert specs[0].family == "route"
    assert specs[0].seed == 20260918


def test_deduplicate_preserves_all_origins_and_prefers_m1_canonical_source():
    trajectory = torch.zeros(4, 25)
    candidates = _deduplicate([
        ("route_0.015", 2, trajectory, "route"),
        ("m1_seed_20260918", 7, trajectory, "m1"),
    ])
    assert len(candidates) == 1
    assert candidates[0].canonical_source == "m1_seed_20260918"
    assert set(candidates[0].origins) == {
        ("route_0.015", 2), ("m1_seed_20260918", 7)
    }


def test_quality_constrained_refinement_never_breaks_quality_constraints():
    # The first ten are the matched baseline. Candidate 10 is diverse but bad
    # quality; candidate 11 is diverse and just good enough to be admitted.
    points = torch.tensor([[float(i) / 10] for i in range(12)])
    distance = torch.cdist(points, points).square()
    frc = torch.tensor([1.0] * 10 + [0.1, 1.0]).numpy()
    frd = torch.tensor([1.0] * 10 + [20.0, 1.0]).numpy()
    baseline = list(range(10))
    selected = refine_quality_constrained(
        distance, frc, frd, baseline, frc_target=10.0, frd_target=10.0)
    assert len(selected) == 10
    assert _feasible(selected, frc, frd, 10.0, 10.0)
    assert 10 not in selected
    assert 11 in selected or selected == baseline

