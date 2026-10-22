import pytest
import torch

from coreset_reactor.eval_multiview_ablation_bidir_all_session import (
    METRICS,
    validate_bidirectional_population,
)
from framework.metrics.FRDvs import compute_FRDvs


def _entries(role, count):
    return [
        {"role": role, "length": 245 + index, "clip_id": f"{role}/{index}"}
        for index in range(count)
    ]


def test_contract_accepts_exact_bidirectional_population():
    entries = _entries("speaker", 571) + _entries("listener", 571)
    validate_bidirectional_population(entries)
    assert METRICS == ("FRC", "exact_FRD", "FRDiv", "FRVar", "TLCC")


def test_contract_rejects_forward_only_population():
    with pytest.raises(ValueError, match="VAL1142"):
        validate_bidirectional_population(_entries("speaker", 571))


def test_contract_rejects_unbalanced_directions():
    entries = _entries("speaker", 572) + _entries("listener", 570)
    with pytest.raises(ValueError, match="571 contexts per direction"):
        validate_bidirectional_population(entries)


def test_react2024_official_frdvs_is_executable():
    predictions = torch.zeros(2, 10, 2, 1)
    predictions[1] = 1
    assert compute_FRDvs(predictions).item() == pytest.approx(1.0)
