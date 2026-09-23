import json

from coreset_reactor.prepare_quality_gated_teacher_manifests import (
    export_rank_manifests,
)


def test_quality_gated_export_preserves_arm_targets_and_rank_count(tmp_path):
    arms = ("T0", "DESC_Q50")
    records = []
    for index in range(2):
        mapping = [f"listener/session/{index}_{rank}" for rank in range(10)]
        records.append({
            "sample_id": f"speaker/session/{index}",
            "paired_target_id": mapping[0],
            "T0": {"target_ids": mapping},
            "DESC_Q50": {"target_ids": [mapping[0], *mapping[:0:-1]]},
        })
    source = tmp_path / "source.json"
    source.write_text(json.dumps({
        "split": "train", "seed": 1, "records": records,
        "summary": {arm: {"FRDiv": .2} for arm in arms},
    }))
    result = export_rank_manifests(source, tmp_path / "out", arms)
    assert result["contexts"] == 2
    assert len(result["ranks"]) == 20
    rank = json.loads((tmp_path / "out" / "DESC_Q50" / "rank_03.json").read_text())
    assert rank["target_selection_arm"] == "DESC_Q50"
    assert len(rank["mapping"]) == 2
