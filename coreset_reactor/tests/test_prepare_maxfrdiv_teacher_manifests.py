import json

from coreset_reactor.prepare_maxfrdiv_teacher_manifests import export_rank_manifests


def test_export_rank_manifests_preserves_frozen_target_order(tmp_path):
    source = tmp_path / "source.json"
    source.write_text(json.dumps({
        "split": "train", "source_role": "speaker", "seed": 1,
        "summary": {"T0": {"FRDiv": .2}, "T1": {"FRDiv": .3}, "T2": {"FRDiv": .4}},
        "records": [{
            "sample_id": "speaker/session0/a.npy",
            "T0": {"target_ids": [f"listener/session0/{x}.npy" for x in ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]]},
            "T1": {"target_ids": [f"listener/session0/{x}.npy" for x in ["a", "j", "i", "h", "g", "f", "e", "d", "c", "b"]]},
        }],
    }))
    result = export_rank_manifests(source, tmp_path / "out")
    assert len(result["ranks"]) == 20
    t1 = json.loads((tmp_path / "out/T1/rank_03.json").read_text())
    assert t1["mapping"]["speaker/session0/a.npy"][0] == "listener/session0/a.npy"
    assert t1["mapping"]["speaker/session0/a.npy"][1] == "listener/session0/j.npy"
    assert t1["fixed_pair_alignment_policy"] == "relative_time_masked"
