from coreset_reactor.fixed_pair_manifest import assign_fixed_random


def test_fixed_random_is_order_independent_and_preserves_native_pair():
    candidates = [f"listener/session1/gt_{index}.npy" for index in range(12)]
    native = candidates[4]
    first = assign_fixed_random(candidates, native, "speaker/session1/x.npy", 7, 10)
    second = assign_fixed_random(list(reversed(candidates)), native,
                                 "speaker/session1/x.npy", 7, 10)
    assert first == second
    assert first[0][0] == native
    assert len(first[0]) == len(set(first[0])) == 10
    assert not any(first[1])


def test_fixed_random_marks_deterministic_repeats_when_pool_is_small():
    candidates = [f"listener/session1/gt_{index}.npy" for index in range(3)]
    native = candidates[0]
    targets, duplicate = assign_fixed_random(
        candidates, native, "speaker/session1/x.npy", 7, 10,
    )
    assert len(targets) == len(duplicate) == 10
    assert targets[0] == native
    assert sum(duplicate) == 7


def test_fixed_random_changes_with_assignment_seed():
    candidates = [f"listener/session1/gt_{index}.npy" for index in range(30)]
    native = candidates[0]
    a, _ = assign_fixed_random(candidates, native, "speaker/session1/x.npy", 7, 10)
    b, _ = assign_fixed_random(candidates, native, "speaker/session1/x.npy", 8, 10)
    assert a != b
