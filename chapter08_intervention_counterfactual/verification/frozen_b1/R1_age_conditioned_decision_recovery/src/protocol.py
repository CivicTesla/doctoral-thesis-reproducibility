"""R1 seed isolation and frozen-scheme checks."""


R0_FINAL_HOLDOUT = frozenset(range(120, 150))


def assert_seed_isolation(config):
    groups = {
        "train": set(config["train_seeds"]),
        "validation": set(config["validation_seeds"]),
        "pilot_test": set(config["pilot_test_seeds"]),
        "sealed_final_test": set(config["sealed_final_test_seeds"]),
    }
    for name, values in groups.items():
        if values & R0_FINAL_HOLDOUT:
            raise ValueError(f"{name} overlaps retired R0 holdout")
    names = list(groups)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            if groups[left] & groups[right]:
                raise ValueError(f"seed overlap: {left}, {right}")
    return groups


def assert_fixed_schemes(frozen_manifest):
    by_id = {row["freeze_id"]: row
             for row in frozen_manifest["frozen_schemes"]}
    if set(by_id) < {"F1", "F2", "F4"}:
        raise ValueError("R0 frozen scheme missing")
    expected = {
        "F1": "FIELDS_1_failure_count",
        "F2": "PERIOD_20_unknown",
        "F4": "ET5_TIME_SUCCESS_CROSS_10_AGE_TAGGED",
    }
    for freeze_id, scheme_id in expected.items():
        if by_id[freeze_id]["scheme_id"] != scheme_id:
            raise ValueError(f"{freeze_id} definition changed")
    return {freeze_id: by_id[freeze_id] for freeze_id in expected}
