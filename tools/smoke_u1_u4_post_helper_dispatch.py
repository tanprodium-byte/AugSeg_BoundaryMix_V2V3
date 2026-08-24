#!/usr/bin/env python3
"""Regression for the U1–U4 post-helper training-loop dispatch seam."""

import ast
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_semi import is_u_saliency_mix_branch, select_unlabeled_mix_branch  # noqa: E402


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def exercise_post_helper_dispatch(mix_branch, *, legacy_adaptive_enabled=True, ar_applied=True):
    """Instrument the production predicate at both actual training-loop decision points."""
    calls = {"u_helper": 0, "legacy_helper": 0}
    initial = (object(), object(), object())
    u_output = (object(), object(), object())
    legacy_output = (object(), object(), object())

    def u_helper():
        calls["u_helper"] += 1
        return u_output

    def legacy_helper():
        calls["legacy_helper"] += 1
        np.random.uniform(0, 1)
        return legacy_output

    current = initial
    if ar_applied:
        if is_u_saliency_mix_branch(mix_branch):
            current = u_helper()
        if is_u_saliency_mix_branch(mix_branch):
            pass
        elif legacy_adaptive_enabled:
            current = legacy_helper()
        else:
            # This mirrors the current final else branch, which calls the same helper.
            current = legacy_helper()
    return calls, current, initial, u_output, legacy_output


def test_actual_source_uses_shared_seam():
    source = (ROOT / "train_semi.py").read_text()
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "train")
    train_source = ast.get_source_segment(source, function)
    require(train_source.count("if is_u_saliency_mix_branch(mix_branch):") == 2, "training loop must use the seam twice")
    require('if mix_branch == "u1":' not in train_source, "U1-only bypass remains")
    require('if mix_branch in ("u1", "u2", "u3", "u4"):' not in train_source, "disconnected U-helper condition remains")
    require(train_source.count("cut_mix_label_adaptive(") == 2, "legacy final branches changed unexpectedly")


def test_u1_u4_call_counts_lineage_and_rng():
    for method in ("u1", "u2", "u3", "u4"):
        np.random.seed(710)
        before = np.random.get_state()
        calls, final, _, u_output, _ = exercise_post_helper_dispatch(method)
        after = np.random.get_state()
        require(calls == {"u_helper": 1, "legacy_helper": 0}, f"{method} call counts")
        require(all(final[index] is u_output[index] for index in range(3)), f"{method} RGB/pseudo/confidence lineage")
        require(before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:], f"{method} legacy RNG")


def test_legacy_and_feature_disabled_paths():
    for adaptive_enabled in (True, False):
        np.random.seed(711)
        calls, final, _, _, legacy_output = exercise_post_helper_dispatch(
            "legacy", legacy_adaptive_enabled=adaptive_enabled
        )
        require(calls == {"u_helper": 0, "legacy_helper": 1}, "legacy call counts")
        require(all(final[index] is legacy_output[index] for index in range(3)), "legacy final lineage")

    np.random.seed(712)
    expected = np.random.get_state()
    expected_draw = np.random.uniform(0, 1)
    expected_after = np.random.get_state()
    np.random.set_state(expected)
    branch, triggered, applied = select_unlabeled_mix_branch(False, False, 0.0)
    actual_after = np.random.get_state()
    require((branch, triggered, applied) == ("legacy", 0, 0), "feature-disabled branch")
    require(0 <= expected_draw < 1, "expected draw")
    require(expected_after[0] == actual_after[0] and np.array_equal(expected_after[1], actual_after[1]) and expected_after[2:] == actual_after[2:], "disabled baseline RNG trajectory")
    calls, final, initial, _, _ = exercise_post_helper_dispatch(branch, ar_applied=bool(applied))
    require(calls == {"u_helper": 0, "legacy_helper": 0}, "disabled baseline helper calls")
    require(all(final[index] is initial[index] for index in range(3)), "disabled baseline lineage")


def main():
    tests = (
        test_actual_source_uses_shared_seam,
        test_u1_u4_call_counts_lineage_and_rng,
        test_legacy_and_feature_disabled_paths,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print("PASS smoke_u1_u4_post_helper_dispatch")


if __name__ == "__main__":
    main()
