"""Synthetic duplicate-cache receipts; never merge independent generations."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import build_metrics as metrics


def receipt(seconds):
    return {
        "usage_delta": {"calls": 1, "input_tokens": 100, "output_tokens": 10, "seconds": seconds},
        "backend_usage": {"input_tokens": 100, "output_tokens": 10, "cached_input_tokens": 20},
        "campaign": "fixture-campaign",
        "subscription_launch": 42,
        "prompt_sha256": "a" * 64,
        "output": '{"fixture":true}',
        "error_type": None,
        "replayed": True,
    }


def test_duplicate_replay_receipt_with_tiny_elapsed_jitter_has_verified_tokens():
    first, second = receipt(5.473836167016998), receipt(5.473835707991384)
    match = metrics.usage_segment(first["usage_delta"], [first, second])
    assert match == [first]
    assert metrics.cost_unit(first["usage_delta"], match)["complete_token_telemetry"] is True


@pytest.mark.parametrize(
    "changed",
    [
        {"subscription_launch": 43},
        {"campaign": "independent-campaign"},
        {"prompt_sha256": "b" * 64},
        {"output": '{"fixture":false}'},
        {"error_type": "ModelOutputError"},
        {"backend_usage": {"input_tokens": 100, "output_tokens": 10, "cached_input_tokens": 21}},
    ],
)
def test_independent_or_nonidentical_receipts_remain_unknown(changed):
    first, second = receipt(1.0), {**receipt(1.0 + 0.4e-6), **changed}
    assert metrics.usage_segment(first["usage_delta"], [first, second]) is None


def test_elapsed_equivalence_never_exceeds_one_microsecond():
    first, second = receipt(1.0), receipt(1.0 + 1.5e-6)
    midpoint = {**first["usage_delta"], "seconds": 1.0 + 0.75e-6}
    assert metrics.usage_segment(midpoint, [first, second]) is None
