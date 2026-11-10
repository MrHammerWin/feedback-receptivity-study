"""Synthetic, offline controls for the read-only post-hoc QA sensitivity."""

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

MODULE = Path(__file__).parents[1] / "analysis" / "posthoc_span_gate_replay.py"
spec = importlib.util.spec_from_file_location("span_replay", MODULE)
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


@pytest.mark.parametrize(
    ("source", "response", "prefix", "expected"),
    [
        ("target", "target", "different prefix", True),
        ("target", "before target after", "before target after", True),
        ("target", "before\n target after", "before target\t after", True),
        ("target", "other true span", "other true span target", False),
        ("target", "target fabricated", "target actual", False),
        ("target", "TARGET", "TARGET target", False),
        ("target", "", "target", False),
        ("target", None, "target", None),
        ("", "any", "any", None),
    ],
)
def test_monotone_prefix_span_criterion(source, response, prefix, expected):
    assert replay.relaxed_probe_pass(source, response, prefix) is expected


def fixture():
    identity = {"model": "synthetic", "reasoning_effort": "low", "task_kind": "qa", "source_identity": "frozen"}
    system, user, schema = "SYSTEM", json.dumps({"observation": "synthetic"}), {"title": "Decision"}
    schema_hash = replay.sha(replay.canonical(schema))
    prompt_hash = replay.sha(system + user)
    request_hash = replay.sha(replay.canonical({"system": system, "user": user, "schema_sha256": schema_hash, "identity": identity}))
    raw = '{"reflection":"synthetic","action":"A"}'
    metadata = {"scope": "case", "mode": "shared", "identity": identity}
    entries = [{"request_sha256": request_hash, "prompt_sha256": prompt_hash, "schema_sha256": schema_hash, "subscription_launch": 7, "output": raw}]
    delegate = {"new_launch_ids": [], "reused_launch_ids": [7], "request_sha256": [prompt_hash]}
    logs = [{"subscription_launch": 7, "prompt_sha256": prompt_hash, "schema": "Decision", "model": "synthetic", "input": json.loads(user), "output": raw, "error_type": None}]
    return [system, user, schema, 256, identity, "case", metadata, entries, delegate, logs]


def test_exact_query_saved_output_only():
    args = fixture()
    raw, proof = replay.match_saved(*args)
    assert raw == args[7][0]["output"]
    assert proof["subscription_launch"] == 7


@pytest.mark.parametrize("mutation", ["system", "user", "schema", "cap", "model", "scope", "mode", "duplicate", "launch", "raw", "payload", "error"])
def test_incompatible_or_missing_query_stops_without_fallback(mutation):
    args = deepcopy(fixture())
    if mutation == "system":
        args[0] += "changed"
    elif mutation == "user":
        args[1] = json.dumps({"observation": "changed"})
    elif mutation == "schema":
        args[2]["title"] = "Changed"
    elif mutation == "cap":
        args[3] = 128
    elif mutation == "model":
        args[4] = {**args[4], "model": "other"}
    elif mutation == "scope":
        args[6]["scope"] = "other-case"
    elif mutation == "mode":
        args[6]["mode"] = "sequence"
    elif mutation == "duplicate":
        args[7].append(deepcopy(args[7][0]))
    elif mutation == "launch":
        args[8]["reused_launch_ids"] = [8]
    elif mutation == "raw":
        args[9][0]["output"] = "different"
    elif mutation == "payload":
        args[9][0]["input"] = {"observation": "changed"}
    elif mutation == "error":
        args[9][0]["error_type"] = "ModelOutputError"
    with pytest.raises(replay.ReplayBlocked):
        replay.match_saved(*args)


def test_score_conditional_denominators_not_condition_rows_as_independent_trials():
    result = replay.score([
        {"before_correct": True, "after_correct": True},
        {"before_correct": True, "after_correct": False},
        {"before_correct": False, "after_correct": True},
    ])
    assert (result["retained_n"], result["initially_correct_n"], result["retention"]) == (1, 2, 0.5)
    assert (result["repaired_n"], result["initially_wrong_n"], result["repair"]) == (1, 1, 1.0)


def probe_fixture():
    profile = {"observed_pairs": 1, "errors": [], "source_quote": "target", "mutated_quote": "NOT target",
               "corrected_quote": "target extra", "preserved_quote": "target",
               "beneficial_correction": 0, "harmful_corruption": 0}
    records = []
    for launch, saved, answer in [(1, "NOT target", "target extra"), (2, "target", "target")]:
        records.append({"subscription_launch": launch, "prompt_sha256": str(launch), "model": "synthetic", "error_type": None,
                        "input": {"source_fragment": "target", "source_observation": "target extra", "saved_quote": saved,
                                  "critique": "The saved quotation is incorrect. Replace it with: " + ("target" if saved == "NOT target" else "NOT target")},
                        "output": json.dumps({"revised_quote": answer})})
    return records, profile


def test_actual_control_binding_and_duplicate_replayed_receipt():
    records, profile = probe_fixture()
    records.append(deepcopy(records[0]))
    assert replay.validate_probe_records(records, profile, "target extra", "synthetic") == "target extra"


@pytest.mark.parametrize("mutation", ["wrong_return", "missing", "invalid_json", "wrong_original_flag", "prefix", "role", "advice", "model", "transport", "conflicting_duplicate"])
def test_probe_fields_and_parsed_outputs_are_not_reconstructed_or_imputed(mutation):
    records, profile = probe_fixture()
    if mutation == "wrong_return":
        profile["corrected_quote"] = "different"
    elif mutation == "missing":
        records.pop()
    elif mutation == "invalid_json":
        records[0]["output"] = json.dumps({"revised_quote": None})
    elif mutation == "wrong_original_flag":
        profile["beneficial_correction"] = 1
    elif mutation == "prefix":
        records[0]["input"]["source_observation"] = "changed"
    elif mutation == "role":
        records[0]["input"]["saved_quote"] = "unexpected"
    elif mutation == "advice":
        records[0]["input"]["critique"] = "other advice"
    elif mutation == "model":
        records[0]["model"] = "other"
    elif mutation == "transport":
        records[0]["error_type"] = "BackendTransportError"
    elif mutation == "conflicting_duplicate":
        records.append(deepcopy(records[0]))
        records[-1]["output"] = json.dumps({"revised_quote": "target"})
    with pytest.raises(replay.ReplayBlocked):
        replay.validate_probe_records(records, profile, "target extra", "synthetic")
