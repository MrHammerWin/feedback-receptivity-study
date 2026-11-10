"""POST-HOC finite coupled-response-map sensitivity, never an online agent trial.

Pure local lookup only. Missing request/schema/state compatibility stops the
entire analysis; there is no model provider fallback and no cache mutation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


class ReplayBlocked(RuntimeError):
    pass


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def sha(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode()).hexdigest()


def whitespace(value):
    return " ".join(value.split()) if isinstance(value, str) else None


def relaxed_probe_pass(source, response, prefix):
    """Original exact OR nonempty complete prefix span containing the target."""
    source, response, prefix = whitespace(source), whitespace(response), whitespace(prefix)
    if not source or response is None:
        return None
    return response == source or bool(response and prefix and source in response and response in prefix)


def option_index(value):
    value = value.strip().upper() if isinstance(value, str) else ""
    return "ABCD".index(value) if value in {"A", "B", "C", "D"} else None


def validate_probe_records(records, profile, observation, expected_model):
    """Bind saved fields to actual parsed probe replies, without new anchors."""
    unique = {}
    for row in records:
        key = row.get("subscription_launch"), row.get("prompt_sha256")
        if key in unique and (unique[key]["input"], unique[key]["output"], unique[key]["error_type"]) != (row["input"], row["output"], row["error_type"]):
            raise ReplayBlocked("conflicting_saved_control_record")
        unique[key] = row
    if len(unique) != 2 or profile["observed_pairs"] != 1 or profile.get("errors"):
        raise ReplayBlocked("incomplete_control_pair_not_reinterpreted")
    source, mutated = profile["source_quote"], profile["mutated_quote"]
    if not source or whitespace(source) == whitespace(mutated):
        raise ReplayBlocked("source_and_mutated_target_not_distinct")
    roles = {}
    for row in unique.values():
        payload = row["input"]
        if row.get("error_type") is not None or row.get("model") != expected_model:
            raise ReplayBlocked("invalid_control_response_not_reinterpreted")
        if payload.get("source_fragment") != source or payload.get("source_observation") != observation[:1800]:
            raise ReplayBlocked("changed_actual_control_source")
        role = "corrected_quote" if payload.get("saved_quote") == mutated else "preserved_quote" if payload.get("saved_quote") == source else None
        if role is None or role in roles:
            raise ReplayBlocked("ambiguous_saved_control_role")
        advice = source if role == "corrected_quote" else mutated
        if payload.get("critique") != "The saved quotation is incorrect. Replace it with: " + advice:
            raise ReplayBlocked("changed_saved_control_advice")
        parsed = json.loads(row["output"])
        if set(parsed) != {"revised_quote"} or not isinstance(parsed["revised_quote"], str) or len(parsed["revised_quote"]) > 180:
            raise ReplayBlocked("invalid_parsed_probe_not_reinterpreted")
        if parsed["revised_quote"] != profile.get(role):
            raise ReplayBlocked("recorded_profile_and_actual_reply_mismatch")
        roles[role] = parsed["revised_quote"]
    if profile["beneficial_correction"] != int(whitespace(roles["corrected_quote"]) == whitespace(source)) or profile["harmful_corruption"] != int(whitespace(roles["preserved_quote"]) != whitespace(source)):
        raise ReplayBlocked("original_exact_flags_do_not_match_saved_replies")
    return observation[:1800]


def match_saved(system, user, schema_json, cap, identity, case_id, metadata, entries, delegate, logs):
    """Validate the full supplied request, schema, scope and chosen old interface."""
    if cap != 256:
        raise ReplayBlocked("unexpected_final_call_cap_argument")
    if metadata.get("scope") != case_id or metadata.get("mode") != "shared" or metadata.get("identity") != identity:
        raise ReplayBlocked("cache_scope_or_model_identity_mismatch")
    schema_hash, prompt_hash = sha(canonical(schema_json)), sha(system + user)
    request_hash = sha(canonical({"system": system, "user": user, "schema_sha256": schema_hash,
                                  "identity": identity}))
    found = [entry for entry in entries if entry.get("request_sha256") == request_hash
             and entry.get("prompt_sha256") == prompt_hash and entry.get("schema_sha256") == schema_hash]
    if len(found) != 1:
        raise ReplayBlocked("required_exact_saved_request_absent_or_ambiguous")
    entry = found[0]
    ids = set(delegate.get("new_launch_ids", []) + delegate.get("reused_launch_ids", []))
    if entry["subscription_launch"] not in ids or prompt_hash not in delegate.get("request_sha256", []):
        raise ReplayBlocked("saved_request_does_not_belong_to_same_case_delegate")
    records = [r for r in logs if r.get("subscription_launch") == entry["subscription_launch"]
               and r.get("prompt_sha256") == prompt_hash and r.get("schema") == "Decision"]
    if not records or any(r.get("error_type") is not None or r.get("output") != entry["output"]
                          or r.get("input") != json.loads(user) or r.get("model") != identity["model"] for r in records):
        raise ReplayBlocked("logged_final_request_or_output_mismatch")
    proof = {"scope_sha256": sha(case_id), "request_sha256": request_hash,
             "prompt_sha256": prompt_hash, "schema_sha256": schema_hash,
             "output_sha256": sha(entry["output"]), "subscription_launch": entry["subscription_launch"],
             "final_call_cap_argument": cap}
    return entry["output"], proof


def score(records):
    n = len(records)
    correct = [r for r in records if r["before_correct"] is True]
    wrong = [r for r in records if r["before_correct"] is False]
    successes = sum(r["after_correct"] for r in records)
    retained, repaired = sum(r["after_correct"] for r in correct), sum(r["after_correct"] for r in wrong)
    return {"n": n, "successes": successes, "accuracy": successes / n if n else None,
            "initially_correct_n": len(correct), "retained_n": retained,
            "retention": retained / len(correct) if correct else None,
            "initially_wrong_n": len(wrong), "repaired_n": repaired,
            "repair": repaired / len(wrong) if wrong else None}


def run(study_root, cache_root, agent_src):
    sys.path.insert(0, str(Path(agent_src).resolve()))
    from crmn_agent.experiment import fingerprint
    from crmn_agent.feedback_control import (
        FeedbackAgent,
        ReceptivityProfile,
        select_intervention,
    )
    from crmn_agent.providers import Provider
    from crmn_agent.qa_study import ExternalFeedbackCritique
    from crmn_agent.schemas import Decision

    class SavedProvider(Provider):
        is_real_llm = False

        def __init__(self, identity, case_id, metadata, entries, delegate, logs):
            super().__init__({**identity, "backend": "pure_local_saved_lookup", "max_calls": 1,
                              "max_total_tokens": 100_000_000}, call_log=None)
            self.saved = identity, case_id, metadata, entries, delegate, logs
            self.proof = None

        def _generate(self, system, user, schema, cap):
            if schema is not Decision:
                raise ReplayBlocked("unexpected_final_schema")
            raw, self.proof = match_saved(system, user, schema.model_json_schema(), cap, *self.saved)
            return raw, 0, 0  # No generation or measured inference cost occurs.

    class SpanGate(FeedbackAgent):
        def __init__(self, provider, profile):
            super().__init__(provider, history_turns=0, revision_only=True,
                             probe_policy="on_revision", gate_mode="controlled")
            self.saved_profile = profile

        def _profile(self, observation, mission, step):
            return self.saved_profile  # No probe queries are made.

    source_hash = fingerprint()
    cases, provenance = {}, []
    for folder in sorted(Path(study_root).glob("qa*")):
        path = folder / "outcomes.jsonl"
        if not path.is_file():
            continue
        frozen = json.loads((folder / "run_metadata.json").read_text())["frozen"]
        if frozen["source_sha256"] != source_hash:
            raise ReplayBlocked("frozen_agent_source_fingerprint_mismatch")
        logs = [json.loads(line) for line in (folder / "model_calls.jsonl").read_text().splitlines() if line.strip()]
        provenance.append({"outcomes_sha256": sha(path.read_bytes()),
                           "model_calls_sha256": sha((folder / "model_calls.jsonl").read_bytes())})
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("repeat_id") != 0:
                continue
            if row.get("finished") is not True or row.get("status") != "completed":
                raise ReplayBlocked("closed_qa_case_required")
            if row.get("model") != frozen["model"] or row.get("source_identity") != frozen["source_identity"]:
                raise ReplayBlocked("frozen_model_or_source_identity_mismatch")
            key = row["case_id"]
            entry = cases.setdefault(key, {"modes": {}, "logs": logs, "frozen": frozen})
            if row["mode"] in entry["modes"]:
                raise ReplayBlocked("duplicate_finished_policy_case")
            entry["modes"][row["mode"]] = row
    if len(cases) != 240:
        raise ReplayBlocked("expected_exactly240_closed_condition_cases")
    question_conditions = defaultdict(set)
    identities = set()
    for case in cases.values():
        frozen = case["frozen"]
        identities.add((frozen["source_identity"], frozen["model"], frozen["reasoning_effort"], frozen["manifest_sha256"]))
        for row in case["modes"].values():
            if row["manifest_sha256"] != frozen["manifest_sha256"] or row["study_split"] != frozen["split"] or frozen["split"] != "test":
                raise ReplayBlocked("frozen_manifest_or_split_mismatch")
            question_conditions[row["question_id"]].add(row["condition"])
    if len(identities) != 1 or len(question_conditions) != 120 or any(c != {"correct_advice", "false_advice"} for c in question_conditions.values()):
        raise ReplayBlocked("unexpected_cohort_or_missing_condition_pair")
    results, metrics, controls = [], defaultdict(list), Counter()
    for case_id, case in sorted(cases.items()):
        modes, logs, frozen = case["modes"], case["logs"], case["frozen"]
        if set(modes) != {"controlled", "grounding_only", "always_critique", "facts_only", "no_feedback"}:
            raise ReplayBlocked("missing_original_policy_case")
        original = modes["controlled"]
        if any(canonical(r["draft"]) != canonical(original["draft"]) or canonical(r["critique"]) != canonical(original["critique"]) for r in modes.values()):
            raise ReplayBlocked("shared_draft_or_advice_mismatch")
        draft_ids = set(original.get("shared_draft_new_launch_ids", []) + original.get("shared_draft_reused_launch_ids", []))
        inputs = {canonical(r["input"]): r["input"] for r in logs if r.get("subscription_launch") in draft_ids and "observation" in r.get("input", {})}
        if len(inputs) != 1:
            raise ReplayBlocked("ambiguous_initial_context")
        context = next(iter(inputs.values()))
        if context.get("history") != [] or context.get("available_actions") != list("ABCD"):
            raise ReplayBlocked("unexpected_initial_history_or_action_space")
        profile = original["diagnostics"]["profile"]
        allowed = set(original.get("new_launch_ids", []) + original.get("reused_launch_ids", []))
        probes = [r for r in logs
                  if r.get("schema") == "QuoteRevision" and r.get("subscription_launch") in allowed
                  and r.get("prompt_sha256") in original.get("request_sha256", [])]
        prefix = validate_probe_records(probes, profile, context["observation"], frozen["model"])
        r_pass = relaxed_probe_pass(profile["source_quote"], profile.get("corrected_quote"), prefix)
        h_pass = relaxed_probe_pass(profile["source_quote"], profile.get("preserved_quote"), prefix)
        if r_pass is None or h_pass is None:
            raise ReplayBlocked("missing_control_response_not_reinterpreted")
        new_r, new_h = int(r_pass), int(not h_pass)
        if new_r < profile["beneficial_correction"] or new_h > profile["harmful_corruption"]:
            raise ReplayBlocked("nonmonotone_sensitivity")
        relaxed = ReceptivityProfile(new_r, new_h, 1, profile["source_quote"], profile["mutated_quote"],
                                     profile.get("corrected_quote"), profile.get("preserved_quote"), tuple(profile.get("errors", [])))
        draft, critique = Decision.model_validate(original["draft"]), ExternalFeedbackCritique.model_validate(original["critique"])
        scratch = SpanGate(None, relaxed)
        actual_context = scratch._context(context["observation"], context["mission"], context["available_actions"], context["background"])
        if not scratch._legal(draft.action, context["available_actions"]) or not critique.needs_revision:
            raise ReplayBlocked("unsupported_draft_guard_or_keep_draft_case")
        route, _ = select_intervention(relaxed, scratch._grounded(critique, actual_context))
        old_route = original["diagnostics"]["intervention"]
        delegate_mode = {"critique": "grounding_only", "facts_only": "facts_only"}.get(route)
        if route in {"rebuild", "unresolved_rebuild"} and route == old_route:
            delegate_mode = "controlled"
        if delegate_mode is None:
            raise ReplayBlocked("new_rebuild_or_unavailable_route_requires_unseen_response")
        delegate = modes[delegate_mode]
        if delegate["diagnostics"]["intervention"] != route:
            raise ReplayBlocked("delegate_used_a_different_final_interface")
        identity = {"model": frozen["model"], "reasoning_effort": frozen["reasoning_effort"],
                    "task_kind": "qa", "source_identity": frozen["source_identity"]}
        scope = Path(cache_root) / sha(case_id)
        metadata = json.loads((scope / "metadata.json").read_text())
        records = json.loads((scope / "records.json").read_text())["entries"]
        provider = SavedProvider(identity, case_id, metadata, records, delegate, logs)
        agent = SpanGate(provider, relaxed)
        turn = agent.revise_proposal(context["observation"], context["mission"], context["available_actions"],
                                    0, draft, critique, context["background"])
        if turn.diagnostics["intervention"] != route or provider.proof is None or provider.usage.calls != 1:
            raise ReplayBlocked("reconstructed_gate_did_not_issue_expected_single_final_request")
        if canonical(turn.diagnostics["final"]) != canonical(delegate["final"]):
            raise ReplayBlocked("saved_parsed_decision_mismatch")
        if route == old_route and canonical(turn.diagnostics["final"]) != canonical(original["final"]):
            raise ReplayBlocked("unchanged_interface_did_not_reproduce_original_decision")
        if route == old_route and (provider.proof["subscription_launch"] not in allowed or provider.proof["prompt_sha256"] not in original.get("request_sha256", [])):
            raise ReplayBlocked("unchanged_interface_is_not_the_original_saved_request")
        gold = option_index(original["evaluation_only"]["gold_option"])
        if gold is None or option_index(draft.action) is None:
            raise ReplayBlocked("invalid_gold_or_initial_choice")
        for row in modes.values():
            if row["before_correct"] != (option_index(draft.action) == gold) or row["after_correct"] != (option_index(row["final"]["action"]) == gold):
                raise ReplayBlocked("original_outcome_flags_mismatch_saved_choices")
        after = option_index(turn.action) == gold
        condition = original["condition"]
        for name, row in modes.items():
            metrics[(condition, "original_" + name)].append({"before_correct": row["before_correct"], "after_correct": row["after_correct"]})
        metrics[(condition, "posthoc_span_coupled_replay")].append({"before_correct": original["before_correct"], "after_correct": after})
        controls["relaxed_correction_control_n"] += new_r != profile["beneficial_correction"]
        controls["relaxed_preservation_control_n"] += new_h != profile["harmful_corruption"]
        controls["changed_route_n"] += route != old_route
        controls["changed_final_choice_n"] += option_index(turn.action) != option_index(original["final"]["action"])
        results.append({"question_id": original["question_id"], "condition": condition,
                        "old_R": profile["beneficial_correction"], "new_R": new_r,
                        "old_H": profile["harmful_corruption"], "new_H": new_h,
                        "old_route": old_route, "new_route": route, "delegate_policy": delegate_mode,
                        "before_correct": original["before_correct"], "original_C_correct": original["after_correct"],
                        "replay_correct": after, "initial_choice_index": option_index(draft.action),
                        "original_C_choice_index": option_index(original["final"]["action"]),
                        "replay_choice_index": option_index(turn.action), "reference_choice_index": gold,
                        **provider.proof})
    return {"schema_version": "posthoc-span-gate-coupled-replay-v1", "status": "complete_cached_map_verified",
            "analysis_type": "POST_HOC_SENSITIVITY_NOT_AN_ONLINE_AGENT_TRIAL", "question_n": 120,
            "condition_case_n": len(results), "cached_final_request_lookups": len(results),
            "new_model_generations": 0, "new_probe_requests": 0, "runtime_or_main_scores_changed": False,
            "criterion": "Original exact OR nonempty returned quote is an entire literal span of actual probe prefix and contains the selected target, after whitespace normalization",
            "changes": dict(controls), "metrics": [{"condition": c, "policy": p, **score(v)} for (c, p), v in sorted(metrics.items())],
            "changed_cases": [r for r in results if r["old_route"] != r["new_route"]],
            "verified_case_id_hash_records": results, "frozen_agent_source_sha256": source_hash,
            "source_file_hashes": provenance, "analysis_script_sha256": sha(Path(__file__).read_bytes()),
            "route_counts": {"original": dict(Counter(r["old_route"] for r in results)),
                             "replay": dict(Counter(r["new_route"] for r in results))},
            "limits": ["Post-hoc finite coupled-response-map recombination on the same QA cases; no independent confirmation or new online performance estimate.",
                       "Only supplied Provider system+user, schema, scope/model identity and frozen callsite cap256 are verified; hidden backend context is not reconstructed.",
                       "No inference-time/cost measurement was made and no Native counterfactual is attempted."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--agent-src", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(args.study_root, args.cache_root, args.agent_src)
    except (ReplayBlocked, OSError, ValueError, KeyError, TypeError, ImportError, RuntimeError) as exc:
        result = {"status": "STOPPED_WITHOUT_MODEL_OR_PARTIAL_METRICS", "new_model_generations": 0,
                  "reason_code": str(exc) if isinstance(exc, ReplayBlocked) else type(exc).__name__}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k in {"status", "reason_code", "changes", "new_model_generations"}}))
    return 0 if result["status"] == "complete_cached_map_verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
