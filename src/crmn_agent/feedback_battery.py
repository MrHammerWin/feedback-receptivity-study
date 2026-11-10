"""Frozen benchmark observations transformed into an explicit fidelity microtask.

This battery is NOT a benchmark's official planning-success protocol. Ground
truth is a literal public quotation; scorer-only labels never enter model input.
Calling ``run_battery`` requires an explicitly supplied provider. The CLI only
builds frozen data and never initializes or invokes a model backend.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from pydantic import Field

from .feedback_control import FeedbackAgent, FeedbackCritique, mutate_quote
from .grounding import observation_anchors
from .schemas import Decision, Schema


class EvaluationOnly(Schema):
    expected_action: str
    draft_condition: Literal["correct", "wrong"]
    feedback_condition: Literal["helpful", "harmful"]


class SourceRecord(Schema):
    file: str
    sha256: str


class BatteryCase(Schema):
    case_id: str
    state_id: str
    benchmark: str
    task_id: str
    seed: int
    step: int
    source_episode_sha256: str
    source_mission: str
    observation: str
    mission: str
    available_actions: list[str] = Field(min_length=2, max_length=2)
    draft: Decision
    critique: FeedbackCritique
    evaluation_only: EvaluationOnly


class BatteryManifest(Schema):
    schema_version: Literal["feedback-source-fidelity-v1"] = "feedback-source-fidelity-v1"
    protocol: Literal["transformed_source_fidelity_not_stock_benchmark"] = (
        "transformed_source_fidelity_not_stock_benchmark"
    )
    split: str
    selection_policy: str
    sources: list[SourceRecord]
    cases: list[BatteryCase]


def _digest(value) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def build_battery(
    episode_paths: list[Path],
    *,
    max_states: int = 4,
    steps: tuple[int, ...] = (0, 2, 4),
    split: str = "heldout",
    output_path: Path | None = None,
) -> BatteryManifest:
    """Freeze a predeclared selection of public observations, without outcome filters.

    Input episode files may contain reward or future observations, but only the
    current public observation and source-identifying metadata are extracted.
    Fixed lexicographic file order and explicit steps, not model outcomes, select
    states. Each selected state yields all four crossed feedback conditions.
    """
    if max_states <= 0 or not steps or any(step < 0 for step in steps):
        raise ValueError("Choose a positive number of states and nonnegative fixed steps")
    if len(set(steps)) != len(steps):
        raise ValueError("Duplicate selection steps are not allowed")
    sources = []
    selected = []
    seen = set()
    for path in sorted({Path(p).resolve() for p in episode_paths}):
        raw = path.read_bytes()
        source_hash = hashlib.sha256(raw).hexdigest()
        episode = json.loads(raw)
        sources.append(SourceRecord(file=path.name, sha256=source_hash))
        for index in sorted(steps):
            trajectory = episode["trajectory"]
            if index >= len(trajectory):
                continue
            frame = trajectory[index]
            observation = frame["observation"]
            source_mission = frame["mission"]
            anchors = observation_anchors(observation, source_mission)
            if not anchors:
                continue
            public_identity = {
                "benchmark": episode["benchmark"], "task_id": episode["task_id"],
                "seed": episode["seed"], "step": frame.get("step", index),
                "observation": observation, "mission": source_mission,
            }
            state_id = _digest(public_identity)[:20]
            if state_id in seen:
                continue
            seen.add(state_id)
            selected.append((episode, frame, index, source_hash, state_id, anchors[0].value))
            if len(selected) == max_states:
                break
        if len(selected) == max_states:
            break
    if not selected:
        raise ValueError("No selected public observations contain a quotation anchor")
    cases = []
    for episode, frame, index, source_hash, state_id, source in selected:
        corrupted = mutate_quote(source)
        # The competing option must not also be a verbatim observation fragment.
        while corrupted in frame["observation"]:
            corrupted = "NOT: " + corrupted
        if len(corrupted) > 200:
            raise ValueError("Unable to construct a bounded, unambiguous quotation control")
        actions = [source, corrupted]
        # Predeclared order counterbalance; every policy sees the identical choices.
        if int(state_id, 16) % 2:
            actions.reverse()
        for draft_condition in ("correct", "wrong"):
            proposed = source if draft_condition == "correct" else corrupted
            for feedback_condition in ("helpful", "harmful"):
                suggested = source if feedback_condition == "helpful" else corrupted
                case_id = f"{state_id}--{draft_condition}--{feedback_condition}"
                cases.append(BatteryCase(
                    case_id=case_id, state_id=state_id, benchmark=episode["benchmark"],
                    task_id=episode["task_id"], seed=episode["seed"],
                    step=frame.get("step", index), source_episode_sha256=source_hash,
                    source_mission=frame["mission"], observation=frame["observation"],
                    mission="Choose exactly the available quotation that occurs verbatim in the observation.",
                    available_actions=actions,
                    draft=Decision(reflection="Proposed exact quotation.", action=proposed),
                    critique=FeedbackCritique(
                        needs_revision=True,
                        issue="The proposal is incorrect. Use this exact quotation: " + suggested,
                        evidence=source,
                        suggested_action=suggested,
                    ),
                    evaluation_only=EvaluationOnly(
                        expected_action=source, draft_condition=draft_condition,
                        feedback_condition=feedback_condition,
                    ),
                ))
    manifest = BatteryManifest(
        split=split,
        selection_policy=(f"Lexicographic episode files, step indices {list(sorted(steps))}, "
                          f"first {max_states} unique anchored public states; no outcome filtering"),
        sources=sources, cases=cases,
    )
    if output_path is not None:
        if Path(output_path).exists():
            raise FileExistsError("Refusing to overwrite a frozen feedback battery")
        _write_json(Path(output_path), manifest.model_dump())
    return manifest


def _usage(provider) -> dict:
    if hasattr(provider, "usage"):
        return asdict(provider.usage)
    # Scripted test providers expose their actual call list without a real backend.
    return {"calls": len(provider.calls)}


def summarize_results(rows: list[dict]) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["mode"]].append(row)
    result = {}
    for mode, items in sorted(grouped.items()):
        wrong = [r for r in items if not r["before_correct"]]
        correct = [r for r in items if r["before_correct"]]
        corrected = sum(r["after_correct"] for r in wrong)
        corrupted = sum(not r["after_correct"] for r in correct)
        result[mode] = {
            "n": len(items), "state_n": len({r["state_id"] for r in items}),
            "accuracy": sum(r["after_correct"] for r in items) / len(items),
            "bad_to_good": {"n": len(wrong), "count": corrected,
                            "rate": corrected / len(wrong) if wrong else None},
            "good_to_bad": {"n": len(correct), "count": corrupted,
                            "rate": corrupted / len(correct) if correct else None},
            "usage": {key: sum(row["usage"].get(key, 0) for row in items)
                      for key in {key for row in items for key in row["usage"]}},
        }
    return result


def validate_battery(manifest: BatteryManifest) -> None:
    """Check that objective labels follow from public quote identity, before calls."""
    seen = set()
    conditions = defaultdict(set)
    for case in manifest.cases:
        label = case.evaluation_only
        if case.case_id in seen:
            raise ValueError("Duplicate battery cases are not independent repeats")
        seen.add(case.case_id)
        if len(set(case.available_actions)) != 2:
            raise ValueError("The quotation choices must be distinct")
        observed = [action for action in case.available_actions if action in case.observation]
        if observed != [label.expected_action]:
            raise ValueError("Quotation target must be uniquely determined by the public observation")
        if case.draft.action not in case.available_actions:
            raise ValueError("A frozen proposal must use one of the shared quotation choices")
        if (case.draft.action == label.expected_action) != (label.draft_condition == "correct"):
            raise ValueError("Draft-condition label disagrees with the fixed proposal")
        if (case.critique.suggested_action == label.expected_action) != (
            label.feedback_condition == "helpful"
        ) or case.critique.suggested_action not in case.available_actions:
            raise ValueError("Feedback-condition label disagrees with the injected quotation")
        conditions[case.state_id].add((label.draft_condition, label.feedback_condition))
    crossed = {(draft, feedback) for draft in ("correct", "wrong")
               for feedback in ("helpful", "harmful")}
    if any(values != crossed for values in conditions.values()):
        raise ValueError("Every frozen state must include all four predeclared conditions")


def run_battery(
    manifest: BatteryManifest | Path | str | dict,
    provider,
    *,
    modes: tuple[str, ...] = ("controlled", "always_critique", "facts_only"),
    output_path: Path | None = None,
) -> dict:
    """Evaluate the same fixed proposals through different admission policies.

    This function invokes only the explicitly supplied provider. Caller owns the
    authorized ledger and launch budget. Every case starts with a fresh agent.
    Invalid output or budget exhaustion is checkpointed and re-raised, never
    silently retried or replaced by the quotation target.
    """
    if isinstance(manifest, (Path, str)):
        manifest = BatteryManifest.model_validate_json(Path(manifest).read_text())
    elif isinstance(manifest, dict):
        manifest = BatteryManifest.model_validate(manifest)
    if not modes or len(set(modes)) != len(modes) or set(modes) - {
        "controlled", "always_critique", "facts_only", "always_rebuild",
    }:
        raise ValueError("Choose distinct, supported policies before running the battery")
    if not manifest.cases:
        raise ValueError("Cannot evaluate an empty battery")
    validate_battery(manifest)
    if output_path is not None and Path(output_path).exists():
        raise FileExistsError("Refusing to silently rerun or overwrite feedback battery results")
    initial_usage = _usage(provider)
    report = {
        "schema_version": "feedback-battery-results-v1", "protocol": manifest.protocol,
        "manifest_sha256": _digest(manifest.model_dump()), "split": manifest.split,
        "model": getattr(provider, "model", "scripted-offline-test"), "modes": list(modes),
        "status": "running", "rows": [], "summaries": {},
        "interpretation": "Transformed source-fidelity microtask, not official benchmark task success.",
        "independence_unit": "state; four conditions from one state are correlated",
        "comparison": "controlled 3 calls versus no-probe ablations 1 call; report overhead",
    }

    def checkpoint():
        report["summaries"] = summarize_results(report["rows"])
        report["usage_total"] = {key: value - initial_usage.get(key, 0)
                                 for key, value in _usage(provider).items()}
        if output_path is not None:
            _write_json(Path(output_path), report)
            records = Path(output_path).with_suffix(".records.jsonl")
            temporary = records.with_suffix(records.suffix + ".tmp")
            temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                         for row in report["rows"]), encoding="utf-8")
            temporary.replace(records)

    checkpoint()
    try:
        for case in manifest.cases:
            for mode in modes:
                before_usage = _usage(provider)
                agent = FeedbackAgent(provider, history_turns=0, gate_mode=mode,
                                      probe_policy="always" if mode == "controlled" else "never")
                # Explicit allowlist: evaluation_only and identifying condition names cannot leak.
                turn = agent.revise_proposal(
                    case.observation, case.mission, case.available_actions, case.step,
                    case.draft, case.critique,
                )
                usage = _usage(provider)
                label = case.evaluation_only
                report["rows"].append({
                    "case_id": case.case_id, "state_id": case.state_id,
                    "benchmark": case.benchmark, "task_id": case.task_id, "seed": case.seed,
                    "step": case.step, "mode": mode,
                    "draft_condition": label.draft_condition,
                    "feedback_condition": label.feedback_condition,
                    "before_correct": case.draft.action == label.expected_action,
                    "after_correct": turn.action == label.expected_action,
                    "action": turn.action, "expected_action": label.expected_action,
                    "intervention": turn.diagnostics["intervention"],
                    "gate_reason": turn.diagnostics["gate_reason"],
                    "profile": turn.diagnostics["profile"],
                    "usage": {key: value - before_usage.get(key, 0) for key, value in usage.items()},
                })
                checkpoint()
    except Exception as exc:
        report["status"] = "interrupted_" + type(exc).__name__
        checkpoint()
        raise
    report["status"] = "completed"
    checkpoint()
    return report


def main():
    parser = argparse.ArgumentParser(description="Build frozen source-fidelity data; no model calls")
    parser.add_argument("episodes", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-states", type=int, default=4)
    parser.add_argument("--steps", type=int, nargs="+", default=[0, 2, 4])
    parser.add_argument("--split", default="heldout")
    args = parser.parse_args()
    build_battery(list(args.episodes.glob("*.json")), max_states=args.max_states,
                  steps=tuple(args.steps), split=args.split, output_path=args.output)


if __name__ == "__main__":
    main()
