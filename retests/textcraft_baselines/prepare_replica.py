"""Prepare fresh StateAct/ReflAct inference without opening recorded campaigns."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import sys
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(directory, campaign, limit, consent):
    if not consent:
        raise ValueError("Explicit new subscription-owner consent is required")
    if not re.fullmatch(r"replica-[A-Za-z0-9_-]{1,60}", campaign) or limit != 2560:
        raise ValueError("Use a new replica- campaign and explicitly authorize at most 2560 attempts")
    if (os.name != "posix" or sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 12)
            or os.environ.get("PYTHONHASHSEED") != "0" or sys.flags.hash_randomization != 0):
        raise ValueError("Use CPython 3.12 on Linux/WSL with PYTHONHASHSEED=0 before Python starts")
    root = Path(__file__).resolve().parent
    prior = root.parent / "textcraft"
    state = Path(directory).resolve()
    if state.exists() or state.is_relative_to(root.parent):
        raise FileExistsError("Choose a new private directory outside published retest files")
    release = json.loads((root / "release_manifest.json").read_text(encoding="utf-8"))
    for name, expected in release["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or digest(path) != expected:
            raise ValueError("A baseline release file changed")
    record = json.loads((root / "source_record.json").read_text(encoding="utf-8"))
    environment = json.loads((prior / "source_record.json").read_text(encoding="utf-8"))["recorded_environment"]
    for name in ("textcraft", "pydantic", "numpy", "gymnasium"):
        if importlib.metadata.version(name) != environment[name]:
            raise ValueError(f"Use the recorded runtime version of {name}")
    inputs = prior / "frozen_inputs"
    if (digest(inputs / "manifest.json") != record["recorded_manifest_sha256"]
            or digest(inputs / "pilot_audit.json") != record["public_audit_sha256"]):
        raise ValueError("The published forty-goal inputs or redacted audit changed")
    files = {}
    for name, expected in record["common_source_sha256"].items():
        path = prior / "frozen_runner/src/crmn_agent" / name
        if not re.fullmatch(r"[A-Za-z0-9_]+\.py", name) or digest(path) != expected:
            raise ValueError("A shared frozen source file changed")
        files[name] = path
    for name, expected in record["source_delta_sha256"].items():
        path = root / "source_delta" / name
        if Path(name).name != name or digest(path) != expected or name in files:
            raise ValueError("An exact baseline source file changed")
        files[name] = path
    package = hashlib.sha256()
    for name in sorted(files):
        package.update(name.encode())
        package.update(files[name].read_bytes())
    if package.hexdigest() != record["executed_source_sha256"]:
        raise ValueError("The assembled source differs from the recorded baseline package")
    original = files["experiment.py"].read_text(encoding="utf-8")
    changes = {
        "CAMPAIGN = 'textcraft-published-baselines'": "CAMPAIGN = " + repr(campaign),
        "RETEST_AUDIT_SHA256 = '" + record["recorded_private_audit_sha256"] + "'":
            "RETEST_AUDIT_SHA256 = '" + record["public_audit_sha256"] + "'",
    }
    updated = original
    for old, new in changes.items():
        if updated.count(old) != 1:
            raise ValueError("A deployment identity assignment is not unique")
        updated = updated.replace(old, new, 1)
    compile(updated, "experiment.py", "exec")
    state.mkdir(parents=True, exist_ok=False)
    target = state / "runner/src/crmn_agent"
    target.mkdir(parents=True)
    for name, path in files.items():
        shutil.copyfile(path, target / name)
    (target / "experiment.py").write_text(updated, encoding="utf-8", newline="\n")
    shutil.copytree(inputs, state / "inputs")
    sys.path.insert(0, str(target.parent))
    from crmn_agent.experiment import CAMPAIGN, MODEL, fingerprint, load_study, preflight_tasks
    from crmn_agent.asset_download import resolve_assets
    from crmn_agent.public_authorization import authorize
    if CAMPAIGN != campaign:
        raise ValueError("An unexpected crmn_agent package was imported")
    tasks, _, _ = load_study(state / "inputs/manifest.json", state / "inputs/pilot_audit.json")
    resolve_assets()
    if len(preflight_tasks(tasks)) != 40:
        raise ValueError("Forty matching baseline starting states are required")
    provider_path = authorize(state, campaign, limit, MODEL, True, resource_profile="recorded-study")
    provider = json.loads(provider_path.read_text(encoding="utf-8"))["provider"]
    checkpoint = state / "checkpoint.json"
    with checkpoint.open("x", encoding="utf-8") as stream:
        json.dump({"stop_after": limit}, stream)
    provider.update(subscription_checkpoint=str(checkpoint), stop_on_transport_error=True)
    config_path = state / "config.json"
    with config_path.open("x", encoding="utf-8") as stream:
        json.dump({"provider": provider, "manifest": str(state / "inputs/manifest.json"),
                   "pilot_audit": str(state / "inputs/pilot_audit.json"), "results_dir": str(state / "results")}, stream, indent=2)
    result = {"campaign": campaign, "authorized_max_attempts": limit,
              "recorded_source_sha256": record["executed_source_sha256"], "replica_source_sha256": fingerprint(),
              "config_sha256": digest(config_path), "changes": record["replica_changes"],
              "new_model_calls": 0, "paid_api": False, "closed_campaign_reopened": False}
    with (state / "preparation.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--max-launches", type=int, required=True)
    parser.add_argument("--i-authorize-subscription-use", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(args.directory, args.campaign, args.max_launches, args.i_authorize_subscription_use), indent=2))
