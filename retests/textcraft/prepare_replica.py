"""Prepare a new owner-authorized replica; never launch a model or reopen a ledger."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(directory, campaign, limit, consent):
    if not consent:
        raise ValueError("Explicit new subscription-owner authorization is required")
    if (not re.fullmatch(r"replica-[A-Za-z0-9_-]{1,60}", campaign) or limit != 5120):
        raise ValueError("Use a new replica- campaign and explicitly authorize at most 5120 attempts")
    if (os.name != "posix" or sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 12)
            or os.environ.get("PYTHONHASHSEED") != "0" or sys.flags.hash_randomization != 0):
        raise ValueError("Use Linux/macOS/WSL, CPython 3.12 and PYTHONHASHSEED=0 before Python starts")
    root = Path(__file__).resolve().parent
    state = Path(directory).resolve()
    if state.exists() or state.is_relative_to(root):
        raise FileExistsError("Choose a new private directory outside the published retest folder")
    release = json.loads((root / "release_manifest.json").read_text(encoding="utf-8"))
    for relative, expected in release["files"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or digest(path) != expected:
            raise ValueError("A published input or source file changed")
    record = json.loads((root / "source_record.json").read_text(encoding="utf-8"))
    for name in ("textcraft", "pydantic", "numpy", "gymnasium"):
        if importlib.metadata.version(name) != record["recorded_environment"][name]:
            raise ValueError(f"Use the recorded runtime version of {name}")
    source = root / "frozen_runner/src"
    inputs = root / "frozen_inputs"
    # This subprocess creates game states only. It imports no Codex provider.
    preflight = """import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from crmn_agent.experiment import read_json, sha256, validate_design, verify_pilot_audit, preflight_tasks
root = Path(sys.argv[2])
tasks = read_json(root / 'manifest.json')['tasks']
verify_pilot_audit(root / 'pilot_audit.json', sha256((root / 'manifest.json').read_bytes()), validate_design(tasks))
if len(preflight_tasks(tasks)) != 40:
    raise ValueError('Forty verified starting states are required')
"""
    subprocess.run([sys.executable, "-B", "-c", preflight, str(source), str(inputs)], check=True)
    original = (source / "crmn_agent/experiment.py").read_text(encoding="utf-8")
    old = "CAMPAIGN = 'textcraft-open-commands-retest'"
    if original.count(old) != 1:
        raise ValueError("The recorded campaign assignment is not unique")
    updated = original.replace(old, "CAMPAIGN = " + repr(campaign), 1)
    compile(updated, "experiment.py", "exec")
    state.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source, state / "runner/src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(inputs, state / "inputs")
    (state / "runner/src/crmn_agent/experiment.py").write_text(updated, encoding="utf-8", newline="\n")
    sys.path.insert(0, str(state / "runner/src"))
    from crmn_agent.experiment import CAMPAIGN, MODEL, fingerprint
    from crmn_agent.public_authorization import authorize
    if CAMPAIGN != campaign:
        raise ValueError("An unexpected crmn_agent package was imported")
    new_source = fingerprint()
    if new_source == record["executed_source_sha256"]:
        raise ValueError("A new campaign must have a distinct deployment fingerprint")
    provider_path = authorize(state, campaign, limit, MODEL, True, resource_profile="recorded-study")
    provider = json.loads(provider_path.read_text(encoding="utf-8"))["provider"]
    checkpoint = state / "checkpoint.json"
    with checkpoint.open("x", encoding="utf-8") as stream:
        json.dump({"stop_after": limit}, stream)
    provider.update(subscription_checkpoint=str(checkpoint), stop_on_transport_error=True)
    config = {"provider": provider, "manifest": str(state / "inputs/manifest.json"),
              "pilot_audit": str(state / "inputs/pilot_audit.json"), "results_dir": str(state / "results")}
    config_path = state / "config.json"
    with config_path.open("x", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2)
    preparation = {"campaign": campaign, "authorized_max_attempts": limit, "new_model_calls": 0,
                   "recorded_source_sha256": record["executed_source_sha256"], "replica_source_sha256": new_source,
                   "change": "Only the CAMPAIGN string in the isolated experiment.py copy was changed.",
                   "config_sha256": digest(config_path), "public_inputs": True,
                   "replay": False, "closed_campaign_reopened": False}
    with (state / "preparation.json").open("x", encoding="utf-8") as stream:
        json.dump(preparation, stream, indent=2)
    return preparation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--max-launches", type=int, required=True)
    parser.add_argument("--i-authorize-subscription-use", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(args.directory, args.campaign, args.max_launches, args.i_authorize_subscription_use), indent=2))
