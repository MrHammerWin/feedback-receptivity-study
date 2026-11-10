"""Portable owner configuration for the public release, not a provider grant."""

import argparse
import json
import os
import re
from pathlib import Path

from .providers import BudgetExceeded


def campaign_maximum(campaign: str, recorded: dict, extra: dict | None = None):
    """Fixed recorded ceilings remain fixed; new local campaigns need explicit data."""
    if extra is None:
        return recorded.get(campaign)
    if not isinstance(extra, dict) or not extra:
        raise BudgetExceeded("Owner-configured campaigns must be a nonempty dictionary")
    for name, limit in extra.items():
        if (not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", name)
                or type(limit) is not int or limit < 1):
            raise BudgetExceeded("Invalid owner-configured campaign name or launch limit")
        if name in recorded and recorded[name] != limit:
            raise BudgetExceeded("Recorded research campaign ceilings cannot be overridden")
    return recorded.get(campaign, extra.get(campaign))


def authorize(directory: Path, campaign: str, limit: int, model: str, consent: bool):
    """Create a fresh local deployment; never overwrite/reset an earlier allowance."""
    if not consent:
        raise BudgetExceeded("Explicit subscription-owner consent is required")
    # Import only at call time; importing this module never authenticates or runs a model.
    from .codex_provider import AUTHORIZED_CAMPAIGNS

    campaign_maximum(campaign, AUTHORIZED_CAMPAIGNS, {campaign: limit})
    if campaign in AUTHORIZED_CAMPAIGNS:
        raise ValueError("Choose a new local campaign, not a recorded research campaign")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("Supply a model that is available to the subscription owner")
    directory = Path(directory)
    ledger, config = directory / "allowance.json", directory / "provider.json"
    if ledger.exists() or ledger.with_suffix('.attempts.jsonl').exists() or config.exists():
        raise FileExistsError("Refusing to overwrite authorization, configuration or attempted calls")
    directory.mkdir(parents=True, exist_ok=True)
    values = {"backend": "codex", "model": model, "allow_subscription": True, "allow_paid": False,
              "subscription_campaign": campaign, "subscription_max_launches": limit,
              "subscription_ledger": str(ledger), "authorized_campaigns": {campaign: limit},
              "isolated_dir": str(directory / "isolated"), "study_cache_dir": str(directory / "cache"),
              "reasoning_effort": "low", "timeout_seconds": 180}
    for path, value in ((ledger, {"campaign": campaign, "limit": limit,
                                 "authorization": "Explicitly authorized by the local subscription owner.",
                                 "paid_api": False}), (config, {"provider": values})):
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--max-launches", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--i-authorize-subscription-use", action="store_true")
    args = parser.parse_args()
    print(authorize(args.directory, args.campaign, args.max_launches, args.model,
                    args.i_authorize_subscription_use))


if __name__ == "__main__":
    main()
