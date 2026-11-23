#!/usr/bin/env bash
set -euo pipefail
umask 077
if [[ $# -ne 2 || "$2" != --start ]]; then
    printf '%s\n' 'Usage: bash retests/textcraft/run_replica.sh PRIVATE_DIRECTORY --start' >&2
    exit 2
fi
state="$(cd -- "$1" && pwd)"
test "$(codex --version)" = 'codex-cli 0.147.0'
exec 9>"$state/run.lock"
flock -n 9
export PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$state/runner/src"
unset OPENAI_API_KEY CODEX_API_KEY OPENAI_BASE_URL
python -B - "$state" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
from crmn_agent.experiment import CAMPAIGN, fingerprint
state = Path(sys.argv[1])
record = json.loads((state / 'preparation.json').read_text(encoding='utf-8'))
if CAMPAIGN != record['campaign'] or fingerprint() != record['replica_source_sha256']:
    raise ValueError('The prepared campaign or source changed')
if hashlib.sha256((state / 'config.json').read_bytes()).hexdigest() != record['config_sha256']:
    raise ValueError('The prepared configuration changed')
PY
exec python -B -m crmn_agent.experiment "$state/config.json"
