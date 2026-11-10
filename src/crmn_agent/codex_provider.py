"""Explicitly authorized ChatGPT-subscription backend, isolated from repository tools."""
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from .providers import BackendTransportError, BudgetExceeded, ModelOutputError, Provider
from .public_authorization import campaign_maximum

CAMPAIGN = 'two-candidates-2026-11-08'
AUTHORIZED_CAMPAIGNS = {CAMPAIGN: 1000, 'feedback-nature-2026-11-10': 5000}


def failure_category(text):
    """Only fixed category names leave the provider, never raw stderr/URLs/keys."""
    for category, pattern in (
        ('rate_limit', r'\b429\b|rate.limit|too many requests'),
        ('usage_limit', r'usage.limit|quota|limit.reached'),
        ('authentication', r'401|unauthorized|authentication|token.refresh|expired.token'),
        ('schema', r'invalid.schema|output.schema'),
        ('network', r'connect|network|socket|502|503|504|stream.disconnected'),
    ):
        if re.search(pattern, text, re.IGNORECASE):
            return category
    return 'unclassified_backend_failure'


def completion_usage(events):
    """Require actual CLI telemetry; missing fields must not become free tokens."""
    usage = [e.get('usage') for e in events if e.get('type') == 'turn.completed']
    keys = ('input_tokens', 'output_tokens', 'cached_input_tokens')
    if not usage or any(not isinstance(u, dict) for u in usage):
        raise ModelOutputError('Codex completion is missing usage evidence')
    for item in usage:
        if any(type(item.get(key)) is not int or item[key] < 0 for key in keys):
            raise ModelOutputError('Codex completion has invalid token telemetry')
        if item['cached_input_tokens'] > item['input_tokens']:
            raise ModelOutputError('Cached input cannot exceed total input tokens')
    return {key: sum(item[key] for item in usage) for key in keys}


def reserve_call(path: Path, authorized_limit: int = 1000, campaign: str = CAMPAIGN,
                 checkpoint: Path | None = None, min_interval_seconds: float = 0,
                 authorized_campaigns: dict | None = None) -> int:
    """Reserve BEFORE starting a process; failures also consume the approved allowance.

    The pre-existing local ledger is deliberately not created by the provider. Both
    independent repositories must point to the SAME ledger for this campaign.
    An append-only journal survives a crash without refunding an attempted call.
    """
    maximum = campaign_maximum(campaign, AUTHORIZED_CAMPAIGNS, authorized_campaigns)
    if (isinstance(min_interval_seconds, bool) or not isinstance(min_interval_seconds, (int, float))
            or not 0 <= min_interval_seconds <= 60):
        raise ValueError('Launch spacing must be between zero and sixty seconds')
    if maximum is None:
        raise BudgetExceeded('Unrecognized campaign; a new explicit authorization is required')
    if not 0 < authorized_limit <= maximum:
        raise ValueError(f'This campaign authorized at most {maximum} subscription launches')
    if not path.is_file():
        raise BudgetExceeded('A separately authorized local campaign ledger is required')
    with path.open('r+', encoding='utf-8') as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        data = json.load(handle)
        if data.get('campaign') != campaign or data.get('limit') != maximum:
            raise BudgetExceeded('Wrong campaign or unrecognized authorization')
        journal = path.with_suffix('.attempts.jsonl')
        records = journal.read_text().splitlines() if journal.exists() else []
        used = len([line for line in records if line.strip()])
        limit = min(data['limit'], authorized_limit)
        if checkpoint is not None:
            if not checkpoint.is_file():
                raise BudgetExceeded('The configured study checkpoint file is missing')
            stop = json.loads(checkpoint.read_text()).get('stop_after')
            if not isinstance(stop, int) or isinstance(stop, bool) or not 0 <= stop <= maximum:
                raise BudgetExceeded('Invalid checkpoint; no generation was launched')
            limit = min(limit, stop)
        if used >= limit:
            raise BudgetExceeded('The persistent subscription launch allowance is exhausted')
        if min_interval_seconds and records:
            from datetime import datetime
            last = json.loads(records[-1]).get('reserved_at')
            if not isinstance(last, str):
                raise BudgetExceeded('Launch spacing requires timestamped reservation evidence')
            wait = min_interval_seconds - (time.time() - datetime.fromisoformat(last).timestamp())
            if wait > 60:
                raise BudgetExceeded('Reservation clock is inconsistent; no generation was launched')
            if wait > 0:
                time.sleep(wait)
            if checkpoint is not None and used >= json.loads(checkpoint.read_text())['stop_after']:
                raise BudgetExceeded('Study stopped while waiting for the next launch slot')
        with journal.open('a', encoding='utf-8') as attempts:
            from datetime import UTC, datetime
            attempts.write(json.dumps({'launch': used + 1, 'campaign': campaign,
                                       'reserved_at': datetime.now(UTC).isoformat()}) + '\n')
            attempts.flush()
            os.fsync(attempts.fileno())
        return used + 1


class CodexProvider(Provider):
    def __init__(self, config: dict, call_log: Path | None = None):
        super().__init__(config, call_log)
        if not config.get('allow_subscription', False):
            raise BudgetExceeded('Subscription use has not been explicitly enabled')
        if not shutil.which('codex'):
            raise RuntimeError('Codex CLI is not installed')
        self.process_env = dict(os.environ)
        for name in ['OPENAI_API_KEY', 'CODEX_API_KEY', 'OPENAI_BASE_URL']:
            self.process_env.pop(name, None)
        status = subprocess.run(['codex', 'login', 'status'], capture_output=True, text=True,
                                env=self.process_env, timeout=20)
        if status.returncode or 'ChatGPT' not in (status.stdout + status.stderr):
            raise RuntimeError('A ChatGPT sign-in, not API-key authentication, is required')
        self.root = Path(config.get('isolated_dir', 'local/codex-isolated')).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.ledger = Path(config['subscription_ledger']).resolve()

    def _transport_failure(self, category, details=None):
        self.last_backend_failure = {'category': category, **(details or {})}
        if self.config.get('stop_on_transport_error'):
            checkpoint = Path(self.config['subscription_checkpoint'])
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=checkpoint.parent,
                                             prefix='.circuit-', delete=False) as stream:
                temporary = Path(stream.name)
                json.dump({'stop_after': 0, 'reason': category}, stream)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(checkpoint)
        raise BackendTransportError(category)

    def _generate(self, system, user, schema, cap):
        self.last_launch = reserve_call(self.ledger, self.config.get('subscription_max_launches', 1000),
                                        self.config.get('subscription_campaign', CAMPAIGN),
                                        Path(self.config['subscription_checkpoint'])
                                        if self.config.get('subscription_checkpoint') else None,
                                        self.config.get('subscription_min_interval_seconds', 0),
                                        self.config.get('authorized_campaigns'))
        with tempfile.TemporaryDirectory(prefix='call-', dir=self.root) as name:
            directory = Path(name)
            schema_path, output_path = directory/'schema.json', directory/'response.json'
            schema_path.write_text(json.dumps(schema.model_json_schema()))
            arguments = [
                'codex', '-a', 'never', 'exec', '--ignore-user-config', '--ignore-rules', '--ephemeral',
                '--skip-git-repo-check', '--sandbox', 'read-only', '--color', 'never', '--json',
                '--disable', 'shell_tool', '--disable', 'apps',
                '-c', 'web_search="disabled"', '-c', 'mcp_servers={}',
                '-c', 'project_doc_max_bytes=0', '-c',
                'model_reasoning_effort=' + json.dumps(self.config.get('reasoning_effort', 'low')),
                '--model', self.model, '--cd', str(directory),
                '--output-schema', str(schema_path), '--output-last-message', str(output_path), '-',
            ]
            domain = 'question-answering' if self.config.get('task_kind') == 'qa' else 'text-game'
            prompt = (
                f'This is an isolated {domain} experiment. Return JSON using ONLY the supplied data. '
                'Do not call tools, run commands, read files, inspect the workspace or use the internet. '
                'Do not perform coding work.\n' + system + '\nINPUT:\n' + user
            )
            try:
                result = subprocess.run(arguments, input=prompt, capture_output=True, text=True,
                                        env=self.process_env, timeout=self.config.get('timeout_seconds', 180))
            except subprocess.TimeoutExpired as exc:
                try:
                    self._transport_failure('timeout')
                except BackendTransportError as error:
                    raise error from exc
            except OSError:
                self._transport_failure('local_execution')
            events = []
            for line in result.stdout.splitlines():
                try:
                    event = json.loads(line)
                    if isinstance(event, dict):
                        events.append(event)
                except ValueError:
                    continue
            forbidden = {'command_execution', 'mcp_tool_call', 'web_search', 'file_change', 'tool_call'}
            if any(isinstance(e.get('item'), dict) and e['item'].get('type') in forbidden for e in events):
                raise ModelOutputError('Rejected: Codex used tools instead of the supplied observations')
            if result.returncode or not output_path.exists():
                errors = [e for e in events if e.get('type') in {'error', 'turn.failed'}]
                self._transport_failure(failure_category(result.stderr + '\n' + json.dumps(errors)), {
                    'returncode': result.returncode,
                    'stderr_sha256': hashlib.sha256(result.stderr.encode()).hexdigest(),
                    'stdout_sha256': hashlib.sha256(result.stdout.encode()).hexdigest(),
                    'event_count': len(events),
                })
            try:
                self.last_backend_usage = completion_usage(events)
            except ModelOutputError:
                self._transport_failure('invalid_usage_telemetry')
            inputs = self.last_backend_usage['input_tokens']
            outputs = self.last_backend_usage['output_tokens']
            return output_path.read_text(), inputs, outputs

