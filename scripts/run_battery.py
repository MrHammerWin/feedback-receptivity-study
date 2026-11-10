"""Run the separately labelled mechanism battery with immutable model/source metadata."""
import argparse
import hashlib
import json
from pathlib import Path

from crmn_agent.experiment import dump, fingerprint
from crmn_agent.providers import make_provider

KIND = 'feedback'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    configuration = json.loads(args.config.read_text())
    provider_config = configuration.get('provider', configuration)
    if provider_config.get('backend') != 'codex':
        raise ValueError('This campaign permits only explicitly authorized subscription calls')
    if set(provider_config) & {'api_key', 'password', 'secret', 'token', 'authorization'}:
        raise ValueError('Secrets do not belong in experiment configurations')
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {'kind': KIND, 'model': provider_config['model'],
                'reasoning_effort': provider_config.get('reasoning_effort', 'low'),
                'source_sha256': fingerprint(),
                'input_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
                'provider_config_sha256': hashlib.sha256(json.dumps(provider_config, sort_keys=True).encode()).hexdigest(),
                'scope': 'Custom mechanism diagnostic, NOT stock benchmark success'}
    meta_path = args.output / 'run_metadata.json'
    if meta_path.exists() and json.loads(meta_path.read_text()) != metadata:
        raise ValueError('Refusing to mix cached calls after model, configuration, source or input changed')
    dump(meta_path, metadata)
    provider = make_provider(provider_config, args.output / 'model_calls.jsonl')
    if KIND == 'diagnostic':
        from crmn_agent.diagnostic_battery import run_battery
        result = run_battery(provider, json.loads(args.input.read_text()), args.output)
    else:
        from crmn_agent.feedback_battery import run_battery
        result = run_battery(args.input, provider, output_path=args.output / 'results.json')
    print(json.dumps(result.get('summaries', result), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
