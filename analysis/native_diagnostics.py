"""Descriptive native steps; inferred logical probes are not measured launches."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def summarize(episode):
    turns = episode['trajectory']
    audits = [t['diagnostics'] for t in turns
              if isinstance(t.get('diagnostics', {}).get('critic_requested_revision'), bool)]
    profiles = [d['profile'] for d in audits if d.get('profile') is not None]
    complete = [p for p in profiles if p.get('observed_pairs') == 1]
    def rejected(turn):
        feedback = turn.get('next_observation', '').lstrip()
        if episode['benchmark'] == 'textcraft':
            return feedback.startswith('Could not')
        if episode['benchmark'] == 'alfworld':
            return feedback.lower().startswith('nothing happens')
        return False
    routes = Counter(d.get('intervention', 'unknown') for d in audits)
    result = {k: episode.get(k) for k in
              ['key', 'benchmark', 'task_id', 'seed', 'variant', 'model', 'source_sha256', 'status', 'success']}
    result.update(
        executed_commands=len(turns), critic_audited_executed_commands=len(audits),
        critic_requested_revision=sum(d['critic_requested_revision'] for d in audits),
        changed_command_string=sum(d.get('draft', {}).get('action') != d.get('final', {}).get('action') for d in audits),
        profile_records_on_executed_commands=len(profiles),
        anchored_profile_records=sum(bool(p.get('source_quote')) for p in profiles),
        complete_profiles=len(complete),
        beneficial_quote_correction=sum(p.get('beneficial_correction') == 1 for p in complete),
        quotation_corruption=sum(p.get('harmful_corruption') == 1 for p in complete),
        passed_both_quote_controls=sum(p.get('beneficial_correction') == 1 and p.get('harmful_corruption') == 0
                                      for p in complete),
        profiles_without_source_anchor=sum(not p.get('source_quote') for p in profiles),
        inferred_logical_probe_attempts_for_executed_commands=2 * sum(bool(p.get('source_quote')) for p in profiles),
        explicit_native_rejection=sum(rejected(t) for t in turns),
        repeated_observation_text=sum(t.get('observation') == t.get('next_observation') for t in turns),
        commands_outside_public_list=sum(bool(t.get('available_actions')) and t['action'] not in t['available_actions']
                                         for t in turns),
        unexecuted_turn_calls=episode.get('unexecuted_turn_calls'),
        route_keep_draft=routes['keep_draft'], route_critique=routes['critique'],
        route_facts_only=routes['facts_only'], route_rebuild=routes['rebuild'],
        route_unresolved_rebuild=routes['unresolved_rebuild'])
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('native', type=Path)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    rows = [summarize(json.loads(f.read_text())) for f in sorted(a.native.glob('**/episodes/*.json'))]
    if not rows:
        raise ValueError('No native episodes; do not fabricate empty outcomes')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({'episodes': len(rows), 'raw_observations_or_quotes_exported': False,
        'limits': ['repeated text is not a semantic loop', 'rejection uses documented native prefixes',
                   'profiles describe executed commands; unexecuted turns reported separately']}))


if __name__ == '__main__':
    main()
