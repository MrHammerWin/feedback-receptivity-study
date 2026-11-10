import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location('native_process_metrics',
    Path(__file__).resolve().parents[1] / 'analysis/native_diagnostics.py')
metrics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(metrics)


def episode(benchmark='textcraft', turns=()):
    return {'benchmark': benchmark, 'trajectory': list(turns), 'variant': 'feedback',
            'task_id': 'fixture', 'seed': 4, 'status': 'success', 'success': True}


def turn(observation='Crafted 1 brick', feedback='Crafted 1 brick', **diagnostics):
    return {'observation': observation, 'next_observation': feedback,
            'action': 'craft 1 brick using 1 clay', 'available_actions': [],
            'diagnostics': diagnostics}


def test_productive_identical_feedback_is_not_reported_as_semantic_loop():
    row = metrics.summarize(episode(turns=[turn(), turn(), turn()]))
    assert row['repeated_observation_text'] == 3
    assert row['explicit_native_rejection'] == 0
    assert not any('loop' in key for key in row)


def test_native_failure_prefix_does_not_call_successful_crafting_a_rejection():
    row = metrics.summarize(episode(turns=[turn(feedback='Could not find enough items to craft brick'), turn()]))
    assert row['explicit_native_rejection'] == 1
    assert metrics.summarize(episode('alfworld', [turn(feedback='Nothing happens.')]))['explicit_native_rejection'] == 1


def test_unanchored_profile_does_not_imply_two_model_calls():
    row = metrics.summarize(episode(turns=[turn(critic_requested_revision=True,
        profile={'source_quote': '', 'observed_pairs': 0})]))
    assert row['profile_records_on_executed_commands'] == 1
    assert row['anchored_profile_records'] == 0
    assert row['inferred_logical_probe_attempts_for_executed_commands'] == 0


def test_anchored_missing_measurement_still_attempted_the_controls():
    row = metrics.summarize(episode(turns=[turn(critic_requested_revision=True,
        profile={'source_quote': 'fixture', 'observed_pairs': 0})]))
    assert row['anchored_profile_records'] == 1
    assert row['complete_profiles'] == 0
    assert row['inferred_logical_probe_attempts_for_executed_commands'] == 2
