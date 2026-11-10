"""Fresh study authorization never expands or resets the previous campaign."""
import json

import pytest

from crmn_agent.codex_provider import reserve_call
from crmn_agent.providers import BudgetExceeded

NEW = 'feedback-nature-2026-11-10'


def test_separate_authorization_and_checkpoint_can_resume_without_reset(tmp_path):
    path = tmp_path / 'new.json'
    path.write_text(json.dumps({'campaign': NEW, 'limit': 5000}))
    checkpoint = tmp_path / 'checkpoint.json'
    checkpoint.write_text(json.dumps({'stop_after': 1}))
    assert reserve_call(path, 5000, NEW, checkpoint) == 1
    with pytest.raises(BudgetExceeded):
        reserve_call(path, 5000, NEW, checkpoint)
    checkpoint.write_text(json.dumps({'stop_after': 2}))
    assert reserve_call(path, 5000, NEW, checkpoint) == 2
    assert len(path.with_suffix('.attempts.jsonl').read_text().splitlines()) == 2


def test_default_old_campaign_cannot_use_new_ledger(tmp_path):
    path = tmp_path / 'new.json'
    path.write_text(json.dumps({'campaign': NEW, 'limit': 5000}))
    with pytest.raises(BudgetExceeded, match='Wrong campaign'):
        reserve_call(path)


def test_new_campaign_cannot_expand_old_ledger(tmp_path):
    path = tmp_path / 'old.json'
    original = json.dumps({'campaign': 'two-candidates-2026-11-08', 'limit': 1000})
    path.write_text(original)
    with pytest.raises(BudgetExceeded, match='Wrong campaign'):
        reserve_call(path, 5000, NEW)
    assert path.read_text() == original
    assert not path.with_suffix('.attempts.jsonl').exists()


@pytest.mark.parametrize('value', [None, '500', -1, 5001, True])
def test_invalid_checkpoint_fails_before_launch(tmp_path, value):
    path = tmp_path / 'new.json'
    path.write_text(json.dumps({'campaign': NEW, 'limit': 5000}))
    checkpoint = tmp_path / 'checkpoint.json'
    checkpoint.write_text(json.dumps({'stop_after': value}))
    with pytest.raises(BudgetExceeded):
        reserve_call(path, 5000, NEW, checkpoint)
    assert not path.with_suffix('.attempts.jsonl').exists()
