"""Offline regression cases for missing telemetry and requested-budget figures."""

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1] / "analysis"
SPEC = importlib.util.spec_from_file_location("study_figures", HERE / "plot_study.py")
plot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plot)
SELECTION = {"phase": "pilot", "cohort": "fixture", "model": "test-model", "repeat_id": 0}


def row(**values):
    return {**SELECTION, **values}


def cost(**values):
    return row(benchmark="textcraft", variant="feedback", n_started=2,
               n_complete_token_cost=0, n_logical_calls_known=1, n_inference_time_known=1,
               token_cost_complete=False, logical_calls_total=12, inference_seconds_total=30,
               input_tokens_total=None, output_tokens_total=None,
               complete_input_tokens_total=None, complete_output_tokens_total=None, **values)


@pytest.fixture(scope="module")
def plt():
    deps = None
    pyplot = plot.configure(deps)
    yield pyplot
    pyplot.close("all")


def test_null_token_telemetry_is_unknown_without_summing_none():
    assert plot._cost_estimate(cost(), "tokens") == (None, 0, 2)


def test_known_call_and_time_denominators_are_independent_of_tokens():
    assert plot._cost_estimate(cost(), "logical_calls_total") == (12, 1, 2)
    assert plot._cost_estimate(cost(), "inference_seconds_total") == (30, 1, 2)


def test_partial_token_prefix_is_not_presented_as_full_mean():
    item = cost()
    item.update(n_complete_token_cost=1, input_tokens_total=1000, output_tokens_total=100,
                complete_input_tokens_total=600, complete_output_tokens_total=70)
    assert plot._cost_estimate(item, "tokens") == (None, 1, 2)


def test_complete_token_mean_uses_complete_totals_and_count():
    item = cost()
    item.update(n_complete_token_cost=2, token_cost_complete=True,
                complete_input_tokens_total=1000, complete_output_tokens_total=100)
    assert plot._cost_estimate(item, "tokens") == (550, 2, 2)


@pytest.mark.parametrize("total", [math.nan, math.inf, -1])
def test_impossible_known_costs_fail_closed(total):
    item = cost()
    item["logical_calls_total"] = total
    with pytest.raises(ValueError, match="finite nonnegative"):
        plot._cost_estimate(item, "logical_calls_total")


def test_more_known_than_started_is_rejected():
    item = cost()
    item["n_logical_calls_known"] = 3
    with pytest.raises(ValueError, match="among started"):
        plot._cost_estimate(item, "logical_calls_total")


def test_qa_source_identities_cannot_be_pooled():
    with pytest.raises(ValueError, match="source_identity"):
        plot.selected([row(source_identity="one"), row(source_identity="two")], SELECTION)


def test_native_package_sha_does_not_hide_qa_and_explicit_qa_identity_selects_one():
    data = [row(source_identity="one"), row(source_identity="two")]
    selection = {**SELECTION, "source_sha256": "native-package", "qa_source_identity": "two"}
    assert plot.selected(data, selection) == [data[1]]
    with pytest.raises(ValueError, match="source_identity"):
        plot.selected(data, {**SELECTION, "source_sha256": "native-package"})


def test_qa_identity_does_not_filter_native_package_selection():
    data = [row(benchmark="textcraft", source_sha256="one"),
            row(benchmark="textcraft", source_sha256="two")]
    selection = {**SELECTION, "source_sha256": "two", "qa_source_identity": "qa-protocol"}
    assert plot.selected(data, selection) == [data[1]]


@pytest.mark.parametrize('required,complete', [(False, False), (True, False), (True, True)])
def test_release_cli_requires_all_figures_but_preview_allows_missing(tmp_path, monkeypatch, required, complete):
    monkeypatch.setattr(plot, 'configure', lambda deps: None)
    monkeypatch.setattr(plot, 'architecture', lambda pyplot: object())
    for name in ('qa_figure', 'native_figure', 'cost_figure'):
        monkeypatch.setattr(plot, name, lambda *args: object() if complete else None)
    monkeypatch.setattr(plot, 'save_figure', lambda *args: {'files': []})
    arguments = ['plot_study.py', '--out', str(tmp_path), '--qa-source-identity', 'synthetic-qa']
    if required:
        arguments.append('--require-all-figures')
    monkeypatch.setattr(sys, 'argv', arguments)
    if required and not complete:
        with pytest.raises(SystemExit, match='Required figures are missing'):
            plot.main()
    else:
        plot.main()
    report = json.loads((tmp_path / 'figure_manifest.json').read_text())
    assert report['selection']['qa_source_identity'] == 'synthetic-qa'
    assert set(report['figures']) | set(report['waiting_for_data']) == {
        'fig1_architecture', 'fig2_qa', 'fig3_native', 'fig4_cost'}
    assert bool(report['waiting_for_data']) is not complete


def test_unknown_cost_never_becomes_a_zero_or_nan_patch(plt):
    fig = plot.cost_figure(plt, {"native_costs": [cost()], "native_failures": []}, SELECTION)
    assert len(fig.axes[0].patches) == 1  # Known calls remain visible.
    assert len(fig.axes[1].patches) == 0  # Token cost stays unknown.
    assert len(fig.axes[2].patches) == 1  # Known inference time remains visible.
    assert any(text.get_text() == "?" for text in fig.axes[1].texts)
    assert all(math.isfinite(patch.get_height()) for ax in fig.axes for patch in ax.patches)
    plt.close(fig)


def test_completed_success_does_not_create_an_unrequested_budget_point(plt):
    common = {"benchmark": "textcraft", "variant": "feedback", "configured_max_calls": 32,
              "n_total": 1, "n_started": 1, "n_observed": 1, "n_success": 1,
              "success_rate": 1, "interval95": [.025, 1]}
    data = {"native_summary": [row(**common, cutoff=32, within_requested_budget=True),
                               row(**common, cutoff=64, within_requested_budget=False),
                               row(**common, cutoff=128, within_requested_budget=False)],
            "native_pairs": [row(benchmark="textcraft", cutoff=64, a="feedback", b="raw_actor",
                                 difference=0, interval95=[-.9, .9], n=1)]}
    fig = plot.native_figure(plt, data, SELECTION)
    assert list(fig.axes[0].lines[0].get_xdata()) == [32]
    assert list(fig.axes[0].get_xticks()) == [32]
    assert not fig.axes[2].lines
    assert any("Нет сопоставимых пар" in text.get_text() for text in fig.axes[2].texts)
    plt.close(fig)


def test_no_requested_budget_means_no_native_figure(plt):
    assert plot.native_figure(plt, {"native_summary": [row(within_requested_budget=False)]}, SELECTION) is None


def test_quote_transfer_omits_zero_n_bar_and_reports_incomplete_profiles(plt):
    summaries = [row(condition=condition, mode="controlled", retention=1,
                     retention_interval95=[.025, 1], initially_correct_n=2,
                     repair=None, repair_interval95=[None, None], initially_wrong_n=0)
                 for condition in ("false_advice", "correct_advice")]
    common = {"condition": "false_advice", "endpoint": "critique_corruption",
              "predictor": "harm_flag", "profile_policy": "controlled",
              "downstream_policy": "always_critique", "matched_case_n": 2}
    transfer = [row(**common, flag=False, n=1, events=1, rate=1, interval95=[.025, 1]),
                row(**common, flag=True, n=0, events=0, rate=None, interval95=[None, None])]
    fig = plot.qa_figure(plt, {"qa_summary": summaries, "qa_profiles": [row()],
                             "qa_routes": [row(condition="false_advice", mode="controlled",
                                               route="rebuild", n=2, count=2)],
                             "qa_proxy_transfer": transfer}, SELECTION)
    assert len(fig.axes[2].patches) == 1
    assert fig.axes[2].patches[0].get_height() == 1
    assert any(text.get_text() == "n=0" for text in fig.axes[2].texts)
    assert any("профилей исключено: 1" in text.get_text() for text in fig.texts)
    assert any("результат политики «Принять критику»" in text.get_text() for text in fig.texts)
    assert [text.get_text() for text in fig.axes[2].get_xticklabels()] == ["Сохранил\nстроку", "Изменил\nстроку"]
    assert fig.axes[2].get_ylabel() == "Порча ответа после критики"
    assert any("точное сохранение выбранной строки" in text.get_text()
               and "изменение не означает ложность цитаты" in text.get_text() for text in fig.texts)
    assert not fig.axes[1].patches  # No initially wrong drafts: no fake repair bar.
    plt.close(fig)


@pytest.mark.parametrize("mode,expected", [("no_feedback", "Без правки"), ("controlled", "Нет записи")])
def test_route_legend_distinguishes_no_feedback_by_design_from_missing_route(plt, mode, expected):
    summaries = [row(condition=condition, mode="controlled", retention=1,
                     retention_interval95=[.025, 1], initially_correct_n=1,
                     repair=None, repair_interval95=[None, None], initially_wrong_n=0)
                 for condition in ("false_advice", "correct_advice")]
    common = {"condition": "false_advice", "endpoint": "critique_corruption", "predictor": "harm_flag",
              "profile_policy": "controlled", "downstream_policy": "always_critique", "matched_case_n": 1}
    transfer = [row(**common, flag=False, n=1, events=0, rate=0, interval95=[0, .975]),
                row(**common, flag=True, n=0, events=0, rate=None, interval95=[None, None])]
    fig = plot.qa_figure(plt, {"qa_summary": summaries, "qa_profiles": [row()],
                             "qa_routes": [row(condition="false_advice", mode=mode,
                                               route="not_observed", n=1, count=1)],
                             "qa_proxy_transfer": transfer}, SELECTION)
    assert [text.get_text() for text in fig.axes[3].get_legend().get_texts()] == [expected]
    plt.close(fig)


def test_architecture_discloses_truncated_probe_observation_and_preserves_history_boundary(plt):
    fig = plot.architecture(plt)
    texts = [text.get_text() for ax in fig.axes for text in ax.texts]
    assert any("до 1800 символов" in text and "source_fragment" in text for text in texts)
    assert "В истории остаются наблюдения и команды; критика и ответы проб не добавляются." in texts
    assert "Выбранная строка\n+ совет её изменить\nСохранил дословно?" in texts
    assert any(text.startswith("Строка изменена\nили неполное измерение") for text in texts)
    assert not any("Вредная порча" in text or "Испорчено верное" in text for text in texts)
    plt.close(fig)
