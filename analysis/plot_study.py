"""Publication figures from frozen aggregated statistics, without model calls.

Without a statistics file only the architecture figure is generated. Missing
cohorts/panels never become zero bars or substituted pilot results.
"""

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path

MM = 1 / 25.4
PALETTE = {
    "controlled": "#0072B2", "feedback": "#0072B2", "grounding_only": "#009E73",
    "always_critique": "#E69F00", "facts_only": "#CC79A7", "no_feedback": "#777777",
    "raw_actor": "#777777", "stateact": "#D55E00", "reflact_paper": "#56B4E9",
}
LABELS = {
    "controlled": "Контроллер", "feedback": "Контроллер", "grounding_only": "Только цитата",
    "always_critique": "Принять критику", "facts_only": "Только факты", "no_feedback": "Без правки",
    "raw_actor": "Актор", "stateact": "StateAct", "reflact_paper": "ReflAct",
}
QA_ORDER = ("controlled", "grounding_only", "always_critique", "facts_only", "no_feedback")
NATIVE_ORDER = ("feedback", "grounding_only", "raw_actor", "stateact", "reflact_paper")
TICK_LABELS = {**{key: value.replace(" ", "\n", 1) for key, value in LABELS.items()},
               "controlled": "Контрол-\nлер", "feedback": "Контрол-\nлер"}
_FONT_CACHE = None


def configure(deps=None):
    global _FONT_CACHE
    _FONT_CACHE = tempfile.TemporaryDirectory(prefix="feedback-figure-cache-")
    (Path(_FONT_CACHE.name) / "matplotlib").mkdir()
    (Path(_FONT_CACHE.name) / "xdg").mkdir()
    os.environ["MPLCONFIGDIR"] = str(Path(_FONT_CACHE.name) / "matplotlib")
    os.environ["XDG_CACHE_HOME"] = str(Path(_FONT_CACHE.name) / "xdg")
    if deps is not None:
        sys.path.insert(0, str(Path(deps).resolve()))
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot
    pyplot.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.2, "axes.titlesize": 9,
        "axes.labelsize": 8.2, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "axes.linewidth": 0.65, "lines.linewidth": 1.15,
        "axes.spines.top": False, "axes.spines.right": False,
        "svg.fonttype": "none", "svg.hashsalt": "feedback-study-2026-11-10",
        "pdf.fonttype": 42,
    })
    return pyplot


def architecture(plt):
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    fig, axes = plt.subplots(3, 1, figsize=(180 * MM, 158 * MM),
                             gridspec_kw={"height_ratios": [1, 1.25, 1.65]})
    fig.subplots_adjust(left=.025, right=.975, top=.975, bottom=.075, hspace=.17)
    for ax in axes:
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.axis("off")

    def box(ax, x, y, w, h, text, face="#F2F5F7", edge="#677681", fontsize=8.2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.009,rounding_size=.018",
                                   facecolor=face, edgecolor=edge, linewidth=.85))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
                linespacing=1.22)

    def arrow(ax, start, end):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=8,
                                    color="#46515A", linewidth=.9))

    a, b, c = axes
    a.text(0, .99, "А  Выбор следующего действия", fontweight="bold", va="top", fontsize=9)
    top = [(.015, "Наблюдение, цель,\nправила и история"), (.26, "Актор:\nчерновик команды"),
           (.505, "Критик:\nотдельный вызов\nтой же модели"), (.75, "Нужна правка\nчерновика?")]
    for x, text in top:
        box(a, x, .41, .225, .38, text)
    for x in (.24, .485, .73):
        arrow(a, (x, .60), (x + .018, .60))
    a.text(.02, .18, "Нет правки: исполнить черновик. Есть правка: перейти к пробам и выбору канала.",
           fontsize=8.2, va="center")
    a.text(.02, .02, "В истории остаются наблюдения и команды; критика и ответы проб не добавляются.",
           fontsize=8.2, va="bottom", color="#4C5962")

    b.text(0, .99, "Б  Изолированные пробы точности цитирования", fontweight="bold", va="top", fontsize=9)
    box(b, .025, .66, .95, .19,
        "Модель видит начало наблюдения (до 1800 символов) и выбранную строку source_fragment.", face="#EDF5FA")
    box(b, .035, .20, .44, .31, "Изменённая строка\n+ исходная строка в совете\nУдалось восстановить?",
        face="#EAF3FA", edge=PALETTE["controlled"])
    box(b, .525, .20, .44, .31, "Выбранная строка\n+ совет её изменить\nСохранил дословно?",
        face="#FFF4DF", edge="#AD7600")
    arrow(b, (.25, .65), (.25, .53))
    arrow(b, (.75, .65), (.75, .53))
    b.text(.02, .02, "Индикаторы точного совпадения строк, а не истинности высказывания или плана.",
           fontsize=8.2, va="bottom", color="#4C5962")

    c.text(0, .99, "В  Контроллер допуска и реальное действие", fontweight="bold", va="top", fontsize=9)
    route_text = [
        "Восстановил и сохранил;\nцитата присутствует\n\nДопустить критику\n\nКонтекст + черновик\n+ текст критика",
        "Пробы полны, строка сохранена;\nнет исправления или цитаты\n\nПередать только факты\n\nКонтекст + черновик\n+ цитаты наблюдения",
        "Строка изменена\nили неполное измерение\n\nНовый выбор из контекста\n\nБез текущего черновика\nи без текста критика",
    ]
    for x, text, color, face in zip((.015, .345, .675), route_text,
                                     ("#0072B2", "#009E73", "#6C6C6C"),
                                     ("#EAF3FA", "#EAF7F2", "#F0F0F0"), strict=True):
        box(c, x, .27, .31, .59, text, face=face, edge=color)
        arrow(c, (x + .155, .26), (x + .155, .20))
    box(c, .025, .015, .95, .17,
        "Одна реальная команда → штатный ответ среды → следующий шаг", face="#FFFFFF")
    fig.text(.025, .037,
             "Недоступный черновик также направляется в новый выбор. Пробы не исполняются в среде.",
             fontsize=8, color="#4C5962")
    fig.text(.025, .013,
             "В QA-опыте критик заменён фиксированным советом; начальный черновик общий для политик.",
             fontsize=8, color="#4C5962")
    return fig


def selected(rows, selection):
    result = []
    for row in rows:
        if row.get("phase") != selection["phase"] or row.get("cohort") != selection["cohort"]:
            continue
        if selection.get("model") is not None and row.get("model") != selection["model"]:
            continue
        if selection.get("repeat_id") is not None and row.get("repeat_id", 0) != selection["repeat_id"]:
            continue
        if row.get("benchmark", "qa") == "qa":
            if (selection.get("qa_source_identity") is not None
                    and row.get("source_identity") != selection["qa_source_identity"]):
                continue
        elif (selection.get("source_sha256") is not None
              and row.get("source_sha256") != selection["source_sha256"]):
            continue
        result.append(row)
    for field in ("model", "repeat_id", "source_sha256", "source_identity", "reasoning_effort"):
        values = {row[field] for row in result if row.get(field) is not None}
        if len(values) > 1:
            raise ValueError(f"Refusing to pool multiple {field} values; select one frozen protocol")
    for benchmark in {row.get("benchmark", "qa") for row in result}:
        protocols = {row["protocol_id"] for row in result
                     if row.get("benchmark", "qa") == benchmark and row.get("protocol_id") is not None}
        if len(protocols) > 1:
            raise ValueError("Refusing to combine incompatible protocols within one benchmark")
    return result


def _rate(value):
    if value is None:
        return math.nan
    if not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Rates must be finite fractions in [0, 1]")
    return float(value)


def _interval(value, estimate):
    if value is None or any(x is None for x in value):
        return None
    low, high = value
    if not all(math.isfinite(x) for x in (low, high)) or low > estimate or high < estimate:
        raise ValueError("Reported interval must bracket the displayed estimate")
    return estimate - low, high - estimate


def _finish_axis(ax, title, percent=True):
    from matplotlib.ticker import PercentFormatter
    ax.set_title(title, loc="left", fontweight="bold", pad=7)
    ax.grid(axis="y", color="#DCE3E7", linewidth=.55)
    ax.set_axisbelow(True)
    if percent:
        ax.set_ylim(0, 1.14)
        ax.yaxis.set_major_formatter(PercentFormatter(1))


def _cohort_title(fig, selection):
    prefix = "Пилот: предварительные данные" if selection["phase"] == "pilot" else "Основной опыт"
    detail = " · ".join(str(value) for value in
                        (selection.get("cohort"), selection.get("model")) if value is not None)
    fig.suptitle(f"{prefix} · {detail}", fontsize=8, color="#4C5962", y=.992)


def _bars(ax, rows, value_key, interval_key, n_key, title):
    by_mode = {row["mode"]: row for row in rows}
    if len(by_mode) != len(rows):
        raise ValueError("Duplicate QA summary cells cannot be pooled")
    modes = [mode for mode in QA_ORDER if mode in by_mode]
    for index, mode in enumerate(modes):
        row = by_mode[mode]
        value = _rate(row.get(value_key))
        if math.isfinite(value):
            ax.bar(index, value, width=.65, color=PALETTE[mode], zorder=3)
            error = _interval(row.get(interval_key), value)
            if error is not None:
                ax.errorbar(index, value, yerr=[[error[0]], [error[1]]], fmt="none", color="#283640",
                            capsize=2, linewidth=.8, zorder=4)
        else:
            ax.text(index, .08, "нет\nоценки", ha="center", color="#677681", fontsize=8)
        ax.text(index, 1.055, f"n={row[n_key]}", ha="center", fontsize=8)
    ax.set_xticks(range(len(modes)), [TICK_LABELS[mode] for mode in modes])
    _finish_axis(ax, title)


def qa_figure(plt, data, selection):
    summaries = selected(data.get("qa_summary", []), selection)
    profiles = selected(data.get("qa_profiles", []), selection)
    routes = selected(data.get("qa_routes", []), selection)
    transfer = selected(data.get("qa_proxy_transfer", []), selection)
    harmful = [row for row in summaries if row["condition"] == "false_advice"]
    helpful = [row for row in summaries if row["condition"] == "correct_advice"]
    if not harmful or not helpful or not profiles or not routes or not transfer:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(180 * MM, 162 * MM))
    fig.subplots_adjust(left=.09, right=.985, top=.915, bottom=.28, hspace=.66, wspace=.30)
    _cohort_title(fig, selection)
    _bars(axes[0, 0], harmful, "retention", "retention_interval95", "initially_correct_n",
          "А  Сохранение верного ответа\nпри ложном совете")
    _bars(axes[0, 1], helpful, "repair", "repair_interval95", "initially_wrong_n",
          "Б  Исправление ошибочного ответа\nпри верном совете")
    transfer_cells = [row for row in transfer
                      if row["condition"] == "false_advice" and row["endpoint"] == "critique_corruption"
                      and row["predictor"] == "harm_flag" and row["profile_policy"] == "controlled"
                      and row["downstream_policy"] == "always_critique"]
    by_flag = {row["flag"]: row for row in transfer_cells}
    if len(by_flag) != len(transfer_cells):
        raise ValueError("Duplicate quote-transfer cells cannot be pooled")
    matched_counts = {row["matched_case_n"] for row in transfer_cells}
    if len(matched_counts) > 1:
        raise ValueError("Quote-transfer cells have inconsistent matched denominators")
    excluded = matched_counts.pop() - sum(row["n"] for row in transfer_cells) if matched_counts else None
    if excluded is not None and excluded < 0:
        raise ValueError("Quote-transfer groups exceed their matched population")
    for index, flag in enumerate((False, True)):
        row = by_flag.get(flag)
        if row is None or not row["n"]:
            axes[1, 0].text(index, .08, "нет\nоценки", ha="center", color="#677681", fontsize=8)
            axes[1, 0].text(index, 1.055, "n=0", ha="center", fontsize=8)
            continue
        value = _rate(row["rate"])
        if not 0 <= row["events"] <= row["n"] or value != row["events"] / row["n"]:
            raise ValueError("Invalid quote-transfer event counts")
        axes[1, 0].bar(index, value, width=.55, color=PALETTE["always_critique"], zorder=3)
        error = _interval(row.get("interval95"), value)
        if error is not None:
            axes[1, 0].errorbar(index, value, yerr=[[error[0]], [error[1]]], fmt="none",
                                color="#283640", capsize=2, linewidth=.8, zorder=4)
        axes[1, 0].text(index, 1.055, f"{row['events']}/{row['n']}", ha="center", fontsize=8)
    axes[1, 0].set_xticks([0, 1], ["Сохранил\nстроку", "Изменил\nстроку"])
    axes[1, 0].set_ylabel("Порча ответа после критики")
    _finish_axis(axes[1, 0], "В  Строка пробы и порча ответа")
    chosen_routes = [row for row in routes if row["condition"] == "false_advice"]
    route_modes = [mode for mode in QA_ORDER if any(row["mode"] == mode for row in chosen_routes)]
    route_names = sorted({row["route"] for row in chosen_routes})
    route_colors = {"critique": "#0072B2", "facts_only": "#009E73", "keep_draft": "#777777",
                    "rebuild": "#E69F00", "unresolved_rebuild": "#CC79A7", "no_feedback": "#777777"}
    route_labels = {"critique": "Критика", "facts_only": "Факты", "keep_draft": "Черновик",
                    "rebuild": "Новый выбор", "unresolved_rebuild": "Нет измерения", "no_feedback": "Без правки",
                    "not_observed": "Нет записи"}
    unrecorded_modes = {row["mode"] for row in chosen_routes
                        if row["route"] == "not_observed" and row["count"]}
    if unrecorded_modes == {"no_feedback"}:
        route_labels["not_observed"] = "Без правки"
    for index, mode in enumerate(route_modes):
        rows = [row for row in chosen_routes if row["mode"] == mode]
        denominators = {row["n"] for row in rows}
        if len(denominators) != 1:
            raise ValueError("Route denominators differ within a policy")
        n = denominators.pop()
        if sum(row["count"] for row in rows) > n:
            raise ValueError("Route counts exceed the number of cases")
        bottom = 0
        for name in route_names:
            count = sum(row["count"] for row in rows if row["route"] == name)
            if n:
                height = count / n
                axes[1, 1].bar(index, height, bottom=bottom, width=.65,
                               color=route_colors.get(name, "#BBBBBB"), label=route_labels.get(name, name) if index == 0 else None)
                bottom += height
    axes[1, 1].set_xticks(range(len(route_modes)), [TICK_LABELS[mode] for mode in route_modes])
    _finish_axis(axes[1, 1], "Г  Канал исправления\nпри ложном совете")
    axes[1, 1].legend(loc="upper center", bbox_to_anchor=(.5, -.30), ncols=2, frameon=False, fontsize=8)
    excluded_label = "нет измерения" if excluded is None else str(excluded)
    fig.text(.09, .10, "В: исходно верные ответы + ложный совет; результат политики «Принять критику».", fontsize=8)
    fig.text(.09, .069, "Группы: точное сохранение выбранной строки; изменение не означает ложность цитаты.", fontsize=8)
    fig.text(.09, .038, f"Неполных профилей исключено: {excluded_label}. Частоты и точные 95% ДИ Клоппера–Пирсона.", fontsize=8)
    fig.text(.09, .007, "Связь диагностическая, не причинная и не калибровка; сверху — условные знаменатели.", fontsize=8)
    return fig


def native_figure(plt, data, selection):
    rows = [row for row in selected(data.get("native_summary", []), selection)
            if row.get("within_requested_budget") is True]
    pairs = selected(data.get("native_pairs", []), selection)
    if not rows:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(180 * MM, 151 * MM))
    fig.subplots_adjust(left=.13, right=.985, top=.92, bottom=.24, hspace=.70, wspace=.44)
    _cohort_title(fig, selection)
    for column, benchmark in enumerate(("textcraft", "alfworld")):
        ax = axes[0, column]
        benchmark_rows = [row for row in rows if row["benchmark"] == benchmark]
        if not benchmark_rows:
            ax.text(.5, .5, "Нет наблюдений", ha="center", va="center", transform=ax.transAxes, color="#677681")
        for method in NATIVE_ORDER:
            cells = sorted([row for row in benchmark_rows if row["variant"] == method],
                           key=lambda row: row["cutoff"])
            if not cells:
                continue
            if len({row["cutoff"] for row in cells}) != len(cells):
                raise ValueError("Duplicate native curve cells cannot be pooled")
            x = [row["cutoff"] for row in cells]
            y = [_rate(row["success_rate"]) for row in cells]
            ax.plot(x, y, marker="o", markersize=3, color=PALETTE[method], label=LABELS[method])
            for row, value in zip(cells, y, strict=True):
                if math.isfinite(value):
                    error = _interval(row.get("interval95"), value)
                    if error is not None:
                        ax.errorbar(row["cutoff"], value, yerr=[[error[0]], [error[1]]], fmt="none",
                                    color=PALETTE[method], linewidth=.65, capsize=1.5, alpha=.7)
        ax.set_xscale("log", base=2)
        requested_cutoffs = sorted({row["cutoff"] for row in benchmark_rows})
        ax.set_xticks(requested_cutoffs, [str(value) for value in requested_cutoffs])
        xlabel = "Логических вызовов на задачу"
        if requested_cutoffs:
            final_cells = [row for row in benchmark_rows if row["cutoff"] == requested_cutoffs[-1]]
            counts = [(row["n_observed"], row["n_started"]) for row in final_cells]
            if len(set(counts)) == 1:
                count_label = f"{counts[0][0]}/{counts[0][1]}"
            else:
                count_label = f"{min(n for n, _ in counts)}–{max(n for n, _ in counts)} / {min(n for _, n in counts)}–{max(n for _, n in counts)}"
            xlabel += f"\nУчтено / начато при {requested_cutoffs[-1]}: {count_label} на метод"
        ax.set_xlabel(xlabel)
        _finish_axis(ax, f"{'А' if column == 0 else 'Б'}  {benchmark.upper()}")
        ax = axes[1, column]
        eligible_methods = {row["variant"] for row in benchmark_rows if row["cutoff"] == 64}
        cells = [row for row in pairs if row["benchmark"] == benchmark and row["cutoff"] == 64
                 and row["a"] == "feedback" and row["a"] in eligible_methods
                 and row["b"] in eligible_methods]
        if not cells:
            ax.text(.5, .5, "Нет сопоставимых пар\nпри запрошенных 64 вызовах", ha="center", va="center",
                    transform=ax.transAxes, color="#677681", linespacing=1.5)
            ax.set_axis_off()
            continue
        for index, row in enumerate(cells):
            value = row["difference"]
            error = _interval(row.get("interval95"), value)
            ax.errorbar(value * 100, index, xerr=None if error is None else [[error[0] * 100], [error[1] * 100]],
                        fmt="o", color=PALETTE.get(row["b"], "#777777"), capsize=2, markersize=4)
            ax.text(.98, index, f"n={row['n']}", transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=8)
        ax.axvline(0, color="#7D8992", linewidth=.7, linestyle="--")
        ax.set_yticks(range(len(cells)), [LABELS.get(row["b"], row["b"]).replace("Только цитата", "Только\nцитата") for row in cells])
        ax.invert_yaxis()
        ax.set_xlim(-105, 105)
        ax.set_xlabel("Контроллер − метод, п.п.")
        ax.set_title(f"{'В' if column == 0 else 'Г'}  Парные различия\nпри 64 вызовах", loc="left", fontweight="bold")
        ax.grid(axis="x", color="#DCE3E7", linewidth=.55)
    handles, labels = [], []
    for ax in axes[0]:
        for handle, label in zip(*ax.get_legend_handles_labels(), strict=True):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .065), ncols=3, frameon=False)
    fig.text(.09, .03, "Показаны только запрошенные бюджеты; незавершённые задачи не заменяются неуспехом.", fontsize=8)
    fig.text(.09, .007, "Адаптированные методы на общей модели; это не полный мировой leaderboard.", fontsize=8)
    return fig


def _cost_estimate(row, metric):
    """Return a complete per-task estimate, never treating missing telemetry as zero."""
    started = row["n_started"]
    if metric == "tokens":
        n = row["n_complete_token_cost"]
        input_total, output_total = row.get("complete_input_tokens_total"), row.get("complete_output_tokens_total")
        total = input_total + output_total if input_total is not None and output_total is not None else None
        complete = row.get("token_cost_complete") is True
    else:
        n = row["n_logical_calls_known" if metric == "logical_calls_total" else "n_inference_time_known"]
        total = row.get(metric)
        complete = True
    if not isinstance(n, int) or not isinstance(started, int) or not 0 <= n <= started:
        raise ValueError("Cost denominators must count known observations among started tasks")
    if total is not None and (not isinstance(total, (float, int)) or not math.isfinite(total) or total < 0):
        raise ValueError("Known costs must be finite nonnegative numbers")
    value = total / n if complete and n and total is not None else None
    return value, n, started


def cost_figure(plt, data, selection):
    costs = selected(data.get("native_costs", []), selection)
    failures = selected(data.get("native_failures", []), selection)
    if not costs:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(180 * MM, 162 * MM))
    fig.subplots_adjust(left=.105, right=.985, top=.915, bottom=.28, hspace=.70, wspace=.34)
    _cohort_title(fig, selection)
    metric_specs = [
        ("logical_calls_total", "А  Логические вызовы / задачу", 1),
        ("tokens", "Б  Токены на задачу,\nвход + выход, тыс.", 1000),
        ("inference_seconds_total", "В  Время обращений / задачу, с", 1),
    ]
    for ax, (metric, title, scale) in zip(axes.flat, metric_specs, strict=False):
        for bi, benchmark in enumerate(("textcraft", "alfworld")):
            for index, method in enumerate(NATIVE_ORDER):
                cells = [row for row in costs if row["benchmark"] == benchmark and row["variant"] == method]
                if not cells:
                    continue
                if len(cells) != 1:
                    raise ValueError("Duplicate native cost cells cannot be pooled")
                row = cells[0]
                value, n, started = _cost_estimate(row, metric)
                x = index + (-.18 if bi == 0 else .18)
                if value is not None:
                    ax.bar(x, value / scale, width=.33, color=PALETTE[method],
                           alpha=.55 if bi == 0 else 1, hatch="//" if bi == 0 else None,
                           linewidth=.5, edgecolor="#FFFFFF")
                else:
                    ax.text(x, .06 + .10 * bi, "?", ha="center", transform=ax.get_xaxis_transform(),
                            fontsize=10, color="#677681")
                ax.text(index, .98 - .095 * bi, f"{'T' if bi == 0 else 'A'} {n}/{started}",
                        ha="center", va="top", transform=ax.get_xaxis_transform(), fontsize=7.3,
                        color="#4C5962")
        ax.margins(y=.28)
        ax.set_xticks(range(len(NATIVE_ORDER)), [TICK_LABELS[method] for method in NATIVE_ORDER])
        ax.set_xlim(-.6, len(NATIVE_ORDER) - .4)
        _finish_axis(ax, title, percent=False)
    ax = axes[1, 1]
    reasons = sorted({row["failure_type"] for row in failures})
    reason_labels = {"running": "Не завершено", "episode_compute_cap": "Лимит вызовов",
                     "infrastructure_censored": "Инфраструктура", "campaign_budget_censored": "Общий бюджет",
                     "action_horizon": "Лимит действий", "model_error": "Ошибка ответа",
                     "terminated": "Завершено средой"}
    colors = ("#D55E00", "#E69F00", "#CC79A7", "#777777", "#56B4E9", "#009E73")
    for bi, benchmark in enumerate(("textcraft", "alfworld")):
        for index, method in enumerate(NATIVE_ORDER):
            rows = [row for row in failures if row["benchmark"] == benchmark and row["variant"] == method]
            if not rows:
                continue
            denominators = {row["n"] for row in rows}
            if len(denominators) != 1:
                raise ValueError("Failure denominators differ within a method/environment")
            n = denominators.pop()
            if sum(row["count"] for row in rows) > n:
                raise ValueError("Stop counts exceed the number of started tasks")
            bottom = 0
            for ri, reason in enumerate(reasons):
                count = sum(row["count"] for row in rows if row["failure_type"] == reason)
                if n and count:
                    value = count / n
                    ax.bar(index + (-.18 if bi == 0 else .18), value, bottom=bottom, width=.33,
                           color=colors[ri % len(colors)], alpha=.55 if bi == 0 else 1,
                           hatch="//" if bi == 0 else None)
                    bottom += value
            ax.text(index, .98 - .095 * bi, f"{'T' if bi == 0 else 'A'} n={n}",
                    ha="center", va="top", transform=ax.get_xaxis_transform(), fontsize=7.3,
                    color="#4C5962")
    ax.set_xticks(range(len(NATIVE_ORDER)), [TICK_LABELS[method] for method in NATIVE_ORDER])
    ax.set_xlim(-.6, len(NATIVE_ORDER) - .4)
    from matplotlib.patches import Patch
    _finish_axis(ax, "Г  Остановки среди начатых задач")
    handles = [Patch(facecolor=colors[index % len(colors)], label=reason_labels.get(reason, reason))
               for index, reason in enumerate(reasons)]
    if handles:
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, -.31), ncols=2, frameon=False, fontsize=8)
    else:
        ax.text(.5, .5, "Остановок без успеха не зарегистрировано", ha="center", va="center",
                transform=ax.transAxes, color="#677681", fontsize=8)
    fig.text(.105, .11, "T: TextCraft (штриховка); A: ALFWorld (сплошной). Над столбцами: учтено / начато.", fontsize=8)
    fig.text(.105, .078, "Знак ? означает неизвестную полную стоимость. Для токенов частичные суммы не показаны.", fontsize=8)
    fig.text(.105, .046, "Кэшированные токены входят во вход. Время включает исторические обращения; это не GPU-время.", fontsize=8)
    fig.text(.105, .014, "Средние рассчитаны только по известным значениям; знаменатель зависит от показателя.", fontsize=8)
    return fig


def save_figure(plt, figure, output, slug):
    files = []
    for extension in ("svg", "pdf", "png"):
        path = output / f"{slug}.{extension}"
        metadata = {"Creator": "Feedback study plotter", "Date": None} if extension == "svg" else (
            {"Creator": "Feedback study plotter", "Author": "", "CreationDate": None, "ModDate": None}
            if extension == "pdf" else {"Software": "Feedback study plotter"}
        )
        figure.savefig(path, dpi=300, metadata=metadata, facecolor="white")
        files.append({"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    size = figure.get_size_inches() * 25.4
    plt.close(figure)
    return {"files": files, "width_mm": float(size[0]), "height_mm": float(size[1]), "png_dpi": 300}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", type=Path)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent)
    parser.add_argument("--deps", type=Path)
    parser.add_argument("--phase", default="main")
    parser.add_argument("--cohort", default="main-v2")
    parser.add_argument("--model")
    parser.add_argument("--repeat-id", type=int)
    parser.add_argument("--source-sha256", help="Native source-package SHA256; does not filter QA protocols")
    parser.add_argument("--qa-source-identity", help="QA protocol_identity; distinct from the native package SHA256")
    parser.add_argument("--require-all-figures", action="store_true",
                        help="Fail release/CI generation if any figure is waiting for compatible data")
    args = parser.parse_args()
    plt = configure(args.deps)
    args.out.mkdir(parents=True, exist_ok=True)
    selection = {"phase": args.phase, "cohort": args.cohort, "model": args.model,
                 "repeat_id": args.repeat_id, "source_sha256": args.source_sha256,
                 "qa_source_identity": args.qa_source_identity}
    report = {"schema_version": "study-figure-manifest-v1", "selection": selection,
              "plot_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "figures": {}, "waiting_for_data": []}
    report["figures"]["fig1_architecture"] = save_figure(plt, architecture(plt), args.out, "fig1_architecture")
    data = json.loads(args.stats.read_text()) if args.stats else {}
    if args.stats:
        if data.get("schema_version") != "feedback-study-metrics-v1":
            raise ValueError("Unexpected aggregated statistics schema")
        report["statistics"] = {"file": args.stats.name, "sha256": hashlib.sha256(args.stats.read_bytes()).hexdigest()}
    for slug, function in (("fig2_qa", qa_figure), ("fig3_native", native_figure), ("fig4_cost", cost_figure)):
        figure = function(plt, data, selection)
        if figure is None:
            if any((args.out / f"{slug}.{extension}").exists() for extension in ("svg", "pdf", "png")):
                raise FileExistsError("Old data figures would be stale; use a fresh output directory")
            report["waiting_for_data"].append(slug)
        else:
            report["figures"][slug] = save_figure(plt, figure, args.out, slug)
    (args.out / "figure_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.require_all_figures and report["waiting_for_data"]:
        raise SystemExit("Required figures are missing: " + ", ".join(report["waiting_for_data"]))


if __name__ == "__main__":
    main()
