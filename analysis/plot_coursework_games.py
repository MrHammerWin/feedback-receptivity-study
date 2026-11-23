"""Build the two coursework game charts from published numerical summaries."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ["feedback", "grounding_only", "raw_actor", "stateact", "reflact_paper"]
LABELS = ["FeedbackAgent", "Без двух проб", "Без критика", "StateAct", "ReflAct"]
COLORS = ["#0072B2", "#009E73", "#8C564B", "#D55E00", "#56B4E9"]
ALFWORLD_SOURCE = "007ece6461170fb34900dd21417414f9221ff17246952821c17ecc3d881089c5"


def read_panels():
    textcraft = json.loads(
        (ROOT / "retests/textcraft_baselines/source_data/summary.json").read_text(encoding="utf-8")
    )
    textcraft_rows = [
        textcraft["per_arm"][variant] for variant in ["feedback_open_commands", *VARIANTS[1:]]
    ]
    if textcraft["protocol"]["episode_call_cap"] != 32 or not all(row["n"] == 40 for row in textcraft_rows):
        raise ValueError("Expected 40 TextCraft goals per variant at 32 calls.")

    game_stats = json.loads((ROOT / "source_data/aggregated_stats.json").read_text(encoding="utf-8"))
    selected = {
        "benchmark": "alfworld",
        "phase": "main",
        "cohort": "main-v5",
        "model": "gpt-5.6-luna",
        "repeat_id": 0,
        "source_sha256": ALFWORLD_SOURCE,
    }
    scores = {
        row["variant"]: row for row in game_stats["native_summary"]
        if row["cutoff"] == 64 and all(row.get(key) == value for key, value in selected.items())
    }
    costs = {
        row["variant"]: row for row in game_stats["native_costs"]
        if all(row.get(key) == value for key, value in selected.items())
    }
    if set(scores) != set(costs) or set(scores) != set(VARIANTS) or not all(
        scores[variant]["n_started"] == scores[variant]["n_observed"] == costs[variant]["n_started"] == 12
        and scores[variant]["configured_max_calls"] == 64 for variant in VARIANTS
    ):
        raise ValueError("Expected 12 ALFWorld tasks per variant at 64 calls.")

    return [
        ("А. TextCraft · 40 целей · до 32 обращений", 40,
         [row["successes"] for row in textcraft_rows], [row["logical_calls"] for row in textcraft_rows]),
        ("Б. ALFWorld · 12 задач · до 64 обращений", 12,
         [scores[key]["n_success"] for key in VARIANTS], [costs[key]["logical_calls_total"] for key in VARIANTS]),
    ]


def render(panels, *, success_chart, filename):
    figure, axes = plt.subplots(2, 1, figsize=(6.3, 4.85))
    figure.subplots_adjust(left=.255, right=.97, top=.905, bottom=.14, hspace=.82)
    for axis, (title, tasks, successes, calls) in zip(axes, panels):
        if success_chart:
            values = [100 * count / tasks for count in successes]
            labels = [f"{count} из {tasks}" for count in successes]
            axis.set_xlim(0, 127)
            axis.set_xticks([0, 25, 50, 75, 100])
            axis.spines["bottom"].set_bounds(0, 100)
            axis.set_xlabel("Доля успешно решённых задач, %", labelpad=7)
        else:
            values = [count / tasks for count in calls]
            labels = [f"{value:.3f}".rstrip("0").rstrip(".").replace(".", ",") for value in values]
            axis.set_xlim(0, 64)
            axis.set_xticks([0, 16, 32, 48, 64])
            axis.set_xlabel("Среднее число обращений к модели на задачу", labelpad=7)

        positions = range(len(LABELS))
        axis.barh(positions, values, color=COLORS, height=.6, linewidth=0)
        axis.set_yticks(positions, LABELS)
        axis.set_ylim(len(LABELS) - .45, -.55)
        axis.set_title(title, loc="left", pad=12, fontweight="bold")
        axis.tick_params(axis="y", length=0, pad=7)
        axis.grid(axis="x", color="#DFE3E7", linewidth=.6)
        axis.set_axisbelow(True)
        for position, (value, label) in enumerate(zip(values, labels)):
            axis.text(value + (1.5 if success_chart else .9), position, label, ha="left", va="center", fontsize=10)

    figure.savefig(filename, dpi=300, facecolor="white", metadata={"Date": None})
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("rebuilt_figures"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 10.5, "axes.labelsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False, "axes.linewidth": .6,
        "svg.hashsalt": "feedbackagent-coursework-readme-v1", "svg.fonttype": "path",
    })
    panels = read_panels()
    render(panels, success_chart=True, filename=args.out / "game-success.svg")
    render(panels, success_chart=False, filename=args.out / "game-calls.svg")


if __name__ == "__main__":
    main()
