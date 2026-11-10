"""Paired stock-task success and actual call cost; no published-score overlays."""
import argparse
import json
from pathlib import Path

from build_report import wilson


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', choices=['diagnostic', 'feedback'], required=True)
    parser.add_argument('--e2e', nargs='+', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    methods = [args.candidate, 'stateact', 'reflact_paper']
    labels = ['Наш агент', 'StateAct', 'ReflAct']
    records = {}
    for directory in args.e2e:
        for path in sorted((directory / 'episodes').glob('*.json')):
            row = json.loads(path.read_text())
            if row['variant'] not in methods or not row.get('finished'):
                continue
            if row['status'] == 'campaign_budget_censored':
                continue
            key = (row['benchmark'], row['key'], row['variant'])
            if key in records:
                raise ValueError('Do not mix repeated or different-version episodes')
            records[key] = row
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = ['#3569b7', '#798792', '#238a87']
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.7))
    results = []
    for column, benchmark in enumerate(['textcraft', 'alfworld']):
        sets = [{k for b, k, m in records if b == benchmark and m == method} for method in methods]
        common = set.intersection(*sets)
        if not common:
            raise ValueError('No complete paired tasks for every plotted method')
        models = {records[(benchmark, key, method)]['model'] for method in methods for key in common}
        if len(models) != 1:
            raise ValueError('Do not pool models in a matched-backbone overview')
        for index, method in enumerate(methods):
            rows = [records[(benchmark, key, method)] for key in sorted(common)]
            solved = sum(row['success'] for row in rows)
            n = len(rows)
            rate = solved / n
            low, high = wilson(solved, n)
            mean_calls = sum(row['usage']['calls'] for row in rows) / n
            axes[0, column].bar(index, rate * 100, color=colors[index], width=.62)
            axes[0, column].errorbar(index, rate * 100,
                                   yerr=[[max(0, (rate-low)*100)], [max(0, (high-rate)*100)]],
                                   fmt='none', ecolor='#17212b', capsize=5, lw=1.2)
            axes[0, column].text(index, 108, f'{solved}/{n}', ha='center', fontsize=12, weight='bold')
            axes[1, column].bar(index, mean_calls, color=colors[index], width=.62)
            axes[1, column].text(index, mean_calls+.7, f'{mean_calls:.1f}', ha='center', fontsize=12)
            results.append({'benchmark': benchmark, 'method': method, 'n': n,
                            'solved': solved, 'wilson95': [low, high], 'mean_calls': mean_calls})
        for row in range(2):
            axis = axes[row, column]
            axis.set_xticks(range(3), labels)
            axis.spines[['top', 'right']].set_visible(False)
            axis.set_axisbelow(True)
            axis.grid(axis='y', color='#e5e9ed')
        axes[0, column].set(title=benchmark.upper(), ylim=(0, 118), yticks=[0, 25, 50, 75, 100])
        axes[1, column].set_ylim(0, max(10, max(r['mean_calls'] for r in results if r['benchmark']==benchmark)*1.25))
    axes[0, 0].set_ylabel('Завершённые задачи, %\nWilson 95%')
    axes[1, 0].set_ylabel('Вызовов LLM на задачу\nСреднее, включая неуспехи')
    title = 'Диагностический контроль' if args.candidate == 'diagnostic' else 'Контроль обратной связи'
    fig.suptitle(title + ': качество и стоимость', fontsize=19, weight='bold', y=.985)
    fig.text(.5, .015, 'GPT-5.6-Sol · одинаковые задачи · предел 64 вызова на эпизод\n'
             'Сравнение адаптированных StateAct/ReflAct; это не официальный мировой leaderboard.',
             ha='center', fontsize=10, color='#465361')
    fig.tight_layout(rect=[0, .065, 1, .95], h_pad=2)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180, facecolor='white')
    fig.savefig(args.out.with_suffix('.svg'), facecolor='white')
    args.out.with_suffix('.json').write_text(json.dumps(results, ensure_ascii=False, indent=2))
    plt.close(fig)


if __name__ == '__main__':
    main()
