"""Rebuild all report figures from stored real observations. No model calls."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from analyze_mechanism import analyze
from build_report import build_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    configuration = args.config.resolve() if args.config else root / 'configs/report.json'
    spec = json.loads(configuration.read_text())
    os.chdir(root)
    cache = root / 'local/report-cache'
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(cache / 'matplotlib'))
    os.environ.setdefault('XDG_CACHE_HOME', str(cache))
    report = build_report(spec['e2e'], spec['candidate'], spec['out'], spec['battery'])
    appendix = analyze(spec['e2e'], spec['candidate'], Path(spec['out']) / 'mechanism_analysis.json')
    subprocess.run([sys.executable, str(root / 'scripts/plot_overview.py'), '--candidate',
                    spec['candidate'], '--e2e', *spec['e2e'], '--out',
                    str(Path(spec['out']) / 'overview.png')], check=True)
    print(json.dumps({'report': report, 'appendix': appendix,
                      'new_model_calls': 0}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
