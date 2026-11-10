"""Download only ALFWorld text-game assets, without visual detectors or BUTLER weights."""
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://github.com/alfworld/alfworld/releases/download/0.4.0/json_2.1.2_tw-pddl.zip'


def main():
    import requests
    target = ROOT / 'data' / 'alfworld'
    target.mkdir(parents=True, exist_ok=True)
    archive = target / 'text_games.zip'
    with requests.get(URL, stream=True, timeout=90) as response:
        response.raise_for_status()
        size = 0
        with archive.open('wb') as handle:
            for chunk in response.iter_content(1_048_576):
                size += len(chunk)
                if size > 250_000_000:
                    raise RuntimeError('Text asset download exceeds 250 MB')
                handle.write(chunk)
    count, expanded = 0, 0
    with zipfile.ZipFile(archive) as zipped:
        for info in zipped.infolist():
            if not info.filename.endswith('.tw-pddl'):
                continue
            path = target / info.filename
            if not path.resolve().is_relative_to(target.resolve()):
                raise RuntimeError('Unsafe archive member')
            expanded += info.file_size
            if expanded > 1_000_000_000:
                raise RuntimeError('Uncompressed text asset size exceeds 1 GB')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(zipped.read(info))
            count += 1
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (target / 'provenance.json').write_text(json.dumps(
        {'url': URL, 'sha256': digest, 'download_bytes': size, 'expanded_bytes': expanded, 'games': count},
        indent=2))
    archive.unlink()
    print(f'ALFWorld text assets: {count} games, {size} compressed bytes, sha256={digest}')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print('ALFWorld preparation failed:', type(exc).__name__)
        raise SystemExit(1) from None

