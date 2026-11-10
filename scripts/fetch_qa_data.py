"""Explicitly fetch the pinned WAFER data locally; never publish its contents."""

import argparse
import hashlib
import os
import tempfile
from pathlib import Path
from urllib.request import urlopen

from crmn_agent.qa_study import DOWNLOAD_URL, PARQUET_SHA256


def fetch_public():
    with urlopen(DOWNLOAD_URL, timeout=60) as response:
        data = response.read(4_000_001)
    if len(data) > 4_000_000:
        raise ValueError("Pinned QA file exceeds expected download bound")
    return data


def prepare(destination: Path, fetch=fetch_public):
    destination = Path(destination)
    if destination.is_symlink():
        raise ValueError("Refusing a symlink destination")
    if destination.exists():
        if hashlib.sha256(destination.read_bytes()).hexdigest() != PARQUET_SHA256:
            raise ValueError("Existing QA data does not match the pinned SHA256; it was not overwritten")
        return destination
    data = fetch()
    if hashlib.sha256(data).hexdigest() != PARQUET_SHA256:
        raise ValueError("Downloaded QA data differs from the pinned SHA256")
    destination.parent.mkdir(parents=True, exist_ok=True)
    pending = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
            pending = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        pending.replace(destination)
    finally:
        if pending is not None:
            pending.unlink(missing_ok=True)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('local/private_cache/wafer.parquet'))
    print(prepare(parser.parse_args().out))
