"""Inspect exportable source text; never print a matched credential value."""

import argparse
import re
from pathlib import Path

SKIP = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache", "build"}
FORBIDDEN_PARTS = {"baseline_assets", "private_cache", "checkpoints", "local", "results", "models"}
PATTERNS = {
    "personal_machine_path": re.compile(r"/U" + r"sers/[^\s\"']+|/private/var/" + r"folders/"),
    "private_key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "api_credential": re.compile(r"sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{24,}|AKIA[0-9A-Z]{16}"),
}


def audit(root: Path) -> list[dict]:
    findings = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in SKIP or part.endswith(".egg-info") for part in relative.parts):
            continue
        if path.is_symlink():
            findings.append({"path": str(relative), "reason": "symlink"})
            continue
        if not path.is_file():
            continue
        if (any(part in FORBIDDEN_PARTS for part in relative.parts)
                or path.name.startswith(".env") or path.name in {"auth.json", "config.toml"}
                or path.suffix in {".jsonl", ".parquet", ".zip", ".pdf", ".png", ".pyc"}
                or (path.suffix == ".txt" and not path.name.startswith("requirements"))):
            findings.append({"path": str(relative), "reason": "private_or_third_party_asset_path"})
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            findings.append({"path": str(relative), "reason": "unexpected_binary"})
            continue
        for reason, pattern in PATTERNS.items():
            if pattern.search(text):
                findings.append({"path": str(relative), "reason": reason})
    return findings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?", default=Path("."))
    args = parser.parse_args()
    findings = audit(args.root)
    for entry in findings:
        print(entry["path"], entry["reason"])
    if findings:
        raise SystemExit(1)
    print("Public source-tree audit passed; no raw study or baseline assets included.")


if __name__ == "__main__":
    main()
