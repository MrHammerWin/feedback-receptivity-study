"""Fetch pinned third-party baseline inputs explicitly; never execute upstream code.

Only provenance and expected digests ship in this package. Upstream prompts and
papers remain outside the repository in a user-selected, ignored cache.
"""

import argparse
import ast
import hashlib
import json
import os
import re
import tempfile
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen


class AssetError(RuntimeError):
    pass


def catalogue():
    return json.loads(Path(__file__).with_name("baseline_sources.json").read_text(encoding="utf-8"))


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def literal_assignments(source: str) -> dict:
    values = {}
    for node in ast.parse(source).body:
        if not isinstance(node, ast.Assign):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                values[target.id] = value
    return values


def _static_text(node, values):
    """A tiny whitelist evaluator for the pinned source's string expressions."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name) and isinstance(values.get(node.id), str):
        return values[node.id]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _static_text(node.left, values) + _static_text(node.right, values)
    if isinstance(node, ast.JoinedStr):
        result = []
        for part in node.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                result.append(part.value)
            elif (isinstance(part, ast.FormattedValue) and part.conversion == -1
                  and part.format_spec is None and isinstance(part.value, ast.Name)
                  and isinstance(values.get(part.value.id), str)):
                result.append(values[part.value.id])
            else:
                raise AssetError("Unsupported upstream interpolation")
        return "".join(result)
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "join" and isinstance(node.func.value, ast.Constant)
            and isinstance(node.func.value.value, str) and len(node.args) == 1 and not node.keywords
            and isinstance(node.args[0], ast.GeneratorExp)):
        expression = node.args[0]
        if len(expression.generators) != 1:
            raise AssetError("Unsupported upstream comprehension")
        iterator = expression.generators[0]
        if not (isinstance(iterator.target, ast.Name) and not iterator.ifs and not iterator.is_async
                and isinstance(iterator.iter, ast.Call) and not iterator.iter.args
                and not iterator.iter.keywords and isinstance(iterator.iter.func, ast.Attribute)
                and iterator.iter.func.attr == "keys" and isinstance(iterator.iter.func.value, ast.Name)
                and isinstance(expression.elt, ast.Subscript)
                and isinstance(expression.elt.value, ast.Name) and isinstance(expression.elt.slice, ast.Name)
                and expression.elt.value.id == iterator.iter.func.value.id
                and expression.elt.slice.id == iterator.target.id):
            raise AssetError("Unsupported upstream join expression")
        mapping = values.get(iterator.iter.func.value.id)
        if not isinstance(mapping, dict) or not all(isinstance(v, str) for v in mapping.values()):
            raise AssetError("Expected literal string dictionary")
        return node.func.value.value.join(mapping.values())
    raise AssetError("Upstream expression is outside the static extraction whitelist")


def _branch(statements, name):
    for node in statements:
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name)
                and test.left.id == "AGENT_TYPE" and len(test.ops) == 1
                and isinstance(test.ops[0], ast.Eq) and len(test.comparators) == 1
                and isinstance(test.comparators[0], ast.Constant) and test.comparators[0].value == name):
            return node
        found = _branch(node.orelse, name)
        if found:
            return found
    return None


def extract_stateact(sources: dict[str, bytes]) -> dict[str, bytes]:
    meta = catalogue()["stateact"]
    for path, expected in meta["source_sha256"].items():
        if path not in sources or digest(sources[path]) != expected:
            raise AssetError("Pinned StateAct source digest mismatch: " + path)
    runner = sources[meta["alfworld_runner"]].decode("utf-8")
    runner_tree = ast.parse(runner)
    constants = literal_assignments(runner)
    function = next(n for n in runner_tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "generate_prompt_from_example")
    template = next(n.value for n in ast.walk(function) if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "raw_prompt" for t in n.targets))
    keys = meta["keys"]
    result = {}
    for family in meta["families"]:
        path = meta["alfworld_sources"][family]
        literals = literal_assignments(sources[path].decode("utf-8"))
        examples = ""
        for n, index in enumerate(meta["prompt_ids"]):
            if n:
                examples += "\n\n"
            for turn, content in enumerate(literals[f"{family}_v4_base_{index}"]):
                if turn % 2 == 0:
                    examples += content + "\n\n"
                else:
                    state = json.loads(content)
                    lines = []
                    for key in keys:
                        value = state[key]
                        if not value:
                            value = "None"
                        elif isinstance(value, list):
                            value = ", ".join(value)
                        lines.append(f"{key.replace('_', ' ')}: {value}")
                    examples += ">" + "\n".join(lines) + "\n\n"
        context = {**constants, "base_prompt": constants["BASE_PROMPT1"],
                   "instructions": constants["INSTRUCTIONS"], "hints": constants["HINTS"],
                   "is_are": "are", "number_of_examples_string": str(len(meta["prompt_ids"])),
                   "s_string": "s", "examples": examples, "task_message": constants["TASK_MESSAGE"]}
        result[f"stateact_alfworld_{family}_system.txt"] = _static_text(template, context).encode()
    craft = sources[meta["textcraft_runner"]].decode("utf-8")
    tree = ast.parse(craft)
    values = literal_assignments(craft)
    branch = _branch(tree.body, "stateact")
    if branch is None:
        raise AssetError("Pinned StateAct branch missing")
    prompt = values["atomic_exec_prompt"]
    for node in branch.body:
        if (isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name)
                and node.target.id == "atomic_exec_prompt" and isinstance(node.op, ast.Add)):
            prompt += _static_text(node.value, values)
    tail = [n for n in tree.body if isinstance(n, ast.AugAssign)
            and isinstance(n.target, ast.Name) and n.target.id == "atomic_exec_prompt"
            and n.lineno > branch.lineno]
    if len(tail) != 1:
        raise AssetError("Pinned StateAct prompt suffix is ambiguous")
    prompt += _static_text(tail[0].value, values)
    result["stateact_textcraft_system.txt"] = prompt.encode()
    result["stateact_textcraft_environment_context.txt"] = values["environment_context"].encode()
    return result


class _PaperText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in ("p", "section", "br", "div", "h2", "h3", "h4"):
            self.parts.append("\n")

    def handle_data(self, data):
        self.parts.append(data)


def extract_reflact(source: bytes) -> dict[str, bytes]:
    parser = _PaperText()
    parser.feed(source.decode("utf-8"))
    text = "".join(parser.parts)
    if catalogue()["reflact"]["version"] not in text:
        raise AssetError("Expected pinned ReflAct paper version")

    def section(start, end):
        first = text.find(start)
        last = text.find(end, first + len(start))
        if first < 0 or last < 0:
            raise AssetError("ReflAct appendix anchors not found")
        return text[first:last]

    block = section("Instruction for ALFWorld Nothinking, ReAct, ReflAct", "Figure 14:")
    block = block[block.index("Interact with a household"):].strip()
    first, last = block.index("For each of your turn,"), block.index("The available actions are:")
    match = re.search(r"\(ReflAct\) (.*?)(?=\n\nThe available actions)", block, re.S)
    if match is None:
        raise AssetError("ReflAct instruction variant missing")
    result = {"reflact_alfworld_instruction.txt":
              (block[:first] + match.group(1).strip() + "\n\n" + block[last:] + "\n").encode()}
    for family, label, end in (("put", "pick and place", "Figure 16:"),
                               ("clean", "pick clean then place", "Figure 18:")):
        value = section(f'Prompt for ALFWorld ReflAct, "{label}" task', end)
        value = value[value.index("You are in the middle"):].strip()
        result[f"reflact_alfworld_{family}_example.txt"] = (value + "\n").encode()
    core = re.search(r"You should first reflect[^.]+and then output[^.]+\.", text)
    if core is None:
        raise AssetError("ReflAct transfer instruction missing")
    result["reflact_core_instruction.txt"] = (core.group() + "\n").encode()
    return result


def validate_asset_bytes(assets):
    expected = catalogue()["assets"]
    for name, content in assets.items():
        if name not in expected or digest(content) != expected[name]:
            raise AssetError("Extracted baseline asset differs from recorded research version: " + name)


def fetch_public(url: str) -> bytes:
    allowed = {"raw.githubusercontent.com", "arxiv.org", "www.arxiv.org"}
    if urlparse(url).scheme != "https" or urlparse(url).hostname not in allowed:
        raise AssetError("Unexpected baseline source URL")
    with urlopen(url, timeout=60) as response:
        if urlparse(response.geturl()).hostname not in allowed:
            raise AssetError("Unexpected baseline redirect")
        value = response.read(4_000_001)
    if len(value) > 4_000_000:
        raise AssetError("Baseline source exceeds download bound")
    return value


def default_cache() -> Path:
    override = os.environ.get("FEEDBACK_BASELINE_ASSETS")
    if override:
        return Path(override)
    base = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    return base / "feedback-receptivity-agent" / "baseline-inputs-v1"


def verify_cache(directory: Path) -> None:
    for name, expected in catalogue()["assets"].items():
        path = directory / name
        if not path.is_file() or digest(path.read_bytes()) != expected:
            raise AssetError("Baseline inputs are missing or differ. Run python -m crmn_agent.asset_download.")


def resolve_assets(explicit=None) -> Path:
    # Explicit fixtures/custom assets are caller-owned; they are not claimed to
    # be the pinned baseline. The default research cache is always verified.
    if explicit is not None:
        return Path(explicit)
    directory = default_cache()
    verify_cache(directory)
    return directory


def prepare_assets(directory=None, fetch=fetch_public) -> Path:
    directory = Path(directory) if directory is not None else default_cache()
    metadata = catalogue()
    state = metadata["stateact"]
    sources = {path: fetch(state["raw_prefix"] + path) for path in state["source_sha256"]}
    generated = extract_stateact(sources)
    generated.update(extract_reflact(fetch(metadata["reflact"]["url"])))
    validate_asset_bytes(generated)
    if set(generated) != set(metadata["assets"]):
        raise AssetError("Incomplete baseline extraction")
    directory.mkdir(parents=True, exist_ok=True)
    for name, content in generated.items():
        destination = directory / name
        if destination.is_symlink():
            raise AssetError("Refusing to replace a baseline-cache symlink")
        with tempfile.NamedTemporaryFile(dir=directory, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        temporary.replace(destination)
    verify_cache(directory)
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    path = prepare_assets(args.cache_dir)
    print("Verified baseline inputs prepared in", path)


if __name__ == "__main__":
    main()
