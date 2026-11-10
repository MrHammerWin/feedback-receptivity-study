# Feedback receptivity in language agents

Research code for a runtime controller that decides how an agent should use criticism of its proposed action. A separate pair of source-quotation tasks checks whether the model can repair an incorrect quotation and preserve a correct quotation when advice is misleading. The controller then admits the critique, passes facts alone, or rebuilds the proposed action.

The quotation controls measure a narrow ability. Their success does not establish that a critique of a game action is correct. Testing whether that measurement transfers to useful action revision is the research question.

This source release includes FeedbackAgent, grounding-only and other ablations, StateAct and ReflAct reproduction adapters, interactive benchmark adapters, a frozen question-answering study, analysis code, and scripted unit tests. Empirical results and a scientific article are not included in this source-only staging export. No claim of state-of-the-art performance or global priority follows from these files.

## Offline quick start

Python 3.11 or newer is required. The tested environment uses Python 3.12. Installing packages needs network access; the following demo and test suite use scripted providers and require no credentials or model downloads.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev,reports]'
python examples/offline_demo.py
python -m pytest -q -p no:cacheprovider
python tools/check_public_tree.py .
```

The demo returns a synthetic action and explicitly marks it as a scripted example. It is not a benchmark score. Tests block socket connections to catch accidental network access.

## Implementation

The import package retains the historical name `crmn_agent` for compatibility. The implemented controller is `FeedbackAgent` in `src/crmn_agent/feedback_control.py`.

1. An actor proposes one action. A critic checks it against the public observation, task, rules and history.
2. When revision is requested, two isolated calls test quote repair and quote preservation using a public source fragment. These conversations are not added to the live task history.
3. A fixed rule chooses critique, facts-only revision, or rebuilding. Incomplete measurements take the unresolved rebuilding route.
4. One resulting command enters the environment. Native environment feedback handles unavailable commands; there is no oracle replacement with a gold action.

The default policy uses two calls for an unchanged legal draft and five for a tested revision. Control outputs, routing decisions, native action validity and resource use are recorded. Literal evidence presence is a grounding check, not a proof of a critic's reasoning. A single control pair is not a calibrated probability estimate.

The mechanism has substantial prior art in self-refinement, reflection, critique learning, selective correction and metacognitive regulation. Its proposed contribution is the particular use of isolated source-quotation controls for admission of live feedback. Novelty and benefit require evidence and comparison; the software does not establish either by itself.

The project's initial architectural inspiration was the cognitive reality monitoring network (CRMN) in Kawato and Cortese, [From internal models toward metacognitive AI](https://doi.org/10.1007/s00422-021-00904-7), Biological Cybernetics 115, 415–430 (2021). The present quotation controller is not a reproduction of that biophysical model and makes no claim about consciousness.

## Baseline inputs

Upstream prompts, demonstrations, paper HTML and benchmark data are excluded from this repository. See [THIRD_PARTY.md](THIRD_PARTY.md). The StateAct adapter uses pinned author-repository source. ReflAct is a paper-based reproduction, not an official standalone implementation.

To download and extract the recorded baseline inputs into a local cache, explicitly run:

```sh
python -m crmn_agent.asset_download
```

The downloader accepts fixed HTTPS sources, verifies pinned source hashes, parses only whitelisted static expressions without executing upstream Python, and checks all twelve generated asset hashes before writing them. ReflAct extraction is pinned to arXiv version `2505.15182v2`; upstream HTML changes cause a failure instead of a silent prompt change. A custom cache can be supplied with `--cache-dir`. Set `FEEDBACK_BASELINE_ASSETS` to that same directory for experiment loading. The default location is under the user's cache directory, outside this repository.

An explicitly supplied `assets` argument is supported for fixtures or custom comparisons. It bypasses the recorded default-cache identity check and must not be described as the pinned published baseline without independent verification.

## Experiments and reproducibility

Optional extras install benchmark adapters (`.[benchmarks]`) and the question-answering data reader (`.[qa]`). ALFWorld needs separate official game assets; TextCraft and ALFWorld licensing and setup remain upstream responsibilities. No dataset is downloaded by the offline quick start. Legacy BabyAI adapter code is retained, but its additional environment dependency and results are outside this source export.

The experiment runner accepts an explicit JSON configuration. The QA module provides `prepare` and `run` commands; use `python -m crmn_agent.qa_study --help` to inspect them. Actual benchmark runs need independently obtained data, a frozen manifest, a selected model/provider and a recorded authorization/resource budget. This release is an implementation package, not a self-contained archive of a completed empirical study.

Model providers are optional. The subscription adapter is research-specific and requires a local authorization ledger; its campaign limits are guards, not permission to spend someone else's quota. No credentials or ledgers are included. A paid API adapter is available only as an optional dependency and is not enabled by installation or tests. Do not start real-model experiments without the subscription or API owner's explicit authorization. The subscription backend records actual token use; its requested `cap` is not a strict backend output-token limit.

### Run a small experiment with your own subscription

Use an installed Codex CLI authenticated through your own ChatGPT account. Choose a model available to that account. The commands below run from the repository root on macOS or Linux. Replace `YOUR_AVAILABLE_MODEL` before running the authorization command. This creates a new local 50-launch allowance; it makes no model call and cannot overwrite or reset an earlier allowance.

The study uses Codex CLI 0.147.0. Install that version and select ChatGPT sign-in when logging in:

```sh
npm install -g @openai/codex@0.147.0
codex login
codex login status
```

```sh
python -m pip install -e '.[benchmarks,qa,reports]'
python -m crmn_agent.public_authorization --directory local/demo --campaign local-demo --max-launches 50 --model YOUR_AVAILABLE_MODEL --i-authorize-subscription-use
```

The consent flag and `allow_subscription: true` are the owner's deployment settings, not a grant from OpenAI or the research authors. The public release adds optional `authorized_campaigns` configuration for new local campaign IDs. The recorded study IDs retain their fixed 1000/5000 ceilings. This transport/setup change is separate from the studied agent mechanism and does not change its prompts or routing rule. Every attempted process counts, including failed calls; no automatic quota refund or retry occurs.

For one TextCraft task, freeze the local environment's actual task identity, then explicitly start inference:

```sh
python scripts/prepare_smoke.py --provider-config local/demo/provider.json --benchmark textcraft --seed 0 --out local/demo/native.json
python -m crmn_agent.experiment local/demo/native.json
```

The two-method smoke configuration is intentionally small. A shared limit of 50 calls may censor an episode before completion; that is a budget stop, not a completed benchmark score. To try ALFWorld instead, first obtain its official text-game assets, then freeze another configuration:

```sh
python scripts/prepare_alfworld.py
python scripts/prepare_smoke.py --provider-config local/demo/provider.json --benchmark alfworld --seed 0 --out local/demo/alfworld.json
```

The ALFWorld asset download is separate from inference. It downloads only official text-game files, with bounded archive size and path traversal checks. Review the upstream data terms first.

Alternatively, run one WAFER-derived development question and both advice conditions. The downloader verifies the pinned revision and SHA256. Raw questions, counterevidence and responses stay in ignored local paths:

```sh
python scripts/fetch_qa_data.py --out local/private_cache/wafer.parquet
python -m crmn_agent.qa_study prepare --data local/private_cache/wafer.parquet --out local/demo/qa-manifest.json
python -m crmn_agent.qa_study run --config local/demo/provider.json --data local/private_cache/wafer.parquet --manifest local/demo/qa-manifest.json --out local/demo/qa-run --split dev --limit 1
```

These alternatives share the same allowance if they use the same provider file. A new experiment does not reset it. Increasing resources requires a separate explicit decision by the subscription owner and a fresh campaign ID. Do not edit or delete the attempted-call journal to regain quota.

For stored native episodes, the existing report command makes no model calls:

```sh
python scripts/build_report.py --help
python scripts/rebuild_reports.py --help
```

These commands require completed, compatible input files. Reproduction of the article's aggregate figures will additionally require a reviewed scored-data release and its frozen analysis configuration; those are not present in this source-only staging export. The example smoke runs do not reproduce the article's empirical sample.

Exact-request replay is separated from fresh inference. Shared QA caches are scoped to the same frozen question, condition and repeat. A replayed response is not a new independent model sample. Correct-advice and misleading-advice conditions also differ in how advice is presented, so their contrast measures the complete advice packages rather than truthfulness alone.

`EXPORT_MANIFEST.json` records source and output hashes using relative filenames. `tools/check_public_tree.py` rejects raw study artifacts, baseline assets, machine-specific paths and common credential patterns. This is an export check, not a comprehensive security or licensing certification. The GitHub workflow runs the source audit and offline tests; it never launches models.

## License and provenance

Own code in this export is provided under [MIT](LICENSE). That license does not cover upstream prompts, datasets, paper text or third-party libraries. Their terms continue to apply.

Code and documentation were developed with substantial AI assistance. Human authors remain responsible for verifying the implementation, data, analysis, citations and final academic claims. AI assistance does not establish independent scientific validation.

Русское описание: [README.ru.md](README.ru.md).
