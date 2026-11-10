# Reproduce analysis without model calls

The analysis separates independent questions from policy records, and native tasks from individual game steps. Replaying a saved model response does not create another independent observation.

After installing the `reports` extra, the reviewed numerical release can be checked and plotted from the repository root:

```sh
python scripts/verify_scored_data.py --source-data source_data
python analysis/topic_sensitivity.py source_data/qa_scored_numeric.csv --out topic_sensitivity.json
python analysis/plot_study.py --stats source_data/aggregated_stats.json --out rebuilt_figures --phase main --cohort main-v5 --model gpt-5.6-luna --repeat-id 0 --source-sha256 007ece6461170fb34900dd21417414f9221ff17246952821c17ecc3d881089c5 --qa-source-identity d5cf6bd22ebd1344fc545c11d5452ae8bf42326a402332008c4a7582247dbdcc --require-all-figures
```

These commands do not authenticate with a model provider or launch inference. The figure renderer writes PNG, SVG and PDF. QA protocol identity and native package source SHA use separate filters. The release command requires all four figures and fails if a selected data panel is missing. Curated SVG figures are included in the repository; rebuilding other formats is optional.

`verify_scored_data.py` recomputes scores, paired differences, exact tests, conservative intervals, missing-value handling and recorded costs from numerical per-question and per-episode rows. It checks shared draft and advice hashes when present, marginal QA/native binomial intervals, all reported QA/native paired differences and exact tests, and the complete four-comparison native Holm family at 64 calls. An incomplete planned Holm family fails full verification. Diagnostic proxy, profile, route and formatting tables are outside this numerical verifier's coverage; they require their separate raw-record audits. It can additionally verify WAFER question identities, reference targets, advice indices and answer scores against a separately downloaded pinned file with `--qa-data local/private_cache/wafer.parquet`; that optional check needs the `qa` extra and makes no network calls. Full raw-response provenance and literal quotation scoring still require the original locally retained study records.

The primary contrast is retention after misleading advice among naturally correct initial answers, comparing `controlled` with `grounding_only`. Its exact paired interval remains the primary interval. The topic bootstrap and leave-one-topic-out analysis are descriptive sensitivity checks. A bootstrap interval that collapses to zero does not establish equivalence.

Native success uses the environment's own terminal signal at 64 logical model calls. Points at 16 and 32 calls show earlier attainment. An unfinished action is excluded until all its model calls complete. Transport interruption before a cutoff is missing; failure to solve before an observed 64-call cap is a failure at that cap. No comparative result at an unrequested 128-call budget is produced.

Costs include logical calls, physical launches where separately recorded, input tokens, their cached subset, output tokens and elapsed inference time. Cached input tokens must not be added to input tokens again. Elapsed time includes CLI, queue and network overhead and is not GPU time. Individual steps and feedback strings are descriptive diagnostics, not additional independent tasks.

`build_metrics.py` analyses local raw experiment directories. Those inputs are not distributed. `native_diagnostics.py` extracts descriptive route and command diagnostics from them. The public numeric release supports arithmetic verification and figure reproduction; a fresh benchmark experiment requires separately obtained data, local authorization and a provider available to the reader.

The separate `source_data/posthoc_span_replay.json` records a post-hoc sensitivity check of exact quotation matching. Its relaxed criterion preserves every original exact match and additionally accepts a nonempty whole-answer span in the actual supplied observation prefix when it contains the selected quotation. It recombines already recorded final responses under changed routes; it is not an online run of an improved agent, an independent confirmation, a Native counterfactual or a measurement of new inference cost.

`posthoc_span_gate_replay.py` performs only read-only lookup and never creates a Codex or API provider. Repeating its full request-hash audit requires the original privately retained records, scoped cache and a compatible frozen `agent-src`; the public numeric JSON alone cannot establish raw-response authenticity. The public runtime package is not a substitute for the original frozen source fingerprint. The script deliberately follows the original layout: `qa*/outcomes.jsonl`, `qa*/model_calls.jsonl`, `qa*/run_metadata.json` and cache directories named by the scope digest. It expects the closed 120-question, two-condition, repeat-zero cohort and is not a generic replay framework. A missing or incompatible request stops the check without a model fallback or partial performance summary.

With those original inputs available locally, the audit command is:

```sh
python analysis/posthoc_span_gate_replay.py --study-root local/recorded/main-v5 --cache-root local/cache/qa-v5 --agent-src local/frozen_agent/src --out posthoc_replay_audit.json
```

The command does not request new inference. Its synthetic tests in `tests/test_posthoc_span_gate_replay.py` exercise the monotone span criterion and rejection of changed model identities, prompts, schemas, scopes, control roles, outputs and missing measurements. They require no raw study data or model access.
