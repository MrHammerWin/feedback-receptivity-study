# Numerical source data

These files describe the closed `main-v5` study. They contain scored indices, IDs, hashes, counts and resource telemetry. They do not distribute question text, counterarguments, source quotations, model transcripts, credentials or authorization ledgers. Upstream data terms apply separately from the code's MIT licence.

## Units and filters

- QA `repeat_id=0`: 120 distinct questions, two advice conditions and five policies, yielding 1,200 policy-condition rows. The initial answer is shared; these rows are dependent.
- QA `repeat_id=1`: a fresh, preplanned repeat of the first ten of those questions, yielding 100 rows. It is a small repeat on existing questions, not ten additional independent questions. Do not pool it with the primary analysis.
- Native: 20 tasks, five variants and 100 terminal episodes. All are observed at the primary cap of 64 logical calls. Sixteen episodes ended unsuccessfully at that cap. The scored CSV has one row per cutoff, not one per episode. Cutoff 128 contains explicit `not_requested` diagnostic records: earlier successes may be carried forward as known successes, while unresolved outcomes remain missing. None is comparative or plotted; only 16, 32 and 64 were requested.
- `source_sha256` identifies the frozen Native package. QA `source_identity` combines the frozen QA protocol, manifest and package. They are different identifiers, not inconsistent hashes.

## Files

`qa_scored_numeric.csv` contains 0-based initial, final, target and advice choice indices. A missing/invalid choice is null. `initial_response_present` and `final_response_present` mean a parsed Decision object was retained, not that a physical server reply is proven. Per-policy logical costs include historical costs of reused responses. They must not be summed as the campaign's physical launch count.

`native_scored_numeric.csv`, `native_summary.csv` and `native_pairs.csv` support cutoff success and paired comparisons. `native_costs.csv` and `native_episode_costs.csv` describe full recorded trajectories, including unfinished final turns. Cached-input tokens are a subset of input tokens. CLI timing includes waiting and network delays, not GPU time.

`qa_manifest.json` publishes IDs and hashes only. Rebuild a full local manifest from the separately downloaded pinned WAFER file before using the QA runner. `native_run_plan.json` preserves the original selection hashes. `native_input_fidelity.csv` records actual initial input hashes separately: all five variants matched within each of the 20 blocks, while TextCraft distractor recipes differed from the selection process. A future run must record its own inputs.

The other QA tables describe profiles, routing, coupled requests and formatting sensitivity. `native_diagnostics.csv` describes executed commands only; probe-attempt estimates there are logical inferences, not measured physical launches. `native_command_audit.json` provides independently checked group counts for exact command transport and environment rejections. Normalized or counterfactual actions were not executed.

`posthoc_span_replay.json` reports a separate, offline sensitivity using saved responses. Five routes and one correct-advice final answer change; the primary misleading-advice result does not. It is not an online evaluation of a new agent. Full replay validation requires original private logs, scoped caches and compatible frozen source code.

`topic_sensitivity.json` keeps the two repeat IDs separate. Topic bootstrap and leave-one-topic-out summaries are descriptive, supplementing the primary exact paired test. The ten-question repeat covers eight topics and cannot support a broad clustering claim.

`campaign_summary.json` distinguishes engineering, primary inference, the optional repeat and an infrastructure-only failed attempt. Physical launch reservations include failures and were not refunded. This aggregate is not an account quota ledger.

## Verification

From the repository root, `python scripts/verify_scored_data.py --source-data source_data --json` checks numerical accuracy, retention, repair, costs, marginal intervals, all published paired comparisons and the complete four-comparison Native Holm family. It does not verify private transcript provenance, hidden environment state, the intrinsic truth of dataset labels or every diagnostic table. Separate raw-data audits were performed locally.

Add `--qa-data local/private_cache/wafer.parquet` after explicitly downloading the pinned file to bind question IDs, hashes, target and advice indices to that source. This verification makes no model call and performs no network request. See `analysis/README.md` for figure and topic-analysis commands.

`export_manifest.json` covers the curated exporter output. The separately added diagnostic summaries are covered by the repository-wide `EXPORT_MANIFEST.json`.
