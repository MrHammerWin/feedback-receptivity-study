# Study release

The FeedbackAgent routing mechanism was frozen before the held-out experiment. Its source package fingerprint is `007ece6461170fb34900dd21417414f9221ff17246952821c17ecc3d881089c5`. Results from earlier prototypes and engineering pilots are excluded from the main estimates.

The release adds the following evaluation and reproduction work around that mechanism:

- A grounding-only control, shared natural QA drafts, fixed advice conditions and exact-request coupling within one question, condition and repeat. These separate a changed decision rule from a lucky second response to the same request.
- Separate handling of transport failures, malformed model output and resource censoring. Launch reservations count attempted processes, while verified token receipts distinguish input, cached input and output tokens.
- Paired score analysis, conservative exact intervals, missing-data checks, topic sensitivity and descriptive native diagnostics. A microsecond-level duplicate-receipt ambiguity was corrected in analysis; QA decisions and primary comparisons were unchanged.
- Numerical per-question and per-episode data, optional independent QA target verification from the pinned upstream file, and separate selection versus observed-input hashes. TextCraft's cross-process distractor variation is retained as a limitation.
- Public installation, owner-authorized subscription setup, pinned baseline-input extraction, guarded fresh-replication helpers, offline tests and figure reconstruction. The startup hash-seed guard checks CPython's actual flag in addition to the environment variable.

Portable authorization and asset-loading wrappers belong to the public distribution. They do not change the controller's prompts or routing rule. The original source-input hashes and the release-file hashes are recorded separately in `EXPORT_MANIFEST.json`.

The work evaluates a specific quotation-based feedback gate. Existing self-refinement, adaptive critique, canary controls and telemetry-guided control are prior art. CalVerT and SAG are disclosed as implemented close approaches; the release does not claim global priority, a reproduction of the original CRMN neural model, or a win over the current world SOTA.
