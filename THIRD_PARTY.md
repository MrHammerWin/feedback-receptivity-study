# Third-party materials

The MIT license in this repository applies to its own code. No upstream prompts, worked examples, datasets, paper HTML, model weights or third-party library source are bundled in this clean export. Installing a dependency or downloading a baseline input does not change its original terms.

| Material | Primary source and pinned identity | Release treatment |
| --- | --- | --- |
| StateAct | [Author repository](https://github.com/ai-nikolai/StateAct), commit `8fb47be6f5f3978a38726608ae347987341f1d82`; [paper](https://aclanthology.org/2025.realm-1.27/) | Runtime download and static extraction only. No license file was located in the pinned repository tree; no permission to redistribute its prompts is inferred. |
| ReflAct | [Paper](https://aclanthology.org/2025.emnlp-main.1697/); [arXiv HTML version 2505.15182v2](https://arxiv.org/html/2505.15182v2) | Paper-based reproduction. The recorded arXiv snapshot uses the arXiv perpetual non-exclusive license, not a general redistribution license for this repository. HTML and extracted prompts stay in a user-managed cache. |
| ALFWorld | [Official repository](https://github.com/alfworld/alfworld); [paper](https://arxiv.org/abs/2010.03768) | Optional library and separately obtained game assets. Follow upstream software and data terms. |
| TextCraft | [ADaPT author repository](https://github.com/archiki/ADaPT) | Optional environment dependency. Source and environment assets remain upstream. |
| WAFER-QA | [Dataset](https://huggingface.co/datasets/Salesforce/WAFER-QA) | No rows or response caches included. The study requires a separately frozen dataset revision and manifest. Follow the dataset card and component-source terms. |
| Other Python libraries | Declared in `pyproject.toml` | Installed separately under their respective licenses. |

The baseline downloader records URLs, source hashes, output hashes and extraction provenance in `src/crmn_agent/baseline_sources.json`. A network or checksum failure is reported; it never silently switches to a different version. Source parsing is deliberately restricted and does not import upstream scripts.

Downloaded inputs must not be committed to this repository. Permission questions concerning upstream materials should be addressed to their rights holders. This file records provenance and distribution boundaries; it is not legal advice or a claim that all possible uses are licensed.
