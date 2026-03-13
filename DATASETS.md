# Dataset Notes

This file documents how AutoLabel accesses data and what data is shipped in the repository. It is intended as a practical provenance note for contributors and users; it is not legal advice.

## Summary

| Dataset | Bundled in repo | Access mode | Provenance note | Redistribution note |
|---------|------------------|-------------|-----------------|---------------------|
| `airline_tweets` | Yes | Local JSONL under `datasets/airline_tweets/` | Project-maintained quickstart and benchmark dataset used for demos, tests, and proof visuals | Redistribute only as part of this repository and retain this file, the main license, and attribution context |
| `imdb` | No | HuggingFace runtime download | Loaded via the `datasets` library from the upstream dataset card | Follow upstream dataset terms |
| `ag_news` | No | HuggingFace runtime download | Loaded via the `datasets` library from the upstream dataset card | Follow upstream dataset terms |
| `yelp` | No | HuggingFace runtime download | Loaded via the `datasets` library from the upstream dataset card | Follow upstream dataset terms |
| `sms_spam` | No | HuggingFace runtime download | Loaded via the `datasets` library from the upstream dataset card | Follow upstream dataset terms |
| `trec` | No | HuggingFace runtime download | Loaded via the `datasets` library from the upstream dataset card | Follow upstream dataset terms |
| `hindi_headlines` | No | HuggingFace runtime download | `ai4bharat/IndicNLP-News-Articles` (`hi`) | Follow upstream dataset terms |
| `marathi_headlines` | No | HuggingFace runtime download | `ai4bharat/IndicNLP-News-Articles` (`mr`) | Follow upstream dataset terms |

## Bundled Dataset

`airline_tweets` lives at `datasets/airline_tweets/dataset_airlines.jsonl` and is packaged to keep the quickstart, benchmarks, and README proof reproducible without additional downloads.

Intended use:

- local quickstart runs
- benchmark demos
- visualization examples
- automated tests that rely on the loader contract

Practical guidance:

- Keep it in the repository when publishing AutoLabel itself.
- If you are building a downstream commercial or regulated product, replace it with your own reviewed dataset instead of treating it as production data.
- If provenance requirements in your environment are stricter than this repository note, do not reuse the bundled dataset blindly.

## Runtime-Loaded Datasets

All other datasets are fetched at runtime from HuggingFace via `autolabel/data/loaders.py`. They are not redistributed by this repository.

User responsibilities:

- review each upstream dataset card and license before use
- verify whether commercial use is allowed
- follow any attribution or distribution obligations required by the upstream source

## Maintainer Rule

When adding a new dataset:

1. Document whether it is bundled or fetched at runtime.
2. Record the upstream source or project-maintained provenance here.
3. Note any obvious redistribution constraints.
4. Avoid shipping third-party data in-tree unless its redistribution status is clear enough for a public repository.
