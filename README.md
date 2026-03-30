# AutoLabel

[![CI](https://github.com/yashpatil582/autolabel/actions/workflows/ci.yml/badge.svg)](https://github.com/yashpatil582/autolabel/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**AutoLabel is an autonomous weak-supervision system that uses an LLM to generate, validate, and iteratively improve labeling functions — with zero-label bootstrap support.**

It turns LF authoring into a self-improving optimization loop with per-LF granular scoring, agentic self-debugging, smart pruning, and inspectable Python rules. v2.0 introduces zero-label mode that achieves 0.810 test F1 without any labeled data.

Project status: beta research system focused on automated LF authoring, reproducibility, and multilingual expansion.

## Why It Matters

- Weak supervision is powerful, but manual labeling-function authoring is slow and brittle.
- AutoLabel uses an LLM to propose new LFs, validates them in an AST sandbox, and only keeps changes that improve held-out dev F1.
- The output is still inspectable Python, not a black-box-only classifier.
- The current roadmap is India-first multilingual support: Hindi first, Marathi next, then broader regional language coverage.

## Measured Results

Measured on `airline_tweets` entity extraction with 40-iteration autonomous runs using `llama-3.1-8b-instant` via Groq.

| Method | Test F1 | Setting |
|--------|---------|---------|
| Random baseline | 0.096 | Label-space random guess |
| Majority class | 0.086 | Predict most frequent train label |
| TF-IDF + LogReg | 0.784 | Supervised baseline trained on the labeled train split |
| **AutoLabel v2.0** (labeled) | **0.792** | Autonomous LF generation with per-LF scoring + pruning |
| **AutoLabel v2.0** (zero-label) | **0.810** | Zero-label bootstrap: no labeled data at all |

Key v2.0 improvements over v1.0 (0.780):
- **Per-LF granular scoring** replaces binary batch keep/discard — only the best LFs are added
- **Smart pruning** removes redundant and harmful LFs, consistently boosting F1
- **Zero-label bootstrap** generates pseudo-labels via LLM self-consistency, achieving 0.810 test F1 with no labeled data
- **Agentic self-debugging** refines low-quality LFs through multi-turn failure analysis

## Proof

### F1 Trajectory

![F1 trajectory](docs/assets/proof-f1-trajectory.png)

The granular ratchet steadily raises best dev F1 across 40 iterations, ending at `0.870` dev F1 and `0.792` test F1. Pruning at iterations 10 and 20 removes harmful LFs, boosting performance.

### Measured Baseline Comparison

![Measured baseline comparison](docs/assets/proof-baseline-comparison.png)

The benchmark chart is built from measured results only. It does not fabricate missing baselines or substitute expected values.

### LF Efficiency

![LF efficiency](docs/assets/proof-lf-efficiency.png)

v2.0 keeps only 41 active LFs from 172 generated (23.8% efficiency), compared to v1.0's 100 active from ~200 generated. Smart pruning removes redundant LFs while maintaining higher F1.

## Reproduce This Result

```bash
# Install with dev + visualization extras
pip install -e ".[dev,viz]"

# Set an LLM provider key
export GROQ_API_KEY=gsk_...

# Reproduce the autonomous run
autolabel run \
  -d airline_tweets \
  -p groq \
  -m llama-3.1-8b-instant \
  -n 40 \
  --run-name proof_v7_8b_mv_40iter

# Evaluate the completed run
autolabel evaluate experiments/proof_v7_8b_mv_40iter

# Generate measured benchmark results
autolabel benchmark \
  -d airline_tweets \
  -p groq \
  -m llama-3.1-8b-instant \
  -n 40 \
  --llm-time-budget-minutes 10

# Render publication-style charts for the run
autolabel visualize \
  experiments/proof_v7_8b_mv_40iter \
  --benchmark-results experiments/benchmark/results.json
```

The benchmark timer is optional. Classical baselines always complete; zero-shot and few-shot LLM baselines can be budget-guarded on free tiers.

## How It Works

```text
1. [Zero-label] Bootstrap pseudo-labels via LLM self-consistency (optional)
2. Analyze held-out dev failures and per-label coverage
3. Meta-learner selects strategy weighted by historical effectiveness
4. Generate candidate Python labeling functions
5. Agentic self-debugging: trial-execute, classify failures, refine (up to 3 turns)
6. Validate in AST sandbox, score each LF individually (precision, coverage, correlation)
7. Granular ratchet: greedily add only LFs that improve F1
8. Periodically prune redundant/harmful LFs
9. Ensemble label model: auto-select best aggregation (majority/weighted/generative)
10. Repeat
```

Core ingredients:

- Self-improving loop with per-LF granular scoring and smart pruning
- Agentic multi-turn LF refinement with structured failure analysis
- Zero-label bootstrap via LLM self-consistency filtering
- Weak supervision via labeling functions plus label-model ensemble
- Meta-learning across iterations for adaptive strategy selection
- Inspectable LLM-generated Python, not opaque prompting alone
- Headless-safe visualization CLI for proof artifacts

## Quick Start

```bash
pip install -e .
export GROQ_API_KEY=gsk_...

autolabel run -d airline_tweets -p groq -m llama-3.1-8b-instant -n 40
autolabel evaluate experiments/<run_dir>

# Zero-label mode (no labeled data needed)
autolabel run -d airline_tweets -p groq -m llama-3.1-8b-instant -n 40 \
  --zero-label --labels "Air Canada,Air France,Alaska Airlines,..."
```

Optional charting support:

```bash
pip install -e ".[viz]"
autolabel visualize experiments/<run_dir>
```

## Multilingual Direction

AutoLabel already includes Unicode-aware prompting and Hindi/Marathi dataset loaders. The public roadmap is to harden multilingual support in this order:

1. English + Hindi proof-quality workflows
2. Marathi expansion
3. Broader India-first regional language coverage

Example commands:

```bash
autolabel run -d hindi_headlines -p groq --language hi
autolabel run -d marathi_headlines -p groq --language mr
```

## Datasets and Provenance

AutoLabel supports one bundled quickstart dataset plus runtime-loaded HuggingFace datasets.

| Dataset | Language | Task | Access |
|---------|----------|------|--------|
| `airline_tweets` | English | Airline entity extraction | Bundled in this repository |
| `imdb` | English | Sentiment | Loaded from HuggingFace at runtime |
| `ag_news` | English | Topic classification | Loaded from HuggingFace at runtime |
| `yelp` | English | Star rating | Loaded from HuggingFace at runtime |
| `sms_spam` | English | Spam detection | Loaded from HuggingFace at runtime |
| `trec` | English | Question classification | Loaded from HuggingFace at runtime |
| `hindi_headlines` | Hindi | News classification | Loaded from HuggingFace at runtime |
| `marathi_headlines` | Marathi | News classification | Loaded from HuggingFace at runtime |

See [DATASETS.md](DATASETS.md) for provenance and redistribution notes.

## Architecture

```text
autolabel/
├── core/        autonomous loop, strategy, ratchet, agent, bootstrap, meta-learning
├── lf/          LF generation, sandbox, registry, scorer, library
├── label_model/ majority, weighted, generative EM
├── llm/         Anthropic, OpenAI, Groq, Gemini, Ollama
├── data/        dataset abstraction and loaders
├── evaluation/  metrics and evaluator
├── benchmark/   baselines, reporting, visualization
└── logging/     experiment logs and Rich output
```

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
ruff check autolabel/ tests/
ruff format --check autolabel/ tests/
```

## Cite and Contribute

- Citation metadata: [CITATION.cff](CITATION.cff)
- Contributing guide: [CONTRIBUTING.md](CONTRIBUTING.md)
- Code of conduct: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- Security policy: [SECURITY.md](SECURITY.md)
- Release notes: [v2.0.0](docs/releases/v2.0.0.md) | [v1.0.0](docs/releases/v1.0.0.md)

## References

1. Karpathy, A. (2025). *autoresearch: AI agents running research automatically*. [GitHub](https://github.com/karpathy/autoresearch)
2. Ratner et al. (2016). *Data Programming: Creating Large Training Sets, Quickly*. NeurIPS.
3. Ratner et al. (2017). *Snorkel: Rapid Training Data Creation with Weak Supervision*. VLDB.
4. Zhang et al. (2024). *Leveraging LLMs for Structure Learning in Prompted Weak Supervision*. arXiv.
5. Huang et al. (2024). *LLM-assisted Labeling Function Generation*. VLDB Workshop.

## License

MIT
