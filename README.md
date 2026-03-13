# AutoLabel

[![CI](https://github.com/yashpatil582/autolabel/actions/workflows/ci.yml/badge.svg)](https://github.com/yashpatil582/autolabel/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**An LLM autonomously generates, evaluates, and iteratively improves labeling functions — no manual labels required.**

AutoLabel combines three ideas that haven't been put together before:

| Idea | Source | What we use |
|------|--------|-------------|
| Autonomous improvement loop | [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) | Keep/discard ratchet — only keep changes that improve F1 |
| Weak supervision | [Snorkel](https://arxiv.org/abs/1711.10160) (Ratner et al.) | Labeling functions + EM-based label aggregation |
| LLM code generation | VLDB/NeurIPS 2024 papers | 8 strategies for generating Python labeling functions |

## Results

On airline entity extraction (2000 tweets, 13 airlines):

| Method | F1 | Labels needed |
|--------|----|---------------|
| Random baseline | 0.077 | — |
| TF-IDF + LogReg (supervised) | 0.310 | All |
| **AutoLabel** (llama-3.1-8b, 40 iters) | **0.656** | **Zero** |

## How It Works

```
┌──────────────────────────────────────────────────┐
│              Autonomous Loop                      │
│                                                   │
│  1. Analyze failures (per-label coverage)         │
│  2. LLM selects strategy + target label           │
│  3. LLM generates labeling functions (Python)     │
│  4. AST sandbox validates (whitelist, no I/O)     │
│  5. Apply all LFs → label matrix                  │
│  6. Label model aggregates → predictions          │
│  7. Evaluate F1 on held-out dev set               │
│  8. KEEP if F1 improved, else DISCARD             │
│  9. Repeat                                        │
└──────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Install
pip install -e .

# Set your API key (any provider works)
export GROQ_API_KEY=gsk_...        # Free — recommended for getting started
# or: export ANTHROPIC_API_KEY=sk-ant-...
# or: export OPENAI_API_KEY=sk-...

# Run on the bundled airline tweets dataset
autolabel run -d airline_tweets -p groq -m llama-3.1-8b-instant -n 40

# View results
autolabel evaluate experiments/<run_dir>/
```

## Multilingual Support

AutoLabel supports Hindi, Marathi, and other languages out of the box. Language-aware prompts guide the LLM to generate labeling functions with proper Unicode handling.

```bash
# Hindi news classification
autolabel run -d hindi_headlines -p groq --language hi

# Marathi news classification
autolabel run -d marathi_headlines -p groq --language mr
```

## Small Model Mode

Optimized for 8B parameter models and below. Adds few-shot examples, reduces output length, and runs a warmup phase:

```bash
autolabel run -d airline_tweets -p groq -m llama-3.1-8b-instant --small-model
```

## Custom Dataset

```python
from autolabel.data.dataset import AutoLabelDataset
from autolabel.core.loop import AutonomousLoop
from autolabel.llm import get_provider

dataset = AutoLabelDataset(
    name="my_task",
    task_description="Classify customer feedback sentiment",
    label_space=["positive", "negative", "neutral"],
    texts=["Great product!", "Terrible service", "It's okay"],
    labels=["positive", "negative", "neutral"],
    train_indices=[0, 1], dev_indices=[2], test_indices=[],
)

provider = get_provider("groq", model="llama-3.1-8b-instant")
loop = AutonomousLoop(dataset=dataset, provider=provider)
loop.run(max_iterations=30)
```

## Architecture

```
autolabel/
├── core/           # Autonomous loop, strategy selection, keep/discard ratchet
├── lf/             # LF generation, AST sandbox, registry, applicator
├── label_model/    # Aggregation: majority vote, weighted vote, generative EM
├── llm/            # Multi-provider: Anthropic, OpenAI, Groq, Ollama
├── data/           # Dataset abstraction and loaders (6 English + 2 Indic)
├── text/           # Unicode normalization, script detection
├── evaluation/     # Metrics, evaluator, per-LF analysis
├── benchmark/      # Baseline comparisons and reporting
└── logging/        # Experiment logging and Rich progress display
```

### LF Generation Strategies

| Strategy | Description |
|----------|-------------|
| keyword | Exact string matching |
| regex | Pattern matching with `re` |
| fuzzy | Misspellings, partial matches |
| semantic | Context clues, related terms |
| abbreviation | Codes, acronyms |
| negation | Exclusion patterns |
| context | Surrounding context, structure |
| compositional | Multi-signal combinations |

### Label Models

| Model | Description |
|-------|-------------|
| Majority Vote | Simple counting |
| Weighted Vote | Accuracy-weighted by LF agreement |
| Generative (EM) | Learns P(LF\|Y) via expectation-maximization (Snorkel-style, pure NumPy) |

### Datasets

| Dataset | Language | Task | Source |
|---------|----------|------|--------|
| airline_tweets | English | Entity extraction (13 airlines) | Bundled |
| imdb | English | Sentiment (pos/neg) | HuggingFace |
| ag_news | English | Topic (4 classes) | HuggingFace |
| yelp | English | Star rating (1-5) | HuggingFace |
| sms_spam | English | Ham vs spam | HuggingFace |
| trec | English | Question type (6 classes) | HuggingFace |
| hindi_headlines | Hindi | News classification | HuggingFace |
| marathi_headlines | Marathi | News classification | HuggingFace |

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
ruff check autolabel/ tests/
ruff format --check autolabel/ tests/
```

## How It Differs from Snorkel

| | Snorkel | AutoLabel |
|---|---------|-----------|
| LF authoring | Manual (domain experts) | Autonomous (LLM) |
| Improvement | Manual iteration | Automated keep/discard ratchet |
| Label model | EM (custom C++) | EM (pure NumPy, from scratch) |
| Cost | $50K+/year (commercial) | Free (Groq/Ollama) |
| Multilingual | Limited | Built-in (Hindi, Marathi, ...) |

## References

1. Karpathy, A. (2025). *autoresearch: AI agents running research automatically*. [GitHub](https://github.com/karpathy/autoresearch)
2. Ratner et al. (2016). *Data Programming: Creating Large Training Sets, Quickly*. NeurIPS.
3. Ratner et al. (2017). *Snorkel: Rapid Training Data Creation with Weak Supervision*. VLDB.
4. Zhang et al. (2024). *Leveraging LLMs for Structure Learning in Prompted Weak Supervision*. arXiv.
5. Huang et al. (2024). *LLM-assisted Labeling Function Generation*. VLDB Workshop.
6. Shin et al. (2024). *Stronger Than You Think: Benchmarking Weak Supervision*. NeurIPS.

## License

MIT
