# Changelog

## [1.0.0] - 2026-03-13

### Added
- **Autonomous improvement loop** — Karpathy-style keep/discard ratchet for iterative F1 improvement
- **8 LF generation strategies** — keyword, regex, fuzzy, semantic, abbreviation, negation, context, compositional
- **3 label models** — majority vote, weighted vote, generative EM (pure NumPy)
- **4 LLM providers** — Anthropic (Claude), OpenAI (GPT), Groq (free), Ollama (local)
- **8 datasets** — airline tweets (bundled), IMDB, AG News, Yelp, SMS Spam, TREC, Hindi headlines, Marathi headlines
- **Multilingual support** — Hindi, Marathi with language-aware prompts and Devanagari handling
- **Small model mode** — few-shot examples, warmup phase, retry logic for 8B models
- **AST sandbox** — whitelist-based validation for safe LF execution
- **CLI** — `autolabel run`, `autolabel benchmark`, `autolabel evaluate`, `autolabel cost`, `autolabel visualize`
- **Cost tracking** — per-call token and USD accounting
- **Publication-style charts** — F1 trajectory, baseline comparison, strategy analysis, coverage/accuracy, LF efficiency
- **Free-tier benchmark guardrails** — shared LLM time budget plus per-request timeout support for benchmark baselines
- **CI pipeline** — GitHub Actions with Python 3.11/3.12 matrix
- **Measured proof run** — 0.656 test F1 on airline entity extraction with `llama-3.1-8b-instant`
- **Launch docs** — citation metadata, dataset notes, security policy, release notes, and GitHub templates
