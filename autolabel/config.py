"""Configuration via Pydantic Settings."""

from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


class AutoLabelConfig(BaseSettings):
    model_config = {"env_prefix": "AUTOLABEL_", "env_file": ".env", "extra": "ignore"}

    # LLM provider
    default_provider: str = Field(
        "anthropic", description="LLM provider: anthropic, openai, ollama"
    )
    default_model: str = Field("claude-sonnet-4-20250514", description="Model name")
    anthropic_api_key: str = Field("", alias="ANTHROPIC_API_KEY")
    openai_api_key: str = Field("", alias="OPENAI_API_KEY")
    groq_api_key: str = Field("", alias="GROQ_API_KEY")
    gemini_api_key: str = Field("", alias="GEMINI_API_KEY")
    ollama_host: str = Field("http://localhost:11434", alias="OLLAMA_HOST")

    # Loop parameters
    max_iterations: int = Field(50, description="Max autonomous loop iterations")
    min_improvement: float = Field(0.001, description="Minimum F1 improvement to keep new LFs")
    lfs_per_iteration: int = Field(5, description="Number of LFs to generate per iteration")
    warmup: bool = Field(True, description="Run warmup phase with simple LFs before main loop")

    # Small model optimization
    small_model_mode: bool = Field(
        False,
        description="Optimize prompts for small models (8B and below): "
        "fewer LFs per iteration, shorter max output, warmup enabled",
    )

    # Multilingual
    language: str = Field("en", description="Language code: en, hi, mr, ta, bn, ...")

    # Sandbox
    sandbox_timeout: int = Field(10, description="LF execution timeout in seconds")
    max_lf_lines: int = Field(100, description="Maximum lines per LF")

    # Granular scoring / pruning (Feature 1)
    prune_interval: int = Field(10, description="Prune redundant/harmful LFs every N iterations")
    min_lf_precision: float = Field(0.6, description="Minimum precision to keep an LF")
    max_lf_correlation: float = Field(
        0.95, description="Max correlation before pruning redundant LFs"
    )

    # Agentic self-debugging (Feature 2)
    agent_max_turns: int = Field(3, description="Max refinement turns per LF in agentic mode")
    agent_min_precision: float = Field(0.7, description="Precision target for agentic refinement")

    # Zero-label bootstrap (Feature 3)
    bootstrap_sample_size: int = Field(
        200, description="Number of texts to pseudo-label in bootstrap"
    )
    bootstrap_consistency_k: int = Field(
        3, description="Number of consistency passes for bootstrap"
    )
    bootstrap_confidence_threshold: float = Field(
        0.8, description="Confidence threshold for keeping pseudo-labels"
    )

    # Meta-learning (Feature 4)
    meta_learning: bool = Field(True, description="Enable meta-learning across iterations")

    # Label model ensemble (Feature 5)
    ensemble_label_models: bool = Field(
        True, description="Auto-select best label model per iteration"
    )

    # Cross-dataset LF transfer (Feature 6)
    lf_library_path: str = Field("", description="Path to LF library for cross-dataset transfer")

    # Paths
    project_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    experiments_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent / "experiments"
    )
    datasets_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "datasets")

    def get_experiments_dir(self, run_name: str) -> Path:
        d = self.experiments_dir / run_name
        d.mkdir(parents=True, exist_ok=True)
        return d
