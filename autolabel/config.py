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
