"""JSON-lines experiment logger."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class ExperimentLogger:
    """Logs iteration results as JSON-lines for reproducibility."""

    def __init__(self, log_dir: Path) -> None:
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "experiment.jsonl"
        self.meta_file = self.log_dir / "meta.json"
        self._start_time = time.time()

    def log_meta(self, meta: dict[str, Any]) -> None:
        meta["start_time"] = self._start_time
        self.meta_file.write_text(json.dumps(meta, indent=2, default=str))

    def log_iteration(self, iteration: int, result: dict[str, Any]) -> None:
        entry = {
            "iteration": iteration,
            "timestamp": time.time(),
            "elapsed_s": time.time() - self._start_time,
            **result,
        }
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def log_final(self, summary: dict[str, Any]) -> None:
        summary["total_time_s"] = time.time() - self._start_time
        final_file = self.log_dir / "final_summary.json"
        final_file.write_text(json.dumps(summary, indent=2, default=str))

    def read_iterations(self) -> list[dict[str, Any]]:
        if not self.log_file.exists():
            return []
        entries = []
        for line in self.log_file.read_text().strip().split("\n"):
            if line:
                entries.append(json.loads(line))
        return entries
