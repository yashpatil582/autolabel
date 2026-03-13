"""Generate markdown benchmark reports."""

from __future__ import annotations

from pathlib import Path


def generate_report(results: dict[str, list[dict]], output_path: Path) -> str:
    """Generate a markdown report from benchmark results."""
    lines = ["# AutoLabel Benchmark Results\n"]

    for ds_name, methods in results.items():
        lines.append(f"## {ds_name}\n")
        lines.append("| Method | Status | F1 | Accuracy | Coverage |")
        lines.append("|--------|--------|---:|--------:|--------:|")
        for m in methods:
            lines.append(
                f"| {m['method']} | {_status_text(m)} | {_metric_text(m.get('f1'), '.4f')} | "
                f"{_metric_text(m.get('accuracy'), '.4f')} | {_metric_text(m.get('coverage'), '.1%')} |"
            )
        lines.append("")

    report = "\n".join(lines)
    output_path.write_text(report)
    return report


def _metric_text(value: float | None, fmt: str) -> str:
    if value is None:
        return "—"
    return format(value, fmt)


def _status_text(result: dict) -> str:
    status = result.get("status")
    if status is None:
        return "completed"
    if status == "timed_out":
        return (
            f"TIMEOUT ({result.get('evaluated_examples', 0)}/{result.get('total_examples', '?')})"
        )
    if status == "skipped_budget":
        return "SKIPPED (budget exhausted)"
    return str(status)
