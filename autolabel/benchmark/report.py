"""Generate markdown benchmark reports."""

from __future__ import annotations

from pathlib import Path


def generate_report(results: dict[str, list[dict]], output_path: Path) -> str:
    """Generate a markdown report from benchmark results."""
    lines = ["# AutoLabel Benchmark Results\n"]

    for ds_name, methods in results.items():
        lines.append(f"## {ds_name}\n")
        lines.append("| Method | F1 | Accuracy | Coverage |")
        lines.append("|--------|---:|--------:|--------:|")
        for m in methods:
            lines.append(
                f"| {m['method']} | {m['f1']:.4f} | "
                f"{m.get('accuracy', 0):.4f} | {m.get('coverage', 1.0):.1%} |"
            )
        lines.append("")

    report = "\n".join(lines)
    output_path.write_text(report)
    return report
