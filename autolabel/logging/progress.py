"""Rich terminal display for autonomous loop progress."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table
from rich.panel import Panel


class ProgressDisplay:
    """Rich-based progress display for the autonomous loop."""

    def __init__(self) -> None:
        self.console = Console()

    def print_header(self, dataset: str, task: str, provider: str, max_iter: int) -> None:
        self.console.print(
            Panel(
                f"[bold]Dataset:[/] {dataset}\n"
                f"[bold]Task:[/] {task}\n"
                f"[bold]Provider:[/] {provider}\n"
                f"[bold]Max iterations:[/] {max_iter}",
                title="[bold cyan]AutoLabel[/]",
                border_style="cyan",
            )
        )

    def print_iteration_start(self, iteration: int, strategy: str, target_label: str) -> None:
        self.console.print(
            f"\n[bold yellow]--- Iteration {iteration} ---[/] "
            f"Strategy: [green]{strategy}[/] | Target: [blue]{target_label}[/]"
        )

    def print_iteration_result(
        self,
        iteration: int,
        f1: float,
        prev_f1: float,
        kept: bool,
        new_lfs: int,
        total_lfs: int,
        coverage: float,
    ) -> None:
        delta = f1 - prev_f1
        status = "[bold green]KEEP[/]" if kept else "[bold red]DISCARD[/]"
        arrow = "+" if delta >= 0 else ""
        self.console.print(
            f"  F1: {f1:.4f} ({arrow}{delta:.4f}) | "
            f"{status} | "
            f"New LFs: {new_lfs} | Active: {total_lfs} | "
            f"Coverage: {coverage:.1%}"
        )

    def print_lf_generated(self, name: str, valid: bool) -> None:
        icon = "[green]OK[/]" if valid else "[red]FAIL[/]"
        self.console.print(f"    LF {name}: {icon}")

    def print_final_summary(
        self,
        best_f1: float,
        total_iterations: int,
        active_lfs: int,
        total_generated: int,
        total_cost: float | None = None,
    ) -> None:
        summary = (
            f"[bold]Best F1:[/] {best_f1:.4f}\n"
            f"[bold]Iterations:[/] {total_iterations}\n"
            f"[bold]Active LFs:[/] {active_lfs}\n"
            f"[bold]Total generated:[/] {total_generated}"
        )
        if total_cost is not None:
            summary += f"\n[bold]Total cost:[/] ${total_cost:.4f}"
        self.console.print(
            Panel(summary, title="[bold green]Final Results[/]", border_style="green")
        )

    def print_error(self, msg: str) -> None:
        self.console.print(f"[bold red]ERROR:[/] {msg}")

    def print_info(self, msg: str) -> None:
        self.console.print(f"[dim]{msg}[/]")

    def print_benchmark_table(self, rows: list[dict]) -> None:
        table = Table(title="Benchmark Results")
        table.add_column("Method", style="cyan")
        table.add_column("F1", justify="right")
        table.add_column("Accuracy", justify="right")
        table.add_column("Coverage", justify="right")
        for row in rows:
            table.add_row(
                row["method"],
                f"{row['f1']:.4f}",
                f"{row.get('accuracy', 0):.4f}",
                f"{row.get('coverage', 1.0):.1%}",
            )
        self.console.print(table)
