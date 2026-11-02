"""Command-line interface for LeakAuditKit."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.table import Table

from .config import AuditConfig, DetectorConfig
from .detectors.ngram import NgramDetector
from .loaders import load
from .report import write_html, write_json, write_markdown

console = Console()


def _build_detector(cfg: DetectorConfig) -> Any:
    if cfg.name == "ngram":
        return NgramDetector(
            n=cfg.n, threshold=cfg.threshold, min_shared=cfg.min_shared, use_idf=cfg.use_idf
        )
    if cfg.name == "semantic":
        from .detectors.semantic import SemanticDetector

        return SemanticDetector(model_name=cfg.model_name, threshold=cfg.threshold,
                               batch_size=cfg.batch_size)
    if cfg.name == "all":
        return None  # run both
    raise click.ClickException(f"Unknown detector: {cfg.name}")


@click.group()
@click.version_option()
def main() -> None:
    """LeakAuditKit — audit ML/LLM evals for data leakage."""


@main.command()
@click.option("--train", "train_path", required=True, type=click.Path(exists=True),
              help="Path to train corpus (jsonl/csv/json).")
@click.option("--eval", "eval_path", required=True, type=click.Path(exists=True),
              help="Path to eval corpus (jsonl/csv/json).")
@click.option("--detector", type=click.Choice(["ngram", "semantic", "all"]), default="all")
@click.option("--threshold", type=float, default=None, help="Override detector threshold.")
@click.option("--n", type=int, default=5, help="N-gram size.")
@click.option("--min-shared", type=int, default=3, help="Minimum shared n-grams.")
@click.option("--use-idf", is_flag=True, help="Weight n-grams by inverse doc frequency.")
@click.option("--output", "output_dir", type=click.Path(), default="./audit_report")
@click.option("--format", "fmt", type=click.Choice(["json", "html", "markdown", "all"]),
              default="all")
def audit(train_path: str, eval_path: str, detector: str, threshold: float | None, n: int,
          min_shared: int, use_idf: bool, output_dir: str, fmt: str) -> None:
    """Run a leakage audit between train and eval corpora."""
    train = load(train_path)
    eval_ = load(eval_path)
    console.print(f"[bold]Loaded[/bold] train={len(train)} eval={len(eval_)}")

    t = threshold if threshold is not None else (0.3 if detector in ("ngram", "all") else 0.85)
    from typing import cast

    cfg = DetectorConfig(name=cast(Any, detector), threshold=t, n=n, min_shared=min_shared,
                        use_idf=use_idf)
    AuditConfig(train_path=Path(train_path), eval_path=Path(eval_path),
                detector=cfg, output_dir=Path(output_dir), format=cast(Any, fmt))

    detectors_to_run: list[Any] = []
    if detector in ("ngram", "all"):
        detectors_to_run.append(NgramDetector(n=n, threshold=t if detector == "ngram" else 0.3,
                                             min_shared=min_shared, use_idf=use_idf))
    if detector in ("semantic", "all"):
        try:
            from .detectors.semantic import SemanticDetector

            detectors_to_run.append(SemanticDetector(threshold=t if detector == "semantic" else 0.85))
        except Exception as exc:  # pragma: no cover
            console.print(f"[yellow]Semantic detector unavailable: {exc}[/yellow]")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for det in detectors_to_run:
        console.print(f"\n[bold cyan]Running {det.name} detector...[/bold cyan]")
        result = det.detect(train, eval_)
        d = result.to_dict()
        console.print(f"  matches={d['n_matches']}  contamination_rate={d['contamination_rate']:.2%}")

        table = Table(title=f"{det.name} top matches")
        table.add_column("Eval ID")
        table.add_column("Train ID")
        table.add_column("Score")
        for m in d["matches"][:10]:
            table.add_row(m["eval_id"], m["train_id"], str(m["score"]))
        if d["matches"]:
            console.print(table)

        stem = f"{det.name}_report"
        if fmt in ("json", "all"):
            write_json(result, out_dir / f"{stem}.json")
        if fmt in ("markdown", "all"):
            write_markdown(result, out_dir / f"{stem}.md")
        if fmt in ("html", "all"):
            write_html(result, out_dir / f"{stem}.html")

    console.print(f"\n[green]Reports written to {out_dir}[/green]")


if __name__ == "__main__":
    main()