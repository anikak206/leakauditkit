"""Command-line interface for LeakAuditKit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.table import Table

from .benchmark import make_scenario, plot_threshold_sweep, run_suite, save_results, threshold_sweep
from .config import AuditConfig, DetectorConfig
from .detectors.ngram import NgramDetector
from .graph import build_contamination_graph, export_graph_json, summarize_graph
from .loaders import load
from .report import write_html, write_json, write_markdown
from .shortcuts import audit_shortcuts

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
@click.option("--detector", type=click.Choice(["ngram", "tfidf", "semantic", "all"]), default="all")
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
    if detector in ("tfidf", "all"):
        from .detectors.tfidf import TfidfDetector

        detectors_to_run.append(TfidfDetector(threshold=t if detector == "tfidf" else 0.5))
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


@main.command()
@click.option("--output", "output_dir", type=click.Path(), default="./benchmark_report")
@click.option("--n-train", type=int, default=60)
@click.option("--n-eval", type=int, default=60)
@click.option("--seed", type=int, default=42)
def benchmark(output_dir: str, n_train: int, n_eval: int, seed: int) -> None:
    """Run the standard contamination benchmark suite and write a report."""
    from .detectors.ngram import NgramDetector
    from .detectors.semantic import SemanticDetector

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = [
        make_scenario("verbatim_25", n_train=n_train, n_eval=n_eval, contamination_rate=0.25,
                      mode="verbatim", seed=seed),
        make_scenario("partial_25", n_train=n_train, n_eval=n_eval, contamination_rate=0.25,
                      mode="partial", seed=seed + 1),
        make_scenario("mixed_25", n_train=n_train, n_eval=n_eval, contamination_rate=0.25,
                      mode="mixed", seed=seed + 2),
        make_scenario("clean_0", n_train=n_train, n_eval=n_eval, contamination_rate=0.0,
                      mode="clean", seed=seed + 3),
    ]

    # n-gram suite
    def ng_factory(threshold: float = 0.3) -> Any:
        return NgramDetector(n=5, threshold=threshold, min_shared=3)

    ng_results = run_suite(lambda: NgramDetector(n=5, threshold=0.3, min_shared=3), scenarios)
    save_results(ng_results, out_dir / "ngram_results.json")

    # threshold sweep on mixed scenario
    mixed = scenarios[2]
    sweep = threshold_sweep(lambda threshold=0.3: NgramDetector(n=5, threshold=threshold, min_shared=3),
                           mixed, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    (out_dir / "ngram_sweep.json").write_text(json.dumps(sweep, indent=2), encoding="utf-8")
    plot_threshold_sweep(sweep, out_dir / "ngram_sweep.png", title="N-gram threshold sweep (mixed)")

    console.print("[bold]N-gram benchmark[/bold]")
    for r in ng_results:
        console.print(f"  {r.scenario}: P={r.precision:.3f} R={r.recall:.3f} F1={r.f1:.3f} (TP={r.tp} FP={r.fp} FN={r.fn})")

    # try semantic if available
    try:
        sem_results = run_suite(lambda: SemanticDetector(threshold=0.85), scenarios)
        save_results(sem_results, out_dir / "semantic_results.json")
        console.print("[bold]Semantic benchmark[/bold]")
        for r in sem_results:
            console.print(f"  {r.scenario}: P={r.precision:.3f} R={r.recall:.3f} F1={r.f1:.3f}")
    except Exception as exc:
        console.print(f"[yellow]Semantic detector skipped: {exc}[/yellow]")

    console.print(f"\n[green]Benchmark report written to {out_dir}[/green]")


@main.command()
@click.option("--input", "input_path", required=True, type=click.Path(exists=True),
              help="JSON/JSONL of records with label, prediction, and features.")
@click.option("--label-key", default="label")
@click.option("--pred-key", default="prediction")
@click.option("--text-key", default=None)
@click.option("--feature-keys", default=None, help="Comma-separated feature columns.")
@click.option("--output", "output_path", type=click.Path(), default=None)
def shortcuts(input_path: str, label_key: str, pred_key: str, text_key: str | None,
              feature_keys: str | None, output_path: str | None) -> None:
    """Audit a prediction file for shortcut learning."""
    import json as _json

    from .loaders import load

    records = load(input_path)
    fk = [x.strip() for x in feature_keys.split(",")] if feature_keys else None
    rep = audit_shortcuts(records, label_key=label_key, pred_key=pred_key,
                          feature_keys=fk, text_key=text_key)
    console.print_json(data=rep.to_dict())
    if output_path:
        Path(output_path).write_text(_json.dumps(rep.to_dict(), indent=2), encoding="utf-8")
        console.print(f"[green]Wrote {output_path}[/green]")


@main.command()
@click.option("--input", "input_path", required=True, type=click.Path(exists=True),
              help="JSON report from a detector run (must contain 'matches').")
@click.option("--output", "output_path", type=click.Path(), default=None)
def graph(input_path: str, output_path: str | None) -> None:
    """Build a contamination graph from a detector report and summarize it."""
    import json as _json

    data = _json.loads(Path(input_path).read_text(encoding="utf-8"))
    matches = data.get("matches", [])
    G = build_contamination_graph(matches)
    summary = summarize_graph(G)
    payload = {"summary": summary.to_dict(), "graph": export_graph_json(G)}
    console.print_json(data=summary.to_dict())
    if output_path:
        Path(output_path).write_text(_json.dumps(payload, indent=2), encoding="utf-8")
        console.print(f"[green]Wrote {output_path}[/green]")


@main.command("audit-hf")
@click.option("--train-dataset", required=True, help="HuggingFace train dataset name.")
@click.option("--eval-dataset", required=True, help="HuggingFace eval/benchmark dataset name.")
@click.option("--train-split", default="train")
@click.option("--eval-split", default="test")
@click.option("--text-col", default="text")
@click.option("--limit", type=int, default=200)
@click.option("--output", "output_path", type=click.Path(), default=None)
def audit_hf(train_dataset: str, eval_dataset: str, train_split: str, eval_split: str,
             text_col: str, limit: int, output_path: str | None) -> None:
    """Audit a HuggingFace dataset pair for benchmark contamination."""
    try:
        from .hf_audit import audit_hf_dataset
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc
    try:
        payload = audit_hf_dataset(train_dataset, eval_dataset, train_split=train_split,
                                  eval_split=eval_split, text_col=text_col, limit=limit)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc
    console.print_json(data=payload["result"])
    if output_path:
        Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        console.print(f"[green]Wrote {output_path}[/green]")


if __name__ == "__main__":
    main()