"""
Feature Engineering System — CLI Entry Point
──────────────────────────────────────────────
Commands:
    run     Execute the full agent pipeline on a CSV/Parquet file.
    ui      Launch the Streamlit web UI.
    demo    Run the pipeline on built-in synthetic data and print a summary.

Examples
--------
    python main.py demo
    python main.py run --data transactions.csv --target is_fraud
    python main.py ui
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# ── Make project root importable ─────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="fe-system",
    help="🧠 Multi-Agent Feature Engineering System",
    add_completion=False,
)
console = Console()


def _load_dataframe(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _selected_definitions(state: object) -> list:
    if hasattr(state, "get_selected_definitions"):
        selected = state.get_selected_definitions()
        if selected:
            return selected
    return getattr(state, "constructed_features", []) or []


def _stage_label(stage_name: str, status: str) -> str:
    if status == "done":
        return f"[green]OK[/green] {stage_name}"
    if status == "error":
        return f"[red]FAIL[/red] {stage_name}"
    return f"[yellow]...[/yellow] {stage_name}"


# ── Commands ──────────────────────────────────────────────────────────────────

@app.command()
def run(
    data: Path = typer.Option(..., "--data", "-d", help="Path to CSV or Parquet dataset."),
    target: str = typer.Option(..., "--target", "-t", help="Name of the target column."),
    task_type: str = typer.Option("classification", "--task", help="'classification' or 'regression'."),
    output_dir: Path = typer.Option(Path("outputs"), "--output", "-o", help="Directory for all outputs."),
    max_features: int = typer.Option(50, "--max-features", help="Max features to select."),
    max_iter: int = typer.Option(3, "--max-iter", help="Max feedback iterations."),
    selection_method: str = typer.Option("pareto", "--selection", help="pareto | greedy | lasso"),
    export_sql: bool = typer.Option(True, "--sql/--no-sql", help="Export SQL script."),
    export_py: bool = typer.Option(True, "--python/--no-python", help="Export Python module."),
) -> None:
    """Run the full feature engineering pipeline on a dataset."""
    from core.config import Config
    from core.orchestrator import FeatureEngineeringOrchestrator
    from pipelines.python_pipeline import PythonPipeline
    from pipelines.sql_pipeline import SQLPipeline
    from utils.feature_store import FeatureStore

    # ── Load data ─────────────────────────────────────────────────────────────
    console.rule("[bold blue]🧠 Feature Engineering System")
    console.print(f"[green]Loading data:[/green] {data}")

    try:
        df = _load_dataframe(data)
    except Exception as e:
        console.print(f"[red]Failed to load data: {e}[/red]")
        raise typer.Exit(1)

    console.print(f"[cyan]Dataset:[/cyan] {df.shape[0]:,} rows × {df.shape[1]} columns")
    console.print(f"[cyan]Target:[/cyan]  {target}  ({task_type})")

    # ── Config ────────────────────────────────────────────────────────────────
    cfg = Config()
    cfg.max_selected_features = max_features
    cfg.max_feedback_iterations = max_iter
    cfg.selection_method = selection_method
    cfg.output_python = export_py
    cfg.output_sql = export_sql
    cfg.set_output_dir(output_dir)
    cfg.ensure_dirs()

    # ── Run ───────────────────────────────────────────────────────────────────
    def progress_cb(stage: str, status: str, metrics: dict) -> None:
        label = _stage_label(stage, status)
        if status in {"done", "error"}:
            elapsed = metrics.get("elapsed_s", "?")
            console.print(f"  {label:<40} {elapsed}s")
            return
            console.print(f"  [green]✓[/green] {stage:<20} {elapsed}s")
        else:
            console.print(f"  {label}")
            return
            console.print(f"  [yellow]⚙[/yellow] {stage:<20}", end="\r")

    console.print("\n[bold]Pipeline stages:[/bold]")
    t_start = time.perf_counter()

    try:
        orch = FeatureEngineeringOrchestrator(config=cfg, progress_callback=progress_cb)
        state = orch.run(
            df=df,
            target_column=target,
            task_type=task_type,
        )
    except Exception as e:
        console.print(f"[red]Pipeline failed: {e}[/red]")
        import traceback; traceback.print_exc()
        raise typer.Exit(1)

    elapsed = time.perf_counter() - t_start
    console.print(f"\n[bold green]Pipeline complete in {elapsed:.1f}s[/bold green]")

    # ── Summary table ─────────────────────────────────────────────────────────
    _print_summary(state)

    # ── Exports ──────────────────────────────────────────────────────────────
    selected_defs = _selected_definitions(state)

    if export_py and selected_defs:
        py_path = output_dir / "features_transform.py"
        pipe = PythonPipeline(selected_defs)
        py_src = pipe.generate_module_code()
        py_path.write_text(py_src, encoding="utf-8")
        console.print(f"[green]Python module →[/green] {py_path}")

    if export_sql and selected_defs:
        sql_path = output_dir / "features.sql"
        sql_pipe = SQLPipeline(
            selected_defs,
            source_table="raw_events",
            dialect=cfg.sql_dialect,
        )
        sql_pipe.save_sql(str(sql_path))
        console.print(f"[green]SQL script   →[/green] {sql_path}")

    try:
        store = FeatureStore(store_root=str(output_dir / "feature_store"))
        store.register_many(getattr(state, "constructed_features", []) or [])
        run_id = getattr(state, "run_id", f"run_{int(time.time())}")
        constructed_df = state.__dict__.get("constructed_df")
        if constructed_df is not None:
            store.save_dataframe(constructed_df, run_id=run_id)
        console.print(f"[green]Feature store →[/green] {output_dir / 'feature_store'}")
    except Exception as e:
        console.print(f"[yellow]Feature store skipped: {e}[/yellow]")


@app.command()
def demo() -> None:
    """Run the pipeline on synthetic e-commerce demo data."""
    console.rule("[bold blue]🎲 Demo Mode — Synthetic Transactions")
    from data.sample_data_generator import generate_transactions

    console.print("[cyan]Generating synthetic dataset…[/cyan]")
    df = generate_transactions(n_users=400, n_rows=8_000, seed=42)

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        tmp_path = tmp.name
    df.to_parquet(tmp_path, index=False)

    try:
        run(
            data=Path(tmp_path),
            target="is_fraud",
            task_type="classification",
            output_dir=Path("demo_output"),
            max_features=30,
            max_iter=2,
            selection_method="pareto",
            export_sql=True,
            export_py=True,
        )
    finally:
        os.unlink(tmp_path)


@app.command()
def ui() -> None:
    """Launch the Streamlit web UI."""
    console.print("[bold green]Launching Streamlit UI…[/bold green]")
    app_path = Path(__file__).parent / "ui" / "app.py"
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path),
         "--server.port", "8501", "--browser.gatherUsageStats", "false"],
        check=False,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _print_summary(state: object) -> None:
    candidates = getattr(state, "feature_candidates", []) or []
    evaluations = getattr(state, "feature_evaluations", []) or []
    sel = getattr(state, "selection_result", None)
    selected_ids = (sel.selected_feature_ids if sel else []) or []

    table = Table(title="Pipeline Summary", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")

    table.add_row("Candidates generated", str(len(candidates)))
    table.add_row("Features selected", str(len(selected_ids)))
    if evaluations:
        top_iv = max((getattr(e, "iv_score", 0) or 0 for e in evaluations), default=0)
        table.add_row("Best IV score", f"{top_iv:.4f}")
    feedback_reports = getattr(state, "feedback_reports", []) or []
    if feedback_reports:
        table.add_row("Feedback iterations", str(len(feedback_reports)))

    console.print(table)

    # Top 10 selected features
    if evaluations and selected_ids:
        sel_evals = sorted(
            [e for e in evaluations if getattr(e, "feature_id", None) in selected_ids],
            key=lambda e: getattr(e, "composite_score", 0) or 0,
            reverse=True,
        )[:10]

        feat_table = Table(title="Top 10 Selected Features", show_header=True, header_style="bold magenta")
        feat_table.add_column("Feature", style="white")
        feat_table.add_column("IV", justify="right")
        feat_table.add_column("SHAP", justify="right")
        feat_table.add_column("Composite", justify="right")

        for ev in sel_evals:
            feat_table.add_row(
                getattr(ev, "feature_name", "?"),
                f"{getattr(ev, 'iv_score', 0) or 0:.4f}",
                f"{getattr(ev, 'shap_mean_abs', 0) or 0:.4f}",
                f"{getattr(ev, 'composite_score', 0) or 0:.4f}",
            )
        console.print(feat_table)


# ── Entry ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app()
