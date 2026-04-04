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

import sys
import time
from pathlib import Path
from typing import Optional

# ── Make project root importable ─────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table

app = typer.Typer(
    name="fe-system",
    help="🧠 Multi-Agent Feature Engineering System",
    add_completion=False,
)
console = Console()


# ── Commands ──────────────────────────────────────────────────────────────────

@app.command()
def run(
    data: Path = typer.Option(..., "--data", "-d", help="Path to CSV or Parquet dataset."),
    target: str = typer.Option(..., "--target", "-t", help="Name of the target column."),
    task_type: str = typer.Option("classification", "--task", help="'classification' or 'regression'."),
    output_dir: Path = typer.Option(Path("output"), "--output", "-o", help="Directory for all outputs."),
    max_features: int = typer.Option(50, "--max-features", help="Max features to select."),
    max_iter: int = typer.Option(3, "--max-iter", help="Max feedback iterations."),
    selection_method: str = typer.Option("pareto", "--selection", help="pareto | greedy | lasso"),
    export_sql: bool = typer.Option(True, "--sql/--no-sql", help="Export SQL script."),
    export_py: bool = typer.Option(True, "--python/--no-python", help="Export Python module."),
) -> None:
    """Run the full feature engineering pipeline on a dataset."""
    import pandas as pd
    from core.config import Config
    from core.orchestrator import FeatureEngineeringOrchestrator
    from pipelines.python_pipeline import PythonPipeline
    from pipelines.sql_pipeline import SQLPipeline

    # ── Load data ─────────────────────────────────────────────────────────────
    console.rule("[bold blue]🧠 Feature Engineering System")
    console.print(f"[green]Loading data:[/green] {data}")

    try:
        df = pd.read_parquet(data) if str(data).endswith(".parquet") else pd.read_csv(data)
    except Exception as e:
        console.print(f"[red]Failed to load data: {e}[/red]")
        raise typer.Exit(1)

    console.print(f"[cyan]Dataset:[/cyan] {df.shape[0]:,} rows × {df.shape[1]} columns")
    console.print(f"[cyan]Target:[/cyan]  {target}  ({task_type})")

    # ── Config ────────────────────────────────────────────────────────────────
    cfg = Config()
    object.__setattr__(cfg, "max_selected_features", max_features)
    object.__setattr__(cfg, "max_feedback_iterations", max_iter)
    object.__setattr__(cfg, "selection_method", selection_method)
    output_dir.mkdir(parents=True, exist_ok=True)
    object.__setattr__(cfg, "output_dir", str(output_dir))

    # ── Run ───────────────────────────────────────────────────────────────────
    stage_times: dict = {}

    def progress_cb(stage: str, status: str, metrics: dict) -> None:
        if status == "done":
            elapsed = metrics.get("elapsed_s", "?")
            console.print(f"  [green]✓[/green] {stage:<20} {elapsed}s")
        elif status == "running":
            console.print(f"  [yellow]⚙[/yellow] {stage:<20}", end="\r")

    console.print("\n[bold]Pipeline stages:[/bold]")
    t_start = time.perf_counter()

    try:
        orch = FeatureEngineeringOrchestrator(config=cfg)
        state = orch.run(
            df=df,
            target_column=target,
            task_type=task_type,
            progress_callback=progress_cb,
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
    candidates = getattr(state, "feature_candidates", []) or []
    sel = getattr(state, "selected_features", None)
    selected_ids = (sel.selected_feature_ids if sel else []) or []
    selected_defs = [c for c in candidates
                     if getattr(c, "feature_id", None) in selected_ids]

    if export_py and selected_defs:
        py_path = output_dir / "features_transform.py"
        pipe = PythonPipeline(selected_defs)
        py_src = pipe.generate_module_code()
        py_path.write_text(py_src, encoding="utf-8")
        console.print(f"[green]Python module →[/green] {py_path}")

    if export_sql and selected_defs:
        sql_path = output_dir / "features.sql"
        sql_pipe = SQLPipeline(selected_defs, source_table="raw_events")
        sql_pipe.save_sql(str(sql_path))
        console.print(f"[green]SQL script   →[/green] {sql_path}")

    # Feature store persist
    try:
        from utils.feature_store import FeatureStore
        store = FeatureStore(store_root=str(output_dir / "feature_store"))
        store.register_many(candidates)
        run_id = f"run_{int(time.time())}"
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

    # Invoke the run logic by setting up a temporary file
    import tempfile, os
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
    import subprocess
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
    evaluations = getattr(state, "evaluations", []) or []
    sel = getattr(state, "selected_features", None)
    selected_ids = (sel.selected_feature_ids if sel else []) or []

    table = Table(title="Pipeline Summary", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")

    table.add_row("Candidates generated", str(len(candidates)))
    table.add_row("Features selected", str(len(selected_ids)))
    if evaluations:
        top_iv = max((getattr(e, "iv_score", 0) or 0 for e in evaluations), default=0)
        table.add_row("Best IV score", f"{top_iv:.4f}")
    fb = getattr(state, "feedback_report", None)
    if fb:
        table.add_row("Feedback iterations", str(getattr(fb, "iteration", 0)))

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
                getattr(ev, "feature_id", "?"),
                f"{getattr(ev, 'iv_score', 0) or 0:.4f}",
                f"{getattr(ev, 'shap_importance', 0) or 0:.4f}",
                f"{getattr(ev, 'composite_score', 0) or 0:.4f}",
            )
        console.print(feat_table)


# ── Entry ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app()
