#!/usr/bin/env python3
"""kagglemate — LangGraph-based Kaggle competition assistant.

Usage:
    python main.py --help
    python main.py check                        # Verify setup
    python main.py list [--search <term>]       # List competitions
    python main.py inspect <competition-slug>   # Show competition info
    python main.py research <competition-slug>  # Full research pipeline
    python main.py profile <competition-slug>   # Data profile only
    python main.py spec <competition-slug>      # Generate SPEC.md only
    python main.py baseline <competition-slug>  # Generate baseline script
    python main.py run <competition-slug>       # Run training script
    python main.py suggest <competition-slug>   # Get next-step recommendations
    python main.py experiments <slug> --action list  # View experiment history
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure the project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import typer
from rich.console import Console
from rich.table import Table

from kagglemate.config import config
from kagglemate.tools.kaggle_cli import KaggleCLI

app = typer.Typer(
    name="kagglemate",
    help="Kaggle competition assistant powered by DeepSeek V4 + LangGraph",
    add_completion=False,
)
console = Console()


# ── check ──


@app.command()
def check():
    """Verify that all dependencies are configured correctly."""
    console.print("[bold]KaggleMate Health Check[/]\n")

    # 1. Python version
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info >= (3, 10):
        console.print(f"  Python {py_ver}  [green]✓[/]")
    else:
        console.print(f"  Python {py_ver}  [red]✗ (need 3.10+)[/]")

    # 2. DeepSeek key
    if config.DEEPSEEK_API_KEY:
        masked = config.DEEPSEEK_API_KEY[:8] + "..." + config.DEEPSEEK_API_KEY[-4:]
        console.print(f"  DeepSeek API key: {masked}  [green]✓[/]")
    else:
        console.print("  DeepSeek API key  [red]✗ (set DEEPSEEK_API_KEY in .env)[/]")

    # 3. Kaggle CLI
    import subprocess
    result = subprocess.run(["kaggle", "--version"], capture_output=True, text=True)
    if result.returncode == 0:
        console.print(f"  Kaggle CLI: {result.stdout.strip()}  [green]✓[/]")
    else:
        console.print("  Kaggle CLI  [red]✗ (install: pip install kaggle)[/]")

    # 4. Kaggle credentials
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        console.print(f"  Kaggle credentials: {kaggle_json}  [green]✓[/]")
    else:
        console.print("  Kaggle credentials  [red]✗ (place kaggle.json in ~/.kaggle/)[/]")

    # 5. Competitions
    try:
        comps = KaggleCLI.list_competitions()
        console.print(f"  Active competitions: {len(comps)}  [green]✓[/]")
    except Exception as e:
        console.print(f"  Competition listing  [red]✗ ({e})[/]")

    console.print("\n" + config.summary())


# ── list ──


@app.command()
def list_competitions(search: str = typer.Option("", "--search", "-s", help="Filter by search term")):
    """List available Kaggle competitions."""
    try:
        comps = KaggleCLI.list_competitions(search=search)
    except Exception as e:
        console.print(f"[red]Error: {e}[/]")
        raise typer.Exit(code=1)

    if not comps:
        console.print("[yellow]No competitions found.[/]")
        return

    table = Table(title=f"Kaggle Competitions ({len(comps)})")
    table.add_column("Ref", style="cyan")
    table.add_column("Deadline", style="yellow")
    table.add_column("Category")
    table.add_column("Teams")

    for c in comps[:30]:
        table.add_row(
            c.get("ref", "?"),
            c.get("deadline", "?")[:10],
            c.get("category", "?"),
            c.get("teamCount", "?"),
        )

    console.print(table)


# ── inspect ──


@app.command()
def inspect(competition: str = typer.Argument(..., help="Competition slug, e.g. 'titanic'")):
    """Show detailed information about a Kaggle competition."""
    # Files
    console.print(f"[bold]Files for: {competition}[/]\n")
    try:
        files = KaggleCLI.list_files(competition)
    except Exception as e:
        console.print(f"[red]Error listing files: {e}[/]")
        raise typer.Exit(code=1)

    if not files:
        console.print("[yellow]No files found (competition may not be active).[/]")
    else:
        table = Table(title="Data Files")
        table.add_column("Name", style="cyan")
        table.add_column("Size", justify="right")
        for f in files:
            table.add_row(f.get("name", "?"), f.get("size", "?"))
        console.print(table)

    # Kernels
    console.print("\n[bold]Top Notebooks (by votes)[/]\n")
    try:
        kernels = KaggleCLI.list_kernels(competition, sort_by="votes", limit=10)
    except Exception:
        kernels = []

    if not kernels:
        console.print("[dim]No public kernels found.[/]")
    else:
        k_table = Table(title="Top Kernels")
        k_table.add_column("Title", style="cyan")
        k_table.add_column("Author")
        k_table.add_column("Votes", justify="right")
        for k in kernels:
            k_table.add_row(
                k.get("title", "?")[:60],
                k.get("author", "?"),
                k.get("totalVotes", "0"),
            )
        console.print(k_table)


# ── research: full pipeline ──


@app.command()
def research(competition: str = typer.Argument(..., help="Competition slug, e.g. 'titanic'"),
             no_download: bool = typer.Option(False, "--no-download", help="Skip data download (data already on disk)")):
    """Run the full Phase 1 research pipeline.

    Downloads data, profiles it, researches public notebooks,
    and generates SPEC.md + research_summary.md + rules_checklist.md.
    """
    from kagglemate.graph.builder import get_research_graph
    from kagglemate.graph.state import KaggleAgentState
    from kagglemate.tools.kaggle_cli import KaggleCLI
    from rich.panel import Panel
    from rich.markdown import Markdown
    import json

    console.print(f"\n[bold cyan]🔬 KaggleMate Research Pipeline[/]\n")
    console.print(f"  Competition: [yellow]{competition}[/]")
    console.print(f"  Model: [dim]{config.DEEPSEEK_MODEL}[/]")

    # Pre-check: does the competition exist?
    try:
        files = KaggleCLI.list_files(competition)
        if files:
            console.print(f"  Files available: [green]{len(files)}[/]")
        else:
            console.print(f"  Files: [yellow]none found (may need to accept rules first)[/]")
    except Exception as e:
        console.print(f"  [red]Error checking competition: {e}[/]")

    # Build initial state
    initial_state: KaggleAgentState = {
        "competition_slug": competition,
        "competition_name": competition,
        "messages": [],
        "competition_type": "other",
        "evaluation_metric": "unknown",
        "files": [],
        "data_dir": "",
        "report_dir": "",
        "submission_dir": "",
        "script_dir": "",
        "notebook_summaries": [],
        "research_complete": False,
        "current_phase": "init",
        "errors": [],
        "human_approval_required": False,
        "human_approved": False,
        "best_cv_score": 0.0,
        "best_lb_score": 0.0,
    }

    # Run graph
    config_dict = {"configurable": {"thread_id": f"research-{competition}"}}

    console.print(f"\n[bold]Running pipeline...[/]\n")

    final_state = None
    with get_research_graph() as graph:
        for event in graph.stream(initial_state, config_dict):
            for node_name, node_output in event.items():
                phase = node_output.get("current_phase", "?")
                errors = node_output.get("errors", [])

                if errors:
                    for err in errors:
                        console.print(f"  [{node_name}] [red]✗ {err[:200]}[/]")
                else:
                    console.print(f"  [{node_name}] → [green]{phase}[/]")

        # Get the full accumulated state (not just last node output)
        full_state = graph.get_state(config_dict)
        if full_state and full_state.values:
            final_state = full_state.values

    console.print("\n[bold green]✅ Research complete![/]\n")

    # Show output files
    from pathlib import Path
    report_dir = None
    if final_state:
        report_dir = final_state.get("report_dir", "")

    if report_dir:
        report_path = Path(report_dir)
        console.print("[bold]Generated files:[/]")
        for f in sorted(report_path.glob("*.md")):
            size_kb = f.stat().st_size / 1024
            console.print(f"  [cyan]📄 {f.name}[/] ({size_kb:.1f} KB)")

    # Quick summary of findings
    if final_state:
        ctype = final_state.get("competition_type", "?")
        metric = final_state.get("evaluation_metric", "?")
        n_nb = len(final_state.get("notebook_summaries", []))
        console.print(f"\n[bold]Quick Summary:[/]")
        console.print(f"  Task type: [yellow]{ctype}[/]")
        console.print(f"  Metric: [yellow]{metric}[/]")
        console.print(f"  Notebooks analyzed: [yellow]{n_nb}[/]")


# ── profile: data profiling only ──


@app.command()
def profile(competition: str = typer.Argument(..., help="Competition slug, e.g. 'titanic'")):
    """Generate data_profile.md without full research."""
    from kagglemate.graph.nodes.init_node import run as init_run
    from kagglemate.graph.nodes.analyze_node import run as analyze_run
    from kagglemate.graph.state import KaggleAgentState

    console.print(f"\n[bold cyan]📊 Data Profiler[/]\n")

    state: KaggleAgentState = {
        "competition_slug": competition,
        "messages": [],
        "current_phase": "init",
        "errors": [],
        "best_cv_score": 0.0,
        "best_lb_score": 0.0,
        "human_approval_required": False,
        "human_approved": False,
    }

    # Run init + analyze
    for node_fn, name in [(init_run, "init"), (analyze_run, "analyze")]:
        updates = node_fn(state)
        state.update(updates)
        errors = updates.get("errors", [])
        if errors:
            for e in errors:
                console.print(f"  [{name}] [red]{e[:300]}[/]")
        else:
            console.print(f"  [{name}] → [green]done[/]")

    console.print(f"\n[green]✅ data_profile.md generated.[/]")


# ── spec: generate SPEC.md from existing research ──


@app.command()
def spec(competition: str = typer.Argument(..., help="Competition slug")):
    """Generate SPEC.md from existing data and research (skip downloads)."""
    from kagglemate.graph.nodes.plan_node import run as plan_run
    from kagglemate.graph.state import KaggleAgentState
    from kagglemate.config import config as cfg
    import json

    comp_dir = cfg.COMPETITIONS_DIR / competition

    console.print(f"\n[bold cyan]📝 SPEC Generator[/]\n")

    if not comp_dir.exists():
        console.print(f"[red]No data found for '{competition}'. Run 'research' first.[/]")
        raise typer.Exit(code=1)

    # Try to load existing data profile
    data_dir = str(comp_dir / "data" / "raw")
    report_dir = str(comp_dir / "reports")

    state: KaggleAgentState = {
        "competition_slug": competition,
        "competition_name": competition,
        "data_dir": data_dir,
        "report_dir": report_dir,
        "messages": [],
        "current_phase": "plan",
        "errors": [],
        "best_cv_score": 0.0,
        "best_lb_score": 0.0,
        "human_approval_required": False,
        "human_approved": False,
    }

    updates = plan_run(state)
    state.update(updates)

    if state.get("spec_path"):
        console.print(f"[green]✅ SPEC.md → {state['spec_path']}[/]")
    if state.get("rules_checklist_path"):
        console.print(f"[green]✅ rules_checklist.md → {state['rules_checklist_path']}[/]")


# ── baseline: generate training script ──


@app.command()
def baseline(competition: str = typer.Argument(..., help="Competition slug, e.g. 'titanic'")):
    """Generate a baseline training script from the research data."""
    from kagglemate.graph.nodes.baseline_node import run as baseline_run
    from kagglemate.graph.nodes.init_node import run as init_run
    from kagglemate.graph.nodes.analyze_node import run as analyze_run
    from kagglemate.graph.state import KaggleAgentState
    from kagglemate.config import config as cfg

    console.print(f"\n[bold cyan]🧪 Baseline Generator[/]\n")

    comp_dir = cfg.COMPETITIONS_DIR / competition
    if not comp_dir.exists():
        console.print(f"[red]No data for '{competition}'. Run 'research' first.[/]")
        raise typer.Exit(code=1)

    state: KaggleAgentState = {
        "competition_slug": competition,
        "competition_name": competition,
        "data_dir": str(comp_dir / "data" / "raw"),
        "report_dir": str(comp_dir / "reports"),
        "script_dir": str(comp_dir / "scripts"),
        "submission_dir": str(comp_dir / "submissions"),
        "messages": [],
        "current_phase": "build",
        "errors": [],
        "best_cv_score": 0.0,
        "best_lb_score": 0.0,
        "human_approval_required": False,
        "human_approved": False,
    }

    # Ensure data profile exists (run analyze if needed)
    data_dir = state["data_dir"]
    if not Path(data_dir).exists() or not list(Path(data_dir).glob("*.csv")):
        console.print("[yellow]Data not found, downloading...[/]")
        init_state = init_run(state)
        state.update(init_state)

    # Run analyze to get data profile
    analyze_state = analyze_run(state)
    state.update(analyze_state)

    console.print(f"  Task type: [yellow]{state.get('competition_type')}[/]")
    console.print(f"  Target: [yellow]{state.get('data_profile', {}).get('target_col', '?')}[/]")

    # Run baseline generation
    updates = baseline_run(state)
    state.update(updates)

    exp = state.get("current_experiment", {})
    if exp.get("script_path"):
        console.print(f"\n[green]✅ Baseline script → {exp['script_path']}[/]")
        console.print(f"  Model: [yellow]{exp.get('model')}[/]")
        console.print(f"  Features: [dim]{len(exp.get('features', []))} selected[/]")
        console.print(f"\n[bold]Next:[/] python main.py run --competition {competition}")
    else:
        console.print("[red]Failed to generate baseline.[/]")


# ── run: execute training script ──


@app.command(name="run")
def run_script(competition: str = typer.Argument(..., help="Competition slug, e.g. 'titanic'"),
               script: str = typer.Option("", "--script", "-s", help="Path to training script (auto-detected if omitted)")):
    """Execute a baseline training script and record results."""
    from kagglemate.graph.nodes.run_node import run as run_node_fn
    from kagglemate.graph.state import KaggleAgentState
    from kagglemate.config import config as cfg
    from pathlib import Path

    console.print(f"\n[bold cyan]🚀 Experiment Runner[/]\n")

    comp_dir = cfg.COMPETITIONS_DIR / competition
    scripts_dir = comp_dir / "scripts"

    # Find the script
    script_path = None
    if script:
        script_path = script
    elif scripts_dir.exists():
        candidates = sorted(scripts_dir.glob("train_baseline_*.py"), reverse=True)
        if candidates:
            script_path = str(candidates[0])
            console.print(f"  Using: [dim]{candidates[0].name}[/]")

    if not script_path or not Path(script_path).exists():
        console.print(f"[red]No training script found. Run 'baseline' first.[/]")
        raise typer.Exit(code=1)

    state: KaggleAgentState = {
        "competition_slug": competition,
        "data_dir": str(comp_dir / "data" / "raw"),
        "report_dir": str(comp_dir / "reports"),
        "submission_dir": str(comp_dir / "submissions"),
        "script_dir": str(scripts_dir),
        "messages": [],
        "current_phase": "run",
        "errors": [],
        "best_cv_score": 0.0,
        "best_lb_score": 0.0,
        "human_approval_required": False,
        "human_approved": False,
        "current_experiment": {
            "name": Path(script_path).stem.replace("train_baseline_", ""),
            "model": "Unknown",
            "script_path": script_path,
            "status": "running",
        },
    }

    console.print(f"  Competition: [yellow]{competition}[/]")
    console.print(f"  Script: [dim]{script_path}[/]\n")

    updates = run_node_fn(state)
    state.update(updates)

    exp = state.get("current_experiment", {})
    errors = state.get("errors", [])

    if errors and not any("Validation warning" in e for e in errors):
        console.print(f"\n[red]❌ Run failed:[/]")
        for e in errors:
            console.print(f"  {e[:300]}")
    else:
        cv = exp.get("cv_score", 0.0)
        cv_std = exp.get("cv_std", 0.0)
        sub = exp.get("submission_path", "")
        exp_id = exp.get("id", "?")

        console.print(f"\n[bold green]✅ Experiment #{exp_id} complete![/]")
        console.print(f"  CV Score: [bold yellow]{cv:.5f}[/] ± {cv_std:.5f}")
        console.print(f"  Submission: [cyan]{sub}[/]")

        if errors:
            console.print(f"\n[yellow]⚠ Warnings:[/]")
            for e in errors:
                console.print(f"  {e[:200]}")


# ── experiments: list/show/compare/log-lb ──


@app.command()
def experiments(
    competition: str = typer.Argument(..., help="Competition slug"),
    action: str = typer.Option("list", "--action", "-a", help="list | show | compare | log-lb"),
    exp_id: int = typer.Option(None, "--id", help="Experiment ID (for show/log-lb)"),
    lb_score: float = typer.Option(None, "--lb", help="Leaderboard score to record"),
    ids: str = typer.Option("", "--ids", help="Comma-separated experiment IDs (for compare)"),
):
    """Manage experiment records."""
    from kagglemate.memory.experiment_store import ExperimentStore
    from kagglemate.config import config as cfg
    from rich.table import Table

    store = ExperimentStore(competition)

    if action == "list":
        exps = store.list_all()
        if not exps:
            console.print(f"[yellow]No experiments for '{competition}'.[/]")
            return

        table = Table(title=f"Experiments — {competition} ({len(exps)} total)")
        table.add_column("ID", style="dim")
        table.add_column("Name", style="cyan")
        table.add_column("Model")
        table.add_column("CV Score", justify="right", style="yellow")
        table.add_column("LB Score", justify="right", style="green")
        table.add_column("Status")
        table.add_column("Date")

        for e in exps[:30]:
            cv = f"{e['cv_score']:.5f}" if e.get("cv_score") else "—"
            lb = f"{e['lb_score']:.5f}" if e.get("lb_score") else "—"
            status = e.get("status", "?")
            status_icon = "✅" if status == "completed" else "❌" if status == "failed" else "⏳"
            date = (e.get("created_at") or "")[:10]
            table.add_row(
                str(e["id"]),
                e.get("experiment_name", "?")[:30],
                e.get("model_name", "?")[:15],
                cv,
                lb,
                f"{status_icon} {status}",
                date,
            )

        console.print(table)
        console.print(f"\n[dim]To record LB: python main.py experiments --action log-lb --id <id> --lb <score>[/]")

    elif action == "show" and exp_id:
        exp = store.get(exp_id)
        if not exp:
            console.print(f"[red]Experiment #{exp_id} not found.[/]")
            return

        console.print(f"\n[bold]Experiment #{exp['id']}: {exp.get('experiment_name')}[/]\n")
        console.print(f"  Model: [yellow]{exp.get('model_name')}[/]")
        console.print(f"  CV: [yellow]{exp.get('cv_score', 'N/A')}[/]")
        console.print(f"  LB: [yellow]{exp.get('lb_score', 'N/A')}[/]")
        console.print(f"  Metric: {exp.get('metric')}")
        console.print(f"  Status: {exp.get('status')}")
        console.print(f"  Features: {exp.get('features', [])}")
        console.print(f"  Params: {json.dumps(exp.get('params', {}), indent=2)}")
        if exp.get("feature_importance"):
            console.print(f"\n  [bold]Top Features:[/]")
            for name, imp in exp["feature_importance"][:10]:
                console.print(f"    {name}: {imp:.4f}")
        if exp.get("error_message"):
            console.print(f"\n  [red]Error: {exp['error_message'][:500]}[/]")

    elif action == "compare" and ids:
        id_list = [int(x.strip()) for x in ids.split(",") if x.strip()]
        exps = store.compare(id_list)
        if not exps:
            console.print("[yellow]No experiments found for those IDs.[/]")
            return

        console.print(f"\n[bold]Comparing {len(exps)} experiments[/]\n")
        table = Table(title="Comparison")
        table.add_column("Field", style="cyan")
        for i, e in enumerate(exps):
            table.add_column(f"#{e['id']}", style="yellow" if i == 0 else "")

        fields = [
            ("Name", "experiment_name"),
            ("Model", "model_name"),
            ("CV Score", "cv_score"),
            ("LB Score", "lb_score"),
            ("Metric", "metric"),
            ("Features", "features"),
        ]
        for label, key in fields:
            row = [label]
            for e in exps:
                val = e.get(key, "—")
                if isinstance(val, list):
                    val = f"{len(val)} items"
                elif isinstance(val, float):
                    val = f"{val:.5f}"
                row.append(str(val)[:40])
            table.add_row(*row)

        console.print(table)

    elif action == "log-lb" and exp_id is not None and lb_score is not None:
        store.update_lb(exp_id, lb_score)
        console.print(f"[green]✅ LB score {lb_score:.5f} recorded for experiment #{exp_id}[/]")

    else:
        console.print("[yellow]Usage:[/]")
        console.print("  experiments <slug> --action list")
        console.print("  experiments <slug> --action show --id <id>")
        console.print("  experiments <slug> --action compare --ids 1,2,3")
        console.print("  experiments <slug> --action log-lb --id <id> --lb <score>")


# ── suggest: generate next-step recommendations ──


@app.command()
def suggest(competition: str = typer.Argument(..., help="Competition slug, e.g. 'titanic'")):
    """Generate next-step recommendations based on experiment history."""
    from kagglemate.graph.nodes.suggest_node import run as suggest_run
    from kagglemate.graph.state import KaggleAgentState
    from kagglemate.config import config as cfg
    from pathlib import Path

    console.print(f"\n[bold cyan]💡 Strategy Advisor[/]\n")

    comp_dir = cfg.COMPETITIONS_DIR / competition
    if not comp_dir.exists():
        console.print(f"[red]No data for '{competition}'. Run 'research' first.[/]")
        raise typer.Exit(code=1)

    state: KaggleAgentState = {
        "competition_slug": competition,
        "competition_name": competition,
        "report_dir": str(comp_dir / "reports"),
        "data_dir": str(comp_dir / "data" / "raw"),
        "messages": [],
        "current_phase": "suggest",
        "errors": [],
        "best_cv_score": 0.0,
        "best_lb_score": 0.0,
        "human_approval_required": False,
        "human_approved": False,
    }

    console.print(f"  Analyzing experiment history for: [yellow]{competition}[/]\n")

    updates = suggest_run(state)
    state.update(updates)

    # Show the results
    report_dir = comp_dir / "reports"
    next_steps = report_dir / "next_steps.md"
    if next_steps.exists():
        console.print(f"[green]✅ next_steps.md → {next_steps}[/]")

        # Show quick preview
        content = next_steps.read_text()
        # Extract recommendations section
        if "## Recommended Next Experiments" in content:
            console.print(content.split("## Recommended Next Experiments")[1].split("---")[0])


# ── Entry point ──

if __name__ == "__main__":
    app()
