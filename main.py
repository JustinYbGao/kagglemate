#!/usr/bin/env python3
"""kagglemate — LangGraph-based Kaggle competition assistant.

Usage:
    python main.py --help
    python main.py check                         # Verify setup
    python main.py list [--search <term>]        # List competitions
    python main.py inspect <slug>                # Show competition info
    python main.py research <slug>               # Full research pipeline
    python main.py profile <slug>                # Data profile only
    python main.py spec <slug>                   # Generate SPEC.md only
    python main.py baseline <slug>               # Generate baseline script
    python main.py run <slug>                    # Run training script
    python main.py suggest <slug>                # Get next-step recommendations
    python main.py experiments <slug> --action list   # View experiment history
    python main.py notebook pull <ref> -c <slug>      # Pull public notebook
    python main.py kernel <push|monitor|status> <ref> # Manage Kaggle kernels
    python main.py submission validate -c <slug> -f <file>  # Validate file
    python main.py submission submit -c <slug> -f <file>    # Submit (human gate)
    python main.py submission status -c <slug>              # Check submissions
    python main.py tune <slug> [--trials 50]                # Hyperparameter tuning
    python main.py ensemble <slug> --ids 1,2,3 [--method weighted_average]
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
from rich.panel import Panel

from kagglemate.config import config
from kagglemate.tools.kaggle_cli import KaggleCLI

app = typer.Typer(
    name="kagglemate",
    help="Kaggle competition assistant powered by DeepSeek V4 + LangGraph",
    add_completion=False,
    invoke_without_command=True,
)
console = Console()


@app.callback()
def _main_callback(ctx: typer.Context):
    """Default: start conversational agent. Use --help to see all commands."""
    if ctx.invoked_subcommand is None:
        from kagglemate.chat_agent import chat as run_chat
        run_chat()


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

    # 2. LLM key
    if config.LLM_API_KEY:
        masked = config.LLM_API_KEY[:8] + "..." + config.LLM_API_KEY[-4:]
        console.print(f"  LLM API key ({config.LLM_PROVIDER}): {masked}  [green]✓[/]")
    else:
        console.print(f"  LLM API key  [red]✗ (set LLM_API_KEY in .env)[/]")

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
def list_competitions(search: str = typer.Option("", "--search", "-s", help="Filter by search term"),
                      category: str = typer.Option("all", "--category", "-c", help="all|featured|research|playground|gettingStarted"),
                      all: bool = typer.Option(False, "--all", help="Show all including old competitions")):
    """List active Kaggle competitions (sorted by newest)."""
    try:
        comps = KaggleCLI.list_competitions(search=search, sort_by="recentlyCreated", page_size=50, category=category)
    except Exception as e:
        console.print(f"[red]Error: {e}[/]")
        raise typer.Exit(code=1)

    if not comps:
        console.print("[yellow]No competitions found.[/]")
        console.print("[dim]Kaggle API 返回有限。直接访问 kaggle.com/competitions 查看完整列表。[/]")
        return

    active = [c for c in comps if c.get("deadline", "") >= "2025"]
    old = [c for c in comps if c.get("deadline", "") < "2025"]

    table = Table(title=f"Active Competitions ({len(active)})")
    table.add_column("Slug", style="cyan")
    table.add_column("Deadline", style="yellow")
    table.add_column("Category")
    table.add_column("Reward")
    table.add_column("Teams", justify="right")

    for c in active:
        ref = c.get("ref", "?").split("/")[-1]
        table.add_row(
            ref[:40],
            (c.get("deadline", "?") or "")[:10],
            c.get("category", "?"),
            (c.get("reward", "") or "")[:25],
            c.get("teamCount", "?"),
        )

    console.print(table)

    if old and all:
        console.print(f"\n[dim]Older competitions: {len(old)} (use --all to hide)[/]")


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
    console.print(f"  Model: [dim]{config.LLM_MODEL} ({config.LLM_PROVIDER})[/]")

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


# ── notebook: pull a public kernel ──


@app.command()
def notebook(
    action: str = typer.Argument(..., help="Action: pull"),
    ref: str = typer.Argument(..., help="Kernel reference: username/kernel-name"),
    competition: str = typer.Option("", "--competition", "-c", help="Competition slug"),
    target: str = typer.Option("", "--target", "-t", help="Output directory (auto if omitted)"),
):
    """Pull a Kaggle notebook WITH metadata preservation (-m flag)."""
    from kagglemate.graph.nodes.kernel_node import run as kernel_run
    from kagglemate.graph.state import KaggleAgentState
    from kagglemate.config import config as cfg

    if action != "pull":
        console.print("[red]Only 'pull' action is supported for notebooks.[/]")
        raise typer.Exit(code=1)

    if not competition:
        console.print("[red]--competition is required (e.g. --competition titanic)[/]")
        raise typer.Exit(code=1)

    console.print(f"\n[bold cyan]📥 Pulling Notebook[/]\n")
    console.print(f"  Source: [yellow]{ref}[/]")
    console.print(f"  Competition: [yellow]{competition}[/]")

    state: KaggleAgentState = {
        "competition_slug": competition,
        "kernel_ref": ref,
        "kernel_action": "pull",
        "messages": [],
        "current_phase": "kernel",
        "errors": [],
        "best_cv_score": 0.0,
        "best_lb_score": 0.0,
        "human_approval_required": False,
        "human_approved": False,
    }

    updates = kernel_run(state)
    errors = updates.get("errors", [])

    if errors:
        console.print(f"\n[bold red]Pull completed with issues:[/]")
        for e in errors:
            console.print(f"  [red]✗ {e}[/]")
    else:
        console.print(f"\n[bold green]✅ Notebook pulled successfully![/]")

    kernel_dir = updates.get("kernel_dir", "")
    if kernel_dir:
        console.print(f"  Saved to: [cyan]{kernel_dir}[/]")
        # Show files
        p = Path(kernel_dir)
        if p.exists():
            for f in sorted(p.iterdir()):
                console.print(f"    📄 {f.name}")

    metadata = updates.get("kernel_metadata")
    if metadata:
        console.print(f"\n[bold]Metadata:[/]")
        console.print(f"  id: [yellow]{metadata.get('id', '?')}[/]")
        console.print(f"  is_private: {'✅' if metadata.get('is_private') else '⚠️'}")
        console.print(f"  competition_sources: {metadata.get('competition_sources', [])}")
        console.print(f"  enable_internet: {'⚠️' if metadata.get('enable_internet') else '✅'}")

        # Hint about what to fix
        username = config.KAGGLE_USERNAME
        if username and not metadata.get("id", "").startswith(f"{username}/"):
            console.print(f"\n[yellow]⚠ Remember to change 'id' to: {username}/<your-kernel-name>[/]")


# ── kernel: push / monitor / status ──


@app.command()
def kernel(
    action: str = typer.Argument(..., help="push | monitor | status"),
    target: str = typer.Argument("", help="Kernel ref (user/name) or path to kernel directory"),
    competition: str = typer.Option("", "--competition", "-c", help="Competition slug"),
    timeout: int = typer.Option(120, "--timeout", "-t", help="Max monitor wait (minutes)"),
):
    """Manage Kaggle kernels: push, monitor, or check status."""
    from kagglemate.graph.nodes.kernel_node import run as kernel_run, MONITOR_MAX_WAIT
    from kagglemate.graph.state import KaggleAgentState
    from kagglemate.config import config as cfg

    if action not in ("push", "monitor", "status"):
        console.print("[red]Action must be: push, monitor, or status[/]")
        raise typer.Exit(code=1)

    if not target:
        console.print("[red]Must specify kernel ref or directory path[/]")
        raise typer.Exit(code=1)

    if action == "push":
        console.print(f"\n[bold cyan]🚀 Pushing Kernel[/]\n")
        console.print(f"  Directory: [yellow]{target}[/]")

        state: KaggleAgentState = {
            "competition_slug": competition or "",
            "kernel_dir": target,
            "kernel_action": "push",
            "messages": [],
            "current_phase": "kernel",
            "errors": [],
            "best_cv_score": 0.0,
            "best_lb_score": 0.0,
            "human_approval_required": False,
            "human_approved": False,
        }

        updates = kernel_run(state)
        errors = updates.get("errors", [])

        if errors:
            console.print(f"\n[bold red]Push FAILED:[/]")
            for e in errors:
                console.print(f"  [red]✗ {e}[/]")
        else:
            kernel_ref = updates.get("kernel_ref", target)
            console.print(f"\n[bold green]✅ Kernel pushed![/]")
            console.print(f"  Ref: [yellow]{kernel_ref}[/]")
            console.print(f"\n[bold]Monitor:[/] python main.py kernel monitor {kernel_ref}[/]")

    elif action == "monitor":
        console.print(f"\n[bold cyan]👀 Monitoring Kernel[/]\n")
        console.print(f"  Kernel: [yellow]{target}[/]")
        console.print(f"  Timeout: [dim]{timeout} min[/]")

        state: KaggleAgentState = {
            "competition_slug": competition or "",
            "kernel_ref": target,
            "kernel_action": "monitor",
            "monitor_timeout": timeout * 60,  # pass through state
            "messages": [],
            "current_phase": "kernel",
            "errors": [],
            "best_cv_score": 0.0,
            "best_lb_score": 0.0,
            "human_approval_required": False,
            "human_approved": False,
        }

        updates = kernel_run(state)
        errors = updates.get("errors", [])

        if errors:
            console.print(f"\n[bold red]Monitor failed:[/]")
            for e in errors:
                console.print(f"  [red]✗ {e}[/]")

        results = updates.get("kernel_results", {})
        if results:
            console.print(f"\n[bold green]Results parsed from output:[/]")
            for k, v in results.items():
                if isinstance(v, float):
                    console.print(f"  {k}: [yellow]{v:.5f}[/]")
                else:
                    console.print(f"  {k}: [yellow]{v}[/]")

        suggestions = updates.get("error_suggestions", [])
        if suggestions:
            console.print(f"\n[bold yellow]💡 Suggestions:[/]")
            for s in suggestions:
                console.print(f"  • {s}")

    elif action == "status":
        console.print(f"\n[bold]Kernel Status: [yellow]{target}[/][/]\n")

        state: KaggleAgentState = {
            "competition_slug": competition or "",
            "kernel_ref": target,
            "kernel_action": "status",
            "messages": [],
            "current_phase": "kernel",
            "errors": [],
            "best_cv_score": 0.0,
            "best_lb_score": 0.0,
            "human_approval_required": False,
            "human_approved": False,
        }

        updates = kernel_run(state)
        errors = updates.get("errors", [])

        if errors:
            for e in errors:
                console.print(f"  [red]✗ {e}[/]")
        else:
            status = updates.get("kernel_status", "unknown")
            console.print(f"  Status: [yellow]{status}[/]")


# ── submission: validate / submit / status ──


@app.command()
def submission(
    action: str = typer.Argument(..., help="validate | submit | status"),
    competition: str = typer.Option("", "--competition", "-c", help="Competition slug"),
    file: str = typer.Option("", "--file", "-f", help="Path to submission CSV"),
    message: str = typer.Option("kagglemate submission", "--message", "-m", help="Kaggle submission message"),
    exp_id: int = typer.Option(None, "--exp-id", help="Experiment ID to link this submission to"),
):
    """Validate, submit, or check status of Kaggle submissions.

    Submit ALWAYS requires human confirmation (type YES).
    """
    from kagglemate.tools.submission_validator import validate
    from kagglemate.tools.kaggle_cli import KaggleCLI
    from kagglemate.memory.experiment_store import ExperimentStore
    from kagglemate.config import config as cfg
    from pathlib import Path

    if action not in ("validate", "submit", "status"):
        console.print("[red]Action must be: validate, submit, or status[/]")
        raise typer.Exit(code=1)

    if action == "validate":
        if not file or not competition:
            console.print("[red]--competition and --file are required for validation[/]")
            raise typer.Exit(code=1)

        console.print(f"\n[bold]🔍 Validating: [yellow]{file}[/] for [yellow]{competition}[/][/]\n")

        data_dir = cfg.COMPETITIONS_DIR / competition / "data" / "raw"
        vr = validate(file, str(data_dir))

        # Show all checks
        for c in vr.checks:
            icon = "[green]✓[/]" if c.passed else "[red]✗[/]"
            console.print(f"  {icon} {c.check}: {c.detail}")

        if vr.warnings:
            console.print(f"\n[yellow]Warnings:[/]")
            for w in vr.warnings:
                console.print(f"  ⚠ {w}")

        if vr.errors:
            console.print(f"\n[red]❌ Validation FAILED ({len(vr.errors)} errors):[/]")
            for e in vr.errors:
                console.print(f"  ✗ {e}")
        else:
            console.print(f"\n[green]✅ Validation passed![/]")
            console.print(f"  File is ready for submission.")
            console.print(f"  Run: python main.py submission submit -c {competition} -f {file}")

    elif action == "submit":
        if not file or not competition:
            console.print("[red]--competition and --file are required[/]")
            raise typer.Exit(code=1)

        file_path = Path(file)
        if not file_path.exists():
            console.print(f"[red]File not found: {file}[/]")
            raise typer.Exit(code=1)

        data_dir = cfg.COMPETITIONS_DIR / competition / "data" / "raw"

        # ── Step 1: Validate ──
        console.print(f"\n[bold]🔍 Pre-submission validation...[/]\n")
        vr = validate(str(file_path), str(data_dir))

        for c in vr.checks:
            icon = "[green]✓[/]" if c.passed else "[red]✗[/]"
            console.print(f"  {icon} {c.check}: {c.detail}")

        if vr.errors:
            console.print(f"\n[red]❌ Cannot submit — {len(vr.errors)} validation errors.[/]")
            for e in vr.errors:
                console.print(f"  ✗ {e}")
            console.print(f"\n[yellow]Fix the issues above before submitting.[/]")
            raise typer.Exit(code=1)

        if vr.warnings:
            console.print(f"\n[yellow]⚠ Warnings (non-blocking):[/]")
            for w in vr.warnings:
                console.print(f"  • {w}")

        # ── Step 2: Show preview ──
        console.print(f"\n[bold]┌{'─'*58}┐[/]")
        console.print(f"[bold]│[/] [bold white]SUBMISSION PREVIEW — REVIEW BEFORE CONFIRMING[/]     [bold]│[/]")
        console.print(f"[bold]├{'─'*58}┤[/]")
        console.print(f"[bold]│[/]  Competition: [yellow]{competition:<46}[/][bold]│[/]")
        console.print(f"[bold]│[/]  File:       [cyan]{file_path.name:<46}[/][bold]│[/]")
        console.print(f"[bold]│[/]  Message:    [dim]{message[:46]:<46}[/][bold]│[/]")

        # Show experiment info if available
        if exp_id:
            store = ExperimentStore(competition)
            exp = store.get(exp_id)
            if exp:
                cv = f"{exp.get('cv_score', 'N/A')}"
                console.print(f"[bold]│[/]  Experiment: [yellow]#{exp_id} (CV: {cv})[/]  [bold]│[/]")

        console.print(f"[bold]├{'─'*58}┤[/]")
        console.print(f"[bold]│[/] [red]⚠ This uses a Kaggle submission slot.[/]             [bold]│[/]")
        console.print(f"[bold]│[/] [red]⚠ Early scores are inflated — wait 4+ hours.[/]     [bold]│[/]")
        console.print(f"[bold]│[/] [red]⚠ Have you checked rules_checklist.md?[/]             [bold]│[/]")
        console.print(f"[bold]└{'─'*58}┘[/]")

        # ── Step 3: HUMAN GATE ──
        console.print()
        answer = typer.prompt("Type YES to confirm submission", default="NO")
        if answer.strip() != "YES":
            console.print("[yellow]Submission cancelled.[/]")
            raise typer.Exit(code=0)

        # ── Step 4: Submit ──
        console.print(f"\n[bold]🚀 Submitting...[/]\n")
        try:
            result = KaggleCLI.submit(competition, file_path, message)
            console.print(f"[green]✅ Submitted![/]")
            console.print(f"  {result.get('stdout', '')[:200]}")
        except RuntimeError as e:
            console.print(f"[red]❌ Submission failed: {e}[/]")
            raise typer.Exit(code=1)

        # ── Step 5: Link to experiment ──
        if exp_id:
            try:
                store = ExperimentStore(competition)
                store.update_field(exp_id, "submission_path", str(file_path))
                console.print(f"\n[green]Linked to experiment #{exp_id}[/]")
            except Exception:
                pass

        # ── Step 6: Show next steps ──
        console.print(f"\n[bold]Next:[/]")
        console.print(f"  Check status:  python main.py submission status -c {competition}")
        console.print(f"  Record LB:     python main.py experiments {competition} --action log-lb --id <id> --lb <score>")
        console.print(f"\n[yellow]Reminder: wait 4+ hours for score to stabilize before judging.[/]")

    elif action == "status":
        if not competition:
            console.print("[red]--competition is required[/]")
            raise typer.Exit(code=1)

        console.print(f"\n[bold]📊 Submission Status: [yellow]{competition}[/][/]\n")

        try:
            subs = KaggleCLI.submissions(competition)
        except Exception as e:
            console.print(f"[red]Failed to fetch submissions: {e}[/]")
            raise typer.Exit(code=1)

        if not subs:
            console.print("[dim]No submissions yet.[/]")
            return

        from rich.table import Table
        table = Table(title=f"Recent Submissions — {competition}")
        table.add_column("Date", style="dim")
        table.add_column("Description")
        table.add_column("Score", justify="right", style="yellow")
        table.add_column("Status", style="green")

        for s in subs[:10]:
            table.add_row(
                (s.get("date", "") or "")[:19],
                (s.get("description", "") or "")[:50],
                s.get("publicScore", "—") or "pending",
                s.get("status", "?"),
            )

        console.print(table)


# ── tune: hyperparameter optimization ──


@app.command()
def tune(
    competition: str = typer.Argument(..., help="Competition slug"),
    trials: int = typer.Option(50, "--trials", "-n", help="Number of Optuna trials"),
    run_after: bool = typer.Option(False, "--run", help="Run tuning script after generating"),
):
    """Run Optuna hyperparameter tuning for a competition."""
    from kagglemate.graph.nodes.tune_node import run as tune_run
    from kagglemate.graph.nodes.init_node import run as init_run
    from kagglemate.graph.nodes.analyze_node import run as analyze_run
    from kagglemate.graph.nodes.run_node import run as run_node_fn
    from kagglemate.graph.state import KaggleAgentState
    from kagglemate.config import config as cfg
    from pathlib import Path

    console.print(f"\n[bold cyan]🎯 Hyperparameter Tuning[/]\n")
    console.print(f"  Competition: [yellow]{competition}[/]")
    console.print(f"  Trials: [yellow]{trials}[/]")

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
        "tune_trials": trials,
        "messages": [],
        "current_phase": "build",
        "errors": [],
        "best_cv_score": 0.0,
        "best_lb_score": 0.0,
        "human_approval_required": False,
        "human_approved": False,
    }

    # Run analyze to get data profile
    analyze_state = analyze_run(state)
    state.update(analyze_state)

    console.print(f"  Task type: [yellow]{state.get('competition_type')}[/]")
    console.print(f"  Target: [yellow]{state.get('data_profile', {}).get('target_col', '?')}[/]")

    # Generate tuning script
    updates = tune_run(state)
    state.update(updates)

    exp = state.get("current_experiment", {})
    script_path = exp.get("script_path", "")
    if script_path:
        console.print(f"\n[green]✅ Tuning script → {script_path}[/]")
        console.print(f"  Model: [yellow]{exp.get('model')}[/]")
        console.print(f"  Trials: [yellow]{trials}[/]")

        if run_after:
            console.print(f"\n[bold]Running tuning (this may take minutes)...[/]\n")
            run_state = {**state, "current_experiment": exp}
            run_updates = run_node_fn(run_state)
            run_state.update(run_updates)

            cv = run_state.get("current_experiment", {}).get("cv_score", 0.0)
            if cv > 0:
                console.print(f"\n[green]✅ Tuning complete! CV: {cv:.5f}[/]")
        else:
            console.print(f"\n[bold]Next:[/] python main.py run {competition}")
    else:
        console.print("[red]Failed to generate tuning script.[/]")


# ── ensemble: blend experiment submissions ──


@app.command()
def ensemble(
    competition: str = typer.Argument(..., help="Competition slug"),
    ids: str = typer.Option(..., "--ids", help="Comma-separated experiment IDs, e.g. '1,2,3'"),
    method: str = typer.Option("weighted_average", "--method", "-m",
                                help="simple_average | weighted_average | rank_average"),
):
    """Blend multiple experiment submissions into one."""
    from kagglemate.graph.nodes.ensemble_node import run as ensemble_run
    from kagglemate.graph.state import KaggleAgentState
    from kagglemate.config import config as cfg

    exp_ids = [int(x.strip()) for x in ids.split(",") if x.strip()]
    if len(exp_ids) < 2:
        console.print("[red]Need at least 2 experiment IDs (--ids 1,2,3)[/]")
        raise typer.Exit(code=1)

    console.print(f"\n[bold cyan]🔀 Ensemble Blending[/]\n")
    console.print(f"  Competition: [yellow]{competition}[/]")
    console.print(f"  Experiments: [yellow]{exp_ids}[/]")
    console.print(f"  Method: [yellow]{method}[/]")

    comp_dir = cfg.COMPETITIONS_DIR / competition
    state: KaggleAgentState = {
        "competition_slug": competition,
        "ensemble_exp_ids": exp_ids,
        "ensemble_method": method,
        "submission_dir": str(comp_dir / "submissions"),
        "data_dir": str(comp_dir / "data" / "raw"),
        "messages": [],
        "current_phase": "build",
        "errors": [],
        "best_cv_score": 0.0,
        "best_lb_score": 0.0,
        "human_approval_required": False,
        "human_approved": False,
    }

    updates = ensemble_run(state)
    state.update(updates)

    errors = state.get("errors", [])
    exp = state.get("current_experiment", {})

    if errors:
        console.print(f"\n[red]❌ Ensemble failed:[/]")
        for e in errors:
            console.print(f"  {e}")
    else:
        console.print(f"\n[bold green]✅ Ensemble complete![/]")
        console.print(f"  Submission: [cyan]{exp.get('submission_path')}[/]")
        console.print(f"  Experiment ID: [yellow]#{exp.get('id')}[/]")

        # Validate
        sub_path = exp.get("submission_path", "")
        if sub_path:
            from kagglemate.tools.submission_validator import validate
            data_dir = str(comp_dir / "data" / "raw")
            vr = validate(sub_path, data_dir)
            if vr.is_valid:
                console.print(f"  Validation: [green]✓ passed[/]")
                console.print(f"\n[bold]Next:[/] python main.py submission submit -c {competition} -f {sub_path}")
            else:
                console.print(f"  Validation: [red]✗ {len(vr.errors)} errors[/]")


# ── types: show competition type registry ──


@app.command()
def types():
    """Show all competition types and what the agent can do for each."""
    from kagglemate.competition_registry import COMPETITION_TYPES
    from rich.table import Table

    table = Table(title="Competition Types / 比赛类型注册表")
    table.add_column("Type / 类型", style="cyan")
    table.add_column("Detection / 检测")
    table.add_column("Baseline")
    table.add_column("Tune / 调参")
    table.add_column("Ensemble / 集成")
    table.add_column("Submit / 提交")
    table.add_column("Research / 调研")

    for type_id, ct in COMPETITION_TYPES.items():
        detection = ", ".join(ct.detection_extensions[:3])
        if not detection:
            detection = "(手动)"
        table.add_row(
            f"{ct.name_zh}\n[dim]{type_id}[/]",
            detection,
            "✅" if ct.can_baseline else "—",
            "✅" if ct.can_tune else "—",
            "✅" if ct.can_ensemble else "—",
            "✅" if ct.can_submit else "—",
            "✅" if ct.can_research else "—",
        )

    console.print(table)
    console.print(f"\n[dim]要添加新类型: 编辑 kagglemate/competition_registry.py 中的 COMPETITION_TYPES 字典。[/]")


# ── harness: view safety harness status / audit log ──


@app.command()
def harness(action: str = typer.Argument("status", help="status | audit | clear")):
    """View agent harness status or audit trail."""
    from kagglemate.harness import AuditTrail, SessionBudget
    from kagglemate.harness import TOOL_RISK_LEVELS

    if action == "status":
        audit = AuditTrail()
        console.print(Panel(
            f"审计日志: {audit.count()} 条\n"
            f"审计文件: {audit.log_path}\n"
            f"工具风险分级: {len(TOOL_RISK_LEVELS)} 个工具已分类",
            title="Harness / 安全护栏",
            border_style="blue",
        ))
        console.print("\n[bold]风险等级分布:[/]")
        from kagglemate.harness import RiskLevel
        for level in [RiskLevel.SAFE, RiskLevel.READ_ONLY, RiskLevel.SIDE_EFFECT, RiskLevel.DANGEROUS, RiskLevel.CRITICAL]:
            tools = [t for t, l in TOOL_RISK_LEVELS.items() if l == level]
            if tools:
                icons = {RiskLevel.SAFE: "🟢", RiskLevel.READ_ONLY: "🔵", RiskLevel.SIDE_EFFECT: "🟡",
                         RiskLevel.DANGEROUS: "🔴", RiskLevel.CRITICAL: "⛔"}
                console.print(f"  {icons.get(level, '•')} {level}: {', '.join(tools[:6])}")

    elif action == "audit":
        audit = AuditTrail()
        entries = audit.recent(30)
        if not entries:
            console.print("[dim]暂无审计记录。[/]")
            return
        from rich.table import Table
        table = Table(title=f"Audit Trail ({len(entries)} entries)")
        table.add_column("Time", style="dim")
        table.add_column("Tool")
        table.add_column("Risk")
        table.add_column("OK")
        table.add_column("Blocked")
        table.add_column("Summary")
        for e in entries:
            table.add_row(
                e["timestamp"][11:19],
                e["tool"][:22],
                e["risk_level"][:10],
                "✓" if e["success"] else "✗",
                "⛔" if e["blocked"] else "",
                e.get("result_summary", "")[:50],
            )
        console.print(table)

    elif action == "clear":
        audit = AuditTrail()
        audit.log_path.unlink(missing_ok=True)
        console.print("[green]✅ Audit log cleared.[/]")

    else:
        console.print("[yellow]Usage: python main.py harness [status|audit|clear][/]")


# ── chat: conversational agent (default) ──


@app.command()
def chat():
    """Start the conversational KaggleMate agent (默认对话模式)."""
    from kagglemate.chat_agent import chat as run_chat
    run_chat()


# ── Entry point ──

if __name__ == "__main__":
    app()
