import json, logging, typer, pathlib
from rich import print as rprint

from .config import Config
from .core import model_loader, activations
from .analyzers import get as get_analyzer, all as all_analyzers
from .benchmarks import mib_adapter
from .visuals import graphviz_export

log = logging.getLogger("mechamap")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

app = typer.Typer(help="MechaMap 2025 CLI")

### ───────────────────────── BASIC SCAN ───────────────────────── ###

@app.command()
def scan(
    model: str,
    analyzer: str = typer.Option("baseline", help=f"Analyzer ({', '.join(all_analyzers())})"),
    out: pathlib.Path = typer.Option("out.json", help="Output file"),
):
    cfg = Config()
    mdl = model_loader.load(model)
    text = cfg.default_text
    toks = mdl.to_str_tokens(text)
    acts = activations.grab_post_mlp(mdl, text)

    anal = get_analyzer(analyzer)()
    result = anal.run(mdl, toks, acts, cfg)

    out.write_text(json.dumps(result, indent=2))
    rprint(f"[bold green]Saved → {out}")

### ─────────────────── CIRCUIT VIS + BENCHMARK ─────────────────── ###

@app.command()
def viz(json_path: pathlib.Path):
    data = json.loads(json_path.read_text())
    if "edges" not in data:
        typer.echo("No edges to visualise"); raise typer.Exit()

    dot = graphviz_export.render(data["edges"],
                                 json_path.with_suffix(".dot"))
    rprint(f"[cyan]DOT saved → {dot}")

@app.command()
def evaluate_mib(json_path: pathlib.Path, task: str):
    data = json.loads(json_path.read_text())
    scores = mib_adapter.evaluate(data, task)
    rprint(scores)

@app.command()
def serve_dashboard(json_path: pathlib.Path):
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "streamlit", "run",
                    "mechamap/visuals/dashboard.py", "--",
                    "--", str(json_path)])

if __name__ == "__main__":
    app()
