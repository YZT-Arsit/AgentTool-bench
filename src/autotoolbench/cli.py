from __future__ import annotations

import json
from pathlib import Path

import typer

from .agent.adaptive_agent import AdaptiveAgent
from .agent.budget import BudgetController
from .agent.react_baseline import ReactAgent
from .constants import DEFAULT_OPENAI_MODEL, DEFAULT_SEED, DEFAULT_TEMPERATURE
from .data_gen import main as generate_data
from .env import tasks as tasksmod
from .eval import ablation, report, runner
from .llm.mock import MockLLM
from .llm.openai_client import OpenAIClient
from .utils.paths import create_run_dir, logs_dir

app = typer.Typer()


@app.command()
def make_data(seed: int = DEFAULT_SEED):
    """Generate the benchmark data bundle."""
    generate_data(seed=seed)
    typer.echo("data generated")


@app.command()
def run(
    task_id: str = typer.Option(..., "--task-id", help="ID of the task to run"),
    agent: str = "adaptive",
    seed: int = DEFAULT_SEED,
    noise: float = 0.0,
    budget_preset: str = "loose",
    llm: str = "mock",
    json_error_rate: float = 0.0,
    openai_model: str = DEFAULT_OPENAI_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
):
    """Run a single task and persist its trajectory."""
    task = tasksmod.get_task(task_id)
    if not task:
        typer.echo(f"task {task_id} not found")
        raise typer.Exit(code=1)
    llm_client = (
        MockLLM(seed, noise, json_error_rate=json_error_rate)
        if llm == "mock"
        else OpenAIClient(model=openai_model, temperature=temperature)
    )
    if agent == "react":
        benchmark_agent = ReactAgent(llm_client)
    else:
        benchmark_agent = AdaptiveAgent(llm_client, budget=BudgetController.from_preset(budget_preset))
    traj = benchmark_agent.run(task, seed=seed, noise=noise, budget_mode=budget_preset)
    run_dir = create_run_dir("run", label=task_id.lower())
    logdir = logs_dir()
    logdir.mkdir(parents=True, exist_ok=True)
    latest_path = logdir / f"trajectory_{task_id}_{seed}.json"
    latest_path.write_text(traj.model_dump_json(indent=2), encoding="utf-8")
    run_path = run_dir / "trajectory.json"
    run_path.write_text(traj.model_dump_json(indent=2), encoding="utf-8")
    typer.echo(traj.model_dump_json(indent=2))
    typer.echo(f"trajectory written to {run_path}")


@app.command()
def eval(
    agent: str = "all",
    seed: int = DEFAULT_SEED,
    noise: float = 0.0,
    matrix: bool = True,
    llm: str = "mock",
    json_error_rate: float = 0.0,
    openai_model: str = DEFAULT_OPENAI_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
):
    """Run the benchmark evaluation matrix."""
    if agent == "all":
        names = ["react", "plan", "adaptive"]
    else:
        names = [agent]
    if matrix:
        matrix_result = runner.run_matrix(
            names,
            seed=seed,
            include_ablation=True,
            llm_type=llm,
            json_error_rate=json_error_rate,
            openai_model=openai_model,
            temperature=temperature,
        )
    else:
        run_dir = create_run_dir("eval")
        matrix_result = {
            "scenarios": [{"label": f"noise={noise}", "noise": noise, "budget_preset": "loose"}],
            "results": {
                f"noise={noise}": runner.run_agents_detailed(
                    names,
                    seed=seed,
                    noise=noise,
                    budget_preset="loose",
                    scenario_label=f"noise={noise}",
                    save_traces=True,
                    llm_type=llm,
                    json_error_rate=json_error_rate,
                    openai_model=openai_model,
                    temperature=temperature,
                    run_dir=run_dir,
                )
            },
            "failures": [],
            "trajectories": [],
            "run_dir": str(run_dir),
            "summary_path": str(run_dir / "summary.json"),
        }
    snapshot = runner.evaluation_snapshot(
        names,
        seed=seed,
        matrix=matrix_result,
        llm_type=llm,
        json_error_rate=json_error_rate,
        openai_model=openai_model,
        temperature=temperature,
    )
    path = report.generate(matrix_result, snapshot=snapshot)
    typer.echo(f"report saved to {path}")


@app.command()
def ablate(
    seed: int = DEFAULT_SEED,
    noise: float = 0.2,
    budget_preset: str = "tight",
    llm: str = "mock",
    json_error_rate: float = 0.0,
    openai_model: str = DEFAULT_OPENAI_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
):
    """Run the ablation suite."""
    run_dir = create_run_dir("ablation")
    res = ablation.ablate(
        seed=seed,
        noise=noise,
        budget_preset=budget_preset,
        llm_type=llm,
        json_error_rate=json_error_rate,
        openai_model=openai_model,
        temperature=temperature,
    )
    matrix_result = {
        "scenarios": [{"label": f"ablation {budget_preset}", "noise": noise, "budget_preset": budget_preset}],
        "results": {f"ablation {budget_preset}": {"summary": res, "task_type_rates": {}, "failures": [], "trajectories": []}},
        "failures": [],
        "trajectories": [],
        "run_dir": str(run_dir),
        "summary_path": str(run_dir / "summary.json"),
    }
    snapshot = {
        "seed": seed,
        "noise": [noise],
        "budget_presets": [budget_preset],
        "llm_type": llm,
        "json_error_rate": json_error_rate,
        "openai_model": openai_model if llm == "openai" else "",
        "temperature": temperature,
        "agent_version": "p2-json-chain",
        "run_dir": str(run_dir),
    }
    Path(matrix_result["summary_path"]).write_text(json.dumps(res, indent=2), encoding="utf-8")
    path = report.generate(matrix_result, snapshot=snapshot)
    typer.echo(f"ablation saved to {path}")


if __name__ == "__main__":
    app()
