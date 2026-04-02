from __future__ import annotations

from typing import Any, Dict

from .runner import ABLATION_AGENTS, run_agents


def ablate(
    seed: int = 0,
    noise: float = 0.2,
    budget_preset: str = "tight",
    llm_type: str = "mock",
    json_error_rate: float = 0.0,
    openai_model: str = "gpt-4o-mini",
    temperature: float = 0.0,
) -> Dict[str, Dict[str, Any]]:
    return run_agents(
        ["adaptive", *ABLATION_AGENTS],
        seed=seed,
        noise=noise,
        budget_preset=budget_preset,
        scenario_label=f"ablation-{budget_preset}",
        save_traces=True,
        llm_type=llm_type,
        json_error_rate=json_error_rate,
        openai_model=openai_model,
        temperature=temperature,
    )
