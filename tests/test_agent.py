from autotoolbench.agent.adaptive_agent import AdaptiveAgent
from autotoolbench.data_gen import main as generate_data
from autotoolbench.data_gen import reset_runtime_artifacts
from autotoolbench.env.tasks import get_task, load_tasks
from autotoolbench.llm.mock import MockLLM


def test_adaptive_agent_runs_task():
    generate_data(seed=0)
    reset_runtime_artifacts()
    tasks = load_tasks()
    assert tasks

    llm = MockLLM(seed=0, noise=0)
    agent = AdaptiveAgent(llm)
    task = get_task("T001")
    assert task is not None

    traj = agent.run(task)

    assert traj.task_id == "T001"
    assert traj.success is True
    assert len(traj.steps) >= 1
    assert traj.steps[0].tool == "sql_query"
