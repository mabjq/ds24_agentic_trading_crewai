import asyncio
import logging
import uuid
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

import litellm
from crewai import Agent, Task, Crew, Process, LLM
from rich.console import Console
from rich.markdown import Markdown
import yaml

from config.config import AppConfig
from config.utils import safe_dump_config
from tools.load_csv import wrapper_load_csv
from tools.analyze_signals import wrapper_analyze_signals
from tools.human_input import human_input_tool
from tools.trade_logic import wrapper_trade_logic
from tools.backtest_tool import run_backtest_tool
from tools.optimize_params import optimize_params_tool

logger = logging.getLogger(__name__)
console = Console()


class TradingAgents:
    """Manages CrewAI crew for trading pipeline.
    Sequential crew: DataAgent → IndicatorAgent → TraderAgent → BacktestAgent → TradeoptAgent → UserProxy (approve/recommend manual config changes).
    Loads full config from /config/agents.yaml; uses xAI via OpenAI-compatible client.
    """

    def __init__(self, config: AppConfig) -> None:
        """Initiate TradingAgents with config and LLM setup.

        Args:
            config: Application configuration for API keys and trading params.
        """
        load_dotenv()
        self.config = config
        if not config.api.xai_api_key:
            raise ValueError("xAI API key required for crew run. Set XAI_API_KEY in .env.")

        yaml_path = Path(__file__).parent.parent / "config" / "agents.yaml"
        self.agents_config = self._load_yaml(yaml_path)
        self.llm_config = {
            "model": "xai/grok-4",
            "api_key": config.api.xai_api_key,
            "base_url": "https://api.x.ai/v1",
            "temperature": 0.2,
            "max_tokens": 2048,
        }
        logger.info(
            f"LLM config loaded: model={self.llm_config['model']}, "
            f"base_url={self.llm_config['base_url'][:20]}... (key hidden)"
        )
        logger.info("CrewAI crew initialized with xAI LLM and YAML config for KC=F pipeline")

    def _load_yaml(self, yaml_path: Path) -> Dict[str, Any]:
        """Load crew config from YAML file.

        Args:
            yaml_path: Path to agents.yaml.

        Returns:
            Dict[str, Any]: Loaded YAML config.
        """
        try:
            with open(yaml_path, "r") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"YAML file not found: {yaml_path}. Falling back to empty config.")
            return {}

    def test_llm_connection(self) -> str:
        """Test xAI LLM connection with a simple completion call. Fallback to grok-3-mini if grok-4 fails.

        Returns:
            str: Response from LLM test.
        """
        models_to_try = [self.llm_config["model"], "xai/grok-3-mini"]
        for model in models_to_try:
            try:
                response = litellm.completion(
                    model=model,
                    messages=[{"role": "user", "content": "Say 'xAI connected!'"}],
                    api_key=self.llm_config["api_key"],
                    api_base=self.llm_config["base_url"],
                )
                used_model = model
                logger.info(f"xAI LLM test successful with {used_model}.")
                if model != self.llm_config["model"]:
                    logger.warning(
                        f"Fallback to {used_model} due to quota/limits on {self.llm_config['model']}."
                    )
                self.llm_config["model"] = used_model
                return response.choices[0].message.content
            except Exception as e:
                logger.warning(f"xAI LLM test failed for {model}: {e}")
                continue
        raise ValueError(f"LLM connection error: All models failed. Check key and LiteLLM config.")

    async def setup_crew(self) -> Crew:
        """Setup sequential crew with agents and tasks from YAML, with context chaining.

        Returns:
            Crew: Configured CrewAI crew instance.
        """
        agents_config = self.agents_config.get("agents", {})
        tasks_config = self.agents_config.get("tasks", {})
        llm_instance = LLM(**self.llm_config)
        agents = {}
        for agent_key, agent_data in agents_config.items():
            tools_list = []
            for tool_name in agent_data.get("tools", []):
                if tool_name == "wrapper_load_csv":
                    tools_list.append(wrapper_load_csv)
                elif tool_name == "wrapper_analyze_signals":
                    tools_list.append(wrapper_analyze_signals)
                elif tool_name == "wrapper_trade_logic":
                    tools_list.append(wrapper_trade_logic)
                elif tool_name == "human_input_tool":
                    tools_list.append(human_input_tool)
                elif tool_name == "run_backtest_tool":
                    tools_list.append(run_backtest_tool)
                elif tool_name == "optimize_params_tool":
                    tools_list.append(optimize_params_tool)
            agents[agent_key] = Agent(
                role=agent_data.get("role"),
                goal=agent_data.get("goal"),
                backstory=agent_data.get("backstory"),
                tools=tools_list,
                llm=llm_instance,
                verbose=agent_data.get("verbose", False),
                allow_delegation=False,
            )
        task_dependencies = {}
        for task_key, task_data in tasks_config.items():
            agent = agents.get(task_data.get("agent"))
            if not agent:
                logger.warning(f"Agent {task_data.get('agent')} not found for task {task_key}")
                continue
            context = []
            for ctx_key in task_data.get("context", []):
                if ctx_key in task_dependencies:
                    context.append(task_dependencies[ctx_key])
            description = task_data.get("description", "")
            task = Task(
                description=description,
                agent=agent,
                context=context,
                expected_output="Concise JSON with signals, trades, and recommendations (truncate data to avoid token limits).",
                max_execution_time=300,
            )
            # Flatten for nested contexts
            if "backtest_results" in description:
                task.description += " Flatten nested dicts (e.g., backtest_results to {'winrate': value})."
            task_dependencies[task_key] = task
        ordered_tasks = [
            task_dependencies[task_key]
            for task_key in tasks_config.keys()
            if task_key in task_dependencies
        ]
        try:
            crew = Crew(
                agents=list(agents.values()),
                tasks=ordered_tasks,
                process=Process.sequential,
                verbose=True,
                memory=False,
            )
            logger.info(
                "Crew setup complete: Sequential flow from YAML config with context chaining and prompted config passing"
            )
            return crew
        except Exception as e:
            logger.error(f"Crew initialization failed: {e}.")
            raise

    async def run_basic_pipeline(
        self, task: str = "Load KC=F CSV and analyze for optimizations", max_retries: int = 1, max_runs: int = 3
    ) -> Dict[str, Any]:
        """Async run of crew pipeline with retry logic, user input pause, and feedback loop (multi-run on non-quit input).

        Args:
            task: Task description for the crew.
            max_retries: Number of retries per run.
            max_runs: Number of pipeline runs.

        Returns:
            Dict[str, Any]: Aggregated results with extracted thoughts for logging.
        """
        self.test_llm_connection()
        run_count = 0
        all_results = []
        previous_feedback = ""
        while run_count < max_runs:
            run_count += 1
            for attempt in range(max_retries):
                try:
                    crew = await self.setup_crew()
                    config_dict = safe_dump_config(self.config)
                    inputs = {"task": task, "config": config_dict} 
                    logger.info(f"Inputs config keys: {list(inputs['config'].keys())}") 
                    logger.info(f"Trading subdict size: {len(inputs['config'].get('trading', {}))}")  
                    inputs = {"task": task, "full_strategy_params": config_dict}
                    if run_count > 1 and previous_feedback:
                        inputs["user_feedback"] = previous_feedback
                    result = crew.kickoff(inputs=inputs)
                    session_id = str(uuid.uuid4())
                    logger.info(
                        f"Crew pipeline executed (run {run_count}): Task '{task}' complete (session: {session_id})"
                    )
                    console.print(Markdown(f"### Pipeline Result (Run {run_count})\n{result}"))
                    all_results.append({"run": run_count, "result": result, "session_id": session_id})

                    # Fallback user input pause (feeds into next run if not quit)
                    user_input = input("\nAdditional feedback after crew? (Enter to continue or 'q' to quit): ")
                    if user_input.lower() == "q":
                        return {"status": "user_quit", "runs": all_results, "final_feedback": user_input}
                    previous_feedback = user_input

                    break  # Success, continue loop
                except KeyboardInterrupt:
                    logger.info("Pipeline interrupted by user (CTRL+C).")
                    return {"status": "interrupted", "runs": all_results}
                except Exception as e:
                    logger.error(f"Pipeline attempt {attempt + 1} (run {run_count}) failed: {str(e)}")
                    if attempt == max_retries - 1:
                        logger.error("Max retries exceeded. Check xAI API key and LiteLLM config.")
                        raise
                    await asyncio.sleep(2**attempt)
            if run_count < max_runs:
                continue_input = input(f"\nRun {run_count + 1}? (y/n): ")
                if continue_input.lower() != "y":
                    break
        return {"status": "success", "runs": all_results, "final_feedback": previous_feedback}


if __name__ == "__main__":
    try:
        from app.logger import setup_logging
    except ImportError:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        setup_logging = lambda *args, **kwargs: None

    config = AppConfig()
    setup_logging(log_path=config.logging.app_log_path, level=config.logging.log_level)
    agents_instance = TradingAgents(config)
    asyncio.run(agents_instance.run_basic_pipeline())