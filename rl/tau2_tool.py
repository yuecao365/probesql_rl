"""Expose a τ²-bench telecom episode to veRL as a set of tools.

veRL drives the agent loop and calls a tool per action; τ² owns the episode and wants
`reset()` / `step(action_string)`. The bridge is one `Tau2Env` per rollout, keyed by verl's
`instance_id`, with one `Tau2Tool` registered per agent-side tool name so verl's tool parser
sees the same thirteen functions the benchmark advertises.

Two things are deliberate.

The per-turn process signal is computed here, in `execute`, because that is the only place
that sees the environment between actions. `progress_k` is the fraction of the task's
`env_assertions` currently satisfied — the assertions are deterministic, side-effect free
and callable mid-episode, which was verified before this design was chosen — and the reward
handed back is `Δ_k = progress_k − progress_{k−1}`. Being a difference, it telescopes to the
final progress, so no sequence of actions can collect more process reward than the state it
ends on; the anti-farming property is structural rather than a penalty bolted on afterwards.

The user simulator stays on the DeepSeek API. It is part of the environment, not a
configuration knob: a local Qwen3-4B was measured and, among episodes that terminated
normally, solved 0.0% against DeepSeek's 38.7%. Swapping it invalidates the SFT data and
every number measured so far.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


class Tau2Env:
    """One τ² episode, plus the running progress needed for the per-turn signal."""

    def __init__(self, domain: str, task_id: str, user_llm: str, user_llm_args: dict):
        from tau2.gym import AgentGymEnv

        self.env = AgentGymEnv(
            domain=domain, task_id=task_id, solo_mode=False,
            user_llm=user_llm, user_llm_args=user_llm_args,
        )
        self.observation, self.info = self.env.reset()
        # Measured after reset, never assumed to be zero: a task's initial_state can already
        # satisfy some of its assertions, and starting the counter at 0 would pay the policy
        # a positive delta on its first turn for doing nothing.
        self.progress = self._progress()
        self.terminated = False
        self.turn = 0

    def _progress(self) -> float:
        """Fraction of the task's env assertions currently satisfied.

        Read-only: every assertion in this domain is a property read on device state, so
        polling it between turns does not perturb the episode.
        """
        task = self.env._get_task()
        assertions = (task.evaluation_criteria.env_assertions or []) if task else []
        if not assertions:
            return 0.0
        env = self.env._orchestrator.environment
        met = sum(1 for a in assertions if env.run_env_assertion(a, raise_assertion_error=False))
        return met / len(assertions)

    def step(self, action: str) -> tuple[str, float, bool, dict]:
        """Returns (observation, Δ progress, terminated, metrics)."""
        obs, reward, terminated, _truncated, info = self.env.step(action)
        self.turn += 1
        self.terminated = terminated
        before = self.progress
        self.progress = self._progress()
        delta = self.progress - before
        return obs, delta, terminated, {
            "turn": self.turn,
            "progress": self.progress,
            "delta": delta,
            "terminal_reward": reward if terminated else None,
        }

    def terminal_reward(self) -> float:
        """The official rule, unchanged: the product of the task's env assertions."""
        info = self.env._get_info().get("reward_info")
        if info is None:
            return 0.0
        return float(getattr(info, "reward", 0.0) or 0.0)

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:  # a torn-down episode must not take the rollout with it
            logger.debug("close failed for a tau2 env", exc_info=True)


class Tau2ToolRegistry:
    """Episodes shared across the tools of one rollout, keyed by verl's instance_id."""

    _envs: dict[str, Tau2Env] = {}
    _lock = asyncio.Lock()

    @classmethod
    async def create(cls, instance_id: str, domain: str, task_id: str,
                     user_llm: str, user_llm_args: dict) -> Tau2Env:
        async with cls._lock:
            if instance_id not in cls._envs:
                cls._envs[instance_id] = await asyncio.to_thread(
                    Tau2Env, domain, task_id, user_llm, user_llm_args
                )
            return cls._envs[instance_id]

    @classmethod
    def get(cls, instance_id: str) -> Optional[Tau2Env]:
        return cls._envs.get(instance_id)

    @classmethod
    async def release(cls, instance_id: str) -> None:
        async with cls._lock:
            env = cls._envs.pop(instance_id, None)
        if env:
            await asyncio.to_thread(env.close)


def _base_tool_class():
    from verl.tools.base_tool import BaseTool

    class Tau2Tool(BaseTool):
        """One of τ²'s agent-side tools, backed by the episode for this instance_id."""

        def __init__(self, config: dict, tool_schema):
            super().__init__(config, tool_schema)
            self.tool_name = tool_schema.function.name
            self.domain = config.get("domain", "telecom")
            self.user_llm = config.get("user_llm", "deepseek/deepseek-chat")
            self.user_llm_args = config.get("user_llm_args", {"temperature": 0.0})
            # Weights for the per-turn signal; w_delta=1 and the rest zero reproduces
            # outcome-only training, which is how arm 2 and arm 3 differ.
            self.w_delta = float(config.get("w_delta", 1.0))
            self.w_terminal = float(config.get("w_terminal", 1.0))

        async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, Any]:
            from verl.tools.schemas import ToolResponse

            instance_id = instance_id or str(uuid4())
            task_id = kwargs.get("task_id") or kwargs.get("extra_info", {}).get("task_id")
            if task_id is None:
                raise ValueError("tau2 needs a task_id; pass it through the dataset's extra_info")
            env = await Tau2ToolRegistry.create(
                instance_id, self.domain, task_id, self.user_llm, self.user_llm_args
            )
            return instance_id, ToolResponse(text=env.observation or "")

        async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs):
            import json

            from verl.tools.schemas import ToolResponse

            env = Tau2ToolRegistry.get(instance_id)
            if env is None:
                return ToolResponse(text="Error: episode not started"), 0.0, {"error": "no_env"}
            if env.terminated:
                return ToolResponse(text="Error: episode already ended"), 0.0, {"error": "ended"}

            # tau2's gym takes an action string; a JSON tool call is one of the forms it parses.
            action = json.dumps({"name": self.tool_name, "arguments": parameters})
            obs, delta, terminated, metrics = await asyncio.to_thread(env.step, action)
            reward = self.w_delta * delta
            if terminated:
                reward += self.w_terminal * env.terminal_reward()
            return ToolResponse(text=obs or ""), reward, metrics

        async def calc_reward(self, instance_id: str, **kwargs) -> float:
            env = Tau2ToolRegistry.get(instance_id)
            return env.terminal_reward() if env else 0.0

        async def release(self, instance_id: str, **kwargs) -> None:
            await Tau2ToolRegistry.release(instance_id)

    return Tau2Tool


def tool_schemas(domain: str = "telecom") -> list[dict]:
    """The agent-side tool schemas, straight from the benchmark rather than hand-copied."""
    from tau2.registry import registry

    env = registry.get_env_constructor(domain)(solo_mode=False)
    return [t.openai_schema for t in env.get_tools()]
