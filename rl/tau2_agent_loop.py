"""Drive a tau2-bench episode as a veRL agent loop.

veRL's built-in `ToolAgentLoop` models the environment as a set of tools and treats a tool
result as the "user" turn. tau2 does not fit that: the agent reaches the environment through
thirteen backend tools, but it reaches the *customer* by speaking plainly, and the customer
is an LLM that answers back and operates thirty device tools of its own. Forcing the customer
into a tool would change the action space the SFT checkpoint was trained on -- it learned to
address the user with an ordinary assistant message -- and a rollout format that differs from
the training format is the one mismatch this project has already paid for once.

So this registers its own loop. tau2's gym wrapper already owns the hard parts: `step()`
takes either a JSON tool call or plain text, routes the former to the environment and the
latter to the user simulator, and decides termination and reward. What is left here is the
part veRL needs -- turning the policy's tokens into an action, and the environment's reply
back into tokens that line up with what the trainer will later see.

Token bookkeeping is deliberately borrowed rather than reinvented. `apply_chat_template`
plus `turn_separator` on the base class is how veRL renders an incremental turn so that the
concatenated sequence matches a single render of the whole conversation; the Qwen3 template
is not prefix-stable, and this project has already been bitten by assuming it was.

`response_logprobs` is carried through because it is the rollout engine's own view of the
sequence. Comparing it against the trainer's recomputed logprobs is the measurement behind
the TIS arm; it costs nothing to keep and cannot be recovered afterwards.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.experimental.agent_loop.tool_parser import ToolParser
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__file__)


def _action_from(content: str, tool_calls: list) -> tuple[str, str]:
    """Turn one assistant turn into the single action string tau2's gym accepts.

    Returns (action, kind). tau2 takes one action per step, so a turn carrying several tool
    calls can only have its first executed; `kind` records that so the share of turns where
    it happens is visible in the metrics rather than silently dropped.
    """
    if tool_calls:
        call = tool_calls[0]
        try:
            arguments = json.loads(call.arguments) if isinstance(call.arguments, str) else call.arguments
        except json.JSONDecodeError:
            arguments = {}
        action = json.dumps({"name": call.name, "arguments": arguments or {}})
        return action, ("tool" if len(tool_calls) == 1 else "tool_multi")
    return (content or "").strip(), "message"


@register("tau2")
class Tau2AgentLoop(AgentLoopBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        multi_turn = self.rollout_config.multi_turn
        self.max_assistant_turns = multi_turn.max_assistant_turns or 40
        self.tool_parser = ToolParser.get_tool_parser(multi_turn.format, self.tokenizer)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

        # The user simulator is part of the environment, not a knob: a local Qwen3-4B was
        # measured and solved 0.0% of episodes that terminated normally, against DeepSeek's
        # 38.7%. It is read from the environment so that a run cannot silently change it.
        self.domain = os.environ.get("TAU2_DOMAIN", "telecom")
        self.user_llm = os.environ.get("TAU2_USER_LLM", "deepseek/deepseek-chat")
        self.user_llm_args = {"temperature": float(os.environ.get("TAU2_USER_TEMP", "0.0"))}
        self.max_steps = int(os.environ.get("TAU2_MAX_STEPS", "100"))

        # w_delta = 0 is outcome-only training and is what arm 2 runs; raising it turns on the
        # per-turn process reward. The two arms differ by this number alone.
        self.w_delta = float(os.environ.get("TAU2_W_DELTA", "0.0"))

        self._tool_schemas = None
        self._task_set_patched = False

    def _use_full_task_set(self) -> None:
        """Point the domain's task loader at every task, not just the benchmark's 114.

        AgentGymEnv takes one `domain` string and uses it both to build the environment and
        to look a task up, but the task sets and the domains are named differently: the
        loader registered as "telecom" returns the 114 `base` tasks, and the 2,285-task set
        is registered as "telecom_full", which is not a domain and cannot be constructed.
        So the training pool's task ids are invisible to the gym until the loader is swapped.

        This makes the base tasks reachable too, which is exactly what must not be trained
        on. What keeps them out is the dataset: build_rl_data.py takes `full \ base` and
        asserts the intersection is empty. This function widens what *can* be looked up; it
        does not widen what is sampled.
        """
        from tau2.registry import registry

        if self._task_set_patched:
            return
        full_name = f"{self.domain}_full"
        if full_name in registry.get_task_sets():
            # register_tasks refuses to replace an existing name, so rebind the entry.
            registry._tasks[self.domain] = registry.get_tasks_loader(full_name)
            logger.info(
                "tau2: %s task loader now serves the %s set (%d tasks)",
                self.domain, full_name, len(registry.get_tasks_loader(self.domain)()),
            )
        self._task_set_patched = True

    def _schemas(self) -> list[dict]:
        """The agent-side tool schemas, read from the benchmark rather than hand-copied."""
        if self._tool_schemas is None:
            from tau2.registry import registry

            env = registry.get_env_constructor(self.domain)(solo_mode=False)
            self._tool_schemas = [t.openai_schema for t in env.get_tools()]
        return self._tool_schemas

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        from tau2.gym import AgentGymEnv

        extra_info = kwargs.get("extra_info") or {}
        task_id = extra_info.get("task_id")
        if task_id is None:
            raise ValueError("tau2 needs extra_info.task_id; the dataset builder must supply it")

        self._use_full_task_set()
        metrics: dict[str, Any] = {}
        request_id = uuid4().hex
        schemas = self._schemas()

        env = await asyncio.to_thread(
            AgentGymEnv,
            domain=self.domain, task_id=task_id, max_steps=self.max_steps, solo_mode=False,
            user_llm=self.user_llm, user_llm_args=self.user_llm_args,
        )
        try:
            observation, _info = await asyncio.to_thread(env.reset)

            # The dataset row carries the system prompt and domain policy; the first customer
            # message comes from the episode, not from the dataset, because it is generated.
            messages = list(kwargs["raw_prompt"])
            messages.append({"role": "user", "content": observation or ""})
            prompt_ids = await self.apply_chat_template(messages, tools=schemas)
            init_len = len(prompt_ids)

            response_mask: list[int] = []
            response_logprobs: list[float] = []
            turn_rewards: list[float] = []
            kinds: list[str] = []
            reward, terminated, turns = 0.0, False, 0

            while not terminated and turns < self.max_assistant_turns:
                output = await self.server_manager.generate(
                    request_id=request_id, prompt_ids=prompt_ids, sampling_params=sampling_params,
                )
                prompt_ids = prompt_ids + output.token_ids
                response_mask += [1] * len(output.token_ids)
                response_logprobs += output.log_probs or [0.0] * len(output.token_ids)
                turns += 1
                if len(response_mask) >= self.response_length:
                    break

                content, tool_calls = await self.tool_parser.extract_tool_calls(output.token_ids)
                action, kind = _action_from(content, tool_calls)
                kinds.append(kind)

                observation, step_reward, terminated, _truncated, _info = await asyncio.to_thread(
                    env.step, action
                )
                turn_rewards.append(float(step_reward))
                reward = float(step_reward)
                if terminated:
                    break

                # An environment reply is a tool message; the customer's reply is a user
                # message. Which one it was is decided by what the policy just did.
                role = "tool" if kind.startswith("tool") else "user"
                reply = [{"role": role, "content": observation or ""}]
                # No tools here. The schemas belong to the opening prompt; passing them again
                # re-renders all thirteen of them into every turn -- 1,850 tokens each time,
                # measured. Episodes then hit the response cap after about thirteen turns and
                # score zero for being truncated, the context the policy conditions on stops
                # matching anything it saw in SFT, and the KV cache fills with copies of the
                # same text. verl's own ToolAgentLoop renders incremental turns without tools.
                reply_ids = await self.apply_chat_template(reply, remove_system_prompt=True)
                reply_ids = self.turn_separator + reply_ids
                if len(response_mask) + len(reply_ids) >= self.response_length:
                    break
                prompt_ids = prompt_ids + reply_ids
                response_mask += [0] * len(reply_ids)
                response_logprobs += [0.0] * len(reply_ids)

            # One line per finished episode. The run had no per-episode marker in its logs,
            # so progress could only be inferred from turn counts, and the markers tried
            # first were printed more than once each.
            logger.warning(
                "TAU2_EPISODE_DONE task=%s turns=%d terminated=%d reward=%.3f",
                task_id, turns, int(terminated), reward,
            )
            metrics["tau2_turns"] = turns
            metrics["tau2_terminated"] = int(terminated)
            metrics["tau2_tool_turns"] = sum(1 for k in kinds if k.startswith("tool"))
            metrics["tau2_multi_call_turns"] = sum(1 for k in kinds if k == "tool_multi")

            response_ids = prompt_ids[init_len:]
            return AgentLoopOutput(
                prompt_ids=prompt_ids[:init_len],
                response_ids=response_ids[: self.response_length],
                response_mask=response_mask[: self.response_length],
                response_logprobs=response_logprobs[: self.response_length],
                reward_score=reward,
                num_turns=turns,
                metrics=metrics,
                extra_fields={"tau2_task_id": task_id, "tau2_turn_rewards": turn_rewards},
            )
        finally:
            await asyncio.to_thread(env.close)
