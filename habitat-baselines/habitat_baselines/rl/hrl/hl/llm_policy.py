# ruff: noqa
from typing import Any, Dict, List, Tuple

import torch
from habitat.tasks.rearrange.multi_task.pddl_action import PddlAction
from habitat_mas.agents.actions.arm_actions import *
from habitat_mas.agents.actions.base_actions import *
from habitat_mas.agents.crab_agent import CrabAgent

# TODO: replace dummy_agent with llm_agent
from habitat_mas.agents.dummy_agent import DummyAgent

from habitat_baselines.rl.hrl.hl.high_level_policy import HighLevelPolicy
from habitat_baselines.rl.ppo.policy import PolicyActionData

ACTION_POOL = [get_agents, send_request, nav_to_obj, nav_to_goal, pick, place]


class LLMHighLevelPolicy(HighLevelPolicy):
    """
    High-level policy that uses an LLM agent to select skills.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._all_actions = self._setup_actions()
        self._n_actions = len(self._all_actions)
        self._active_envs = torch.zeros(self._num_envs, dtype=torch.bool)
        
        environment_action_name_set = set(
            [action._name for action in self._all_actions]
        )

        llm_actions = [
            action
            for action in ACTION_POOL
            if action.name in environment_action_name_set
        ]
        # Initialize the LLM agent
        self._llm_agent = self._init_llm_agent(kwargs["agent_name"], llm_actions)

    def _init_llm_agent(self, agent_name, action_list):
        # Initialize the LLM agent here based on the config
        # This could load a pre-trained model, set up prompts, etc.
        # Return the initialized agent
        action_list.append(send_request)
        return DummyAgent(agent_name=agent_name, action_list=action_list)
        # return CrabAgent(
        #     agent_name,
        #     'Send a request " tell me your name" to another agent. If you are "agent_0", send to "agent_1". If you are "agent_1", send to "agent_0". ',
        #     action_list,
        # )

    def _parse_function_call_args(self, llm_args: Dict) -> str:
        """
        Parse the arguments of a function call from the LLM agent to the policy input argument format.
        """
        return llm_args

    def apply_mask(self, mask):
        """
        Apply the given mask to the agent in parallel envs.

        Args:
            mask: Binary mask of shape (num_envs, ) to be applied to the agents.
        """
        self._active_envs = mask

    def get_next_skill(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        plan_masks,
        deterministic,
        log_info,
        **kwargs,
    ) -> Tuple[torch.Tensor, List[Any], torch.BoolTensor, PolicyActionData]:
        """
        Get the next skill to execute from the LLM agent.
        """
        # TODO: use these text context to query the LLM agent with function call
        envs_text_context = kwargs.get("envs_text_context", None)
        agent_task_assignments = kwargs.get("agent_task_assignments", None)
        
        batch_size = masks.shape[0]
        next_skill = torch.zeros(batch_size)
        skill_args_data = [None for _ in range(batch_size)]
        immediate_end = torch.zeros(batch_size, dtype=torch.bool)

        for batch_idx, should_plan in enumerate(plan_masks):
            if should_plan != 1.0:
                continue

            # Query the LLM agent with the current observations
            # to get the next action and arguments
            llm_output = self._llm_agent.chat(observations[batch_idx])
            if llm_output is None:
                next_skill[batch_idx] = self._skill_name_to_idx["wait"]
                skill_args_data[batch_idx] = ["1"]
                continue

            action_name = llm_output["name"]
            action_args = self._parse_function_call_args(llm_output["arguments"])

            if action_name in self._skill_name_to_idx:
                next_skill[batch_idx] = self._skill_name_to_idx[action_name]
                skill_args_data[batch_idx] = action_args
            else:
                # If the action is not valid, do nothing
                next_skill[batch_idx] = self._skill_name_to_idx["wait"]
                skill_args_data[batch_idx] = ["1"]

        return (
            next_skill,
            skill_args_data,
            immediate_end,
            PolicyActionData(),
        )
