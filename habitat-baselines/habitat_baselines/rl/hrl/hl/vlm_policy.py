# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple
import torch
import numpy as np

from habitat.tasks.rearrange.multi_task.rearrange_pddl import parse_func
from habitat_baselines.common.logging import baselines_logger
from habitat_baselines.rl.hrl.hl.high_level_policy import HighLevelPolicy
from habitat_baselines.rl.ppo.policy import PolicyActionData
from habitat.tasks.rearrange.rearrange_sensors import (
    LocalizationSensor,
    EEPositionSensor,
)

from habitat_mas.prompt.run_vip import run_vip
from habitat_mas.prompt.prompter import run_prompter
import json


class VLMHighLevelPolicy(HighLevelPolicy):
    """
    Executes a fixed sequence of high-level actions as specified by the
    `solution` field of the PDDL problem file.
    :property _solution_actions: List of tuples where the first tuple element
        is the action name and the second is the action arguments. Stores a plan
        for each environment.
    """

    _solution_actions: List[List[Tuple[str, List[str]]]]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._update_solution_actions(
            [self._parse_solution_actions() for _ in range(self._num_envs)]
        )

        self._next_sol_idxs = torch.zeros(self._num_envs, dtype=torch.int32)

        self.obs = None
        self.samples = None
        self.skill_name = None
        self.skill_args = None
        self.should_prompt = False
        self.base_pos = None
        self.base_heading = None
        self.verbose = True

    def _update_solution_actions(
        self, solution_actions: List[List[Tuple[str, List[str]]]]
    ) -> None:
        if len(solution_actions) == 0:
            raise ValueError(
                "Solution actions must be non-empty (if want to execute no actions, just include a no-op)"
            )
        self._solution_actions = solution_actions

    def _parse_solution_actions(self) -> List[Tuple[str, List[str]]]:
        """
        Returns the sequence of actions to execute as a list of:
        - The action name.
        - A list of the action arguments.
        """
        solution = self._pddl_prob.solution

        solution_actions = []
        for i, hl_action in enumerate(solution):
            sol_action = (
                hl_action.name,
                [x.name for x in hl_action.param_values],
            )
            # Avoid adding plan actions that are assigned to other agents to the list.
            agent_idx = self._agent_name.split("_")[-1]
            for j, param in enumerate(hl_action.params):
                param_name = param.name
                param_value = hl_action.param_values[j].name
                # if the action is assigned to current agent, add it to the list
                if param_name == "robot" and param_value.split("_")[-1] == agent_idx:
                    solution_actions.append(sol_action)

            if self._config.add_arm_rest and i < (len(solution) - 1):
                solution_actions.append(parse_func("reset_arm(0)"))

        # Add a wait action at the end.
        # solution_actions.append(parse_func("wait(3000)"))

        return solution_actions

    def apply_mask(self, mask):
        """
        Apply the given mask to the next skill index.

        Args:
            mask: Binary mask of shape (num_envs, ) to be applied to the next
                skill index.
        """
        self._next_sol_idxs *= mask.cpu().view(-1)

    def _get_next_sol_idx(self, batch_idx, immediate_end):
        """
        Get the next index to be used from the list of solution actions.

        Args:
            batch_idx: The index of the current environment.

        Returns:
            The next index to be used from the list of solution actions.
        """
        if self._next_sol_idxs[batch_idx] >= len(
            self._solution_actions[batch_idx]
        ):
            baselines_logger.info(
                f"Calling for immediate end with {self._next_sol_idxs[batch_idx]}"
            )
            immediate_end[batch_idx] = True
            # Just repeat the last action.
            return len(self._solution_actions[batch_idx]) - 1
        else:
            return self._next_sol_idxs[batch_idx].item()

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        # We assign a value of 0. This is needed so that we can concatenate values in multiagent
        # policies
        return torch.zeros(rnn_hidden_states.shape[0], 1).to(
            rnn_hidden_states.device
        )

    def get_new_sample(
        self,
        observations,
        **kwargs
    ):
        """get query"""
        vip_info = json.loads(kwargs['vip_sim_info'][0]['vip_info'])
        camera_info = json.loads(kwargs['vip_sim_info'][0]['camera_info'])
        query = self._pddl_prompting(self.skill_name, self.skill_args.copy(), vip_info)

        """check if should get new sample"""
        ee_pos = observations[EEPositionSensor.cls_uuid].cpu().numpy()
        base_pos = observations[LocalizationSensor.cls_uuid][0][:3].cpu().numpy()
        base_heading = observations[LocalizationSensor.cls_uuid][0][3].cpu().numpy()

        if self.samples is None or len(self.samples) == 0:
            should_update_new_sample = True
        else:
            action = self.samples[0].action
            coord = self.samples[0].coord.xy
            if self.verbose:
                print(f"skill: {self.skill_name}, action: {action}, coord: {coord}")
                print(f"base_pos: {base_pos}, base_heading: {base_heading}")
            if self.skill_name == "pick" or self.skill_name == "place":
                should_update_new_sample = np.linalg.norm(ee_pos - action) < 0.05
            elif self.skill_name == "nav_to_obj" or self.skill_name == "nav_to_goal":
                should_update_new_sample = (np.linalg.norm(
                        np.array([base_pos[0], base_pos[2]]) -
                        np.array([action[0], action[2]])) < 0.2)
                # agent get stuck
                if self.base_pos is not None and self.base_heading is not None:
                    if (np.linalg.norm(self.base_pos - base_pos) < 0.01
                            and np.abs(self.base_heading - base_heading) < 0.01):
                        should_update_new_sample = True
                        direction = "right" if coord[0] >= 128 else "left"
                        query = query + (f" Note: The agent is now stuck, you should choose the most suitable points "
                                         f"that can make the agent turn left or turn right or turn back. ")
                                         # f"Your last action is to move to the direction that is "
                                         # f"{abs(coord[0]-128)/1.28}% {direction} relative to the middle of your observation image")
            else:
                should_update_new_sample = False
        self.base_pos = base_pos
        self.base_heading = base_heading

        """get new sample"""
        if should_update_new_sample:
            with open('api.json', 'r') as file:
                api = json.load(file)
            if self.verbose:
                print(f"query: {query}")

            if self.skill_name == "pick" or self.skill_name == "place":
                rgb = observations[0]['arm_workspace_rgb'].cpu().numpy().copy()
                points = observations[0]['arm_workspace_points'].cpu().numpy().copy()
                self.should_prompt = True
                if len(points[0]) > 0 and len(points[1]) > 0:
                    vlm_obs = run_vip(
                        im=rgb,
                        query=query,
                        n_samples_init=8,
                        n_samples_opt=6,
                        n_iters=2,
                        n_parallel_trials=1,
                        openai_api_key=api['key'],
                        points=points,
                        camera_info=camera_info,
                        skill_name=self.skill_name
                    )
                    self.obs = vlm_obs[0][-1]
                    self.samples = vlm_obs[1]
                else:
                    self.obs = rgb
                    self.samples = None
            if self.skill_name == "nav_to_obj" or self.skill_name == "nav_to_goal":
                self.should_prompt = True
                rgb = observations[0]['nav_workspace_rgb'].cpu().numpy().copy()
                points = observations[0]['nav_workspace_points'].cpu().numpy().copy()
                if len(points[0]) > 0 and len(points[1]) > 0:
                    vlm_obs = run_vip(
                        im=rgb,
                        query=query,
                        n_samples_init=8,
                        n_samples_opt=6,
                        n_iters=1,
                        n_parallel_trials=1,
                        openai_api_key=api['key'],
                        points=points,
                        camera_info=camera_info,
                        skill_name=self.skill_name
                    )
                    self.obs = vlm_obs[0][-1]
                    self.samples = vlm_obs[1]
                else:
                    self.obs = rgb
                    self.samples = None

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
    ):
        batch_size = masks.shape[0]
        next_skill = torch.zeros(batch_size)
        skill_args_data = [None for _ in range(batch_size)]
        immediate_end = torch.zeros(batch_size, dtype=torch.bool)
        for batch_idx, should_plan in enumerate(plan_masks):
            if should_plan == 1.0:
                use_idx = self._get_next_sol_idx(batch_idx, immediate_end)
                skill_name, skill_args = self._solution_actions[batch_idx][use_idx]
                self.skill_name = skill_name
                self.skill_args = skill_args

                vip_info = json.loads(kwargs['vip_sim_info'][0]['vip_info'])
                camera_info = json.loads(kwargs['vip_sim_info'][0]['camera_info'])
                query = self._pddl_prompting(skill_name, skill_args.copy(), vip_info)

                with open('api.json', 'r') as file:
                    api = json.load(file)

                if self.verbose:
                    print(f"query: {query}")

                if skill_name == "pick" or skill_name == "place":
                    rgb = observations[0]['arm_workspace_rgb'].cpu().numpy().copy()
                    points = observations[0]['arm_workspace_points'].cpu().numpy().copy()
                    self.should_prompt = True
                    if len(points[0]) > 0 and len(points[1]) > 0:
                        vlm_obs = run_vip(
                            im=rgb,
                            query=query,
                            n_samples_init=8,
                            n_samples_opt=6,
                            n_iters=2,
                            n_parallel_trials=1,
                            openai_api_key=api['key'],
                            points=points,
                            camera_info=camera_info,
                            skill_name=skill_name
                        )
                        self.obs = vlm_obs[0][-1]
                        self.samples = vlm_obs[1]
                    else:
                        self.obs = rgb
                        self.samples = None
                if skill_name == "nav_to_obj" or skill_name == "nav_to_goal":
                    self.should_prompt = True
                    rgb = observations[0]['nav_workspace_rgb'].cpu().numpy().copy()
                    points = observations[0]['nav_workspace_points'].cpu().numpy().copy()
                    if len(points[0]) > 0 and len(points[1]) > 0:
                        vlm_obs = run_vip(
                            im=rgb,
                            query=query,
                            n_samples_init=8,
                            n_samples_opt=6,
                            n_iters=1,
                            n_parallel_trials=1,
                            openai_api_key=api['key'],
                            points=points,
                            camera_info=camera_info,
                            skill_name=skill_name
                        )
                        self.obs = vlm_obs[0][-1]
                        self.samples = vlm_obs[1]
                    else:
                        self.obs = rgb
                        self.samples = None

                # 这里适用于，看见物体，需要pick，这时候需要将arm移动到特定位置（生成action）然后判定pick
                # 需要nav，可以通过prompting获得目标点，然后nav到目标点（生成action），然后判定nav_to_goal
                # TODO(zxz): 需要新写nav到特定点和移动到特定点的action和skill

                baselines_logger.info(
                    f"Got next element of the plan with {skill_name}, {skill_args}"
                )
                if skill_name not in self._skill_name_to_idx:
                    raise ValueError(
                        f"Could not find skill named {skill_name} in {self._skill_name_to_idx}"
                    )
                next_skill[batch_idx] = self._skill_name_to_idx[skill_name]

                skill_args_data[batch_idx] = skill_args  # type: ignore[call-overload]

                self._next_sol_idxs[batch_idx] += 1

        return next_skill, skill_args_data, immediate_end, PolicyActionData()

    def _pddl_prompting(self, skill_name, skill_args, vip_info):
        for key, value in vip_info.items():
            skill_args = [value if entity == key else entity for entity in skill_args]
        query = f"{skill_name}({[arg for arg in skill_args]})"
        if skill_name == "nav_to_goal":
            obj_name = skill_args[0]
            if f"recep_any_targets|{obj_name[-1]}" in vip_info.keys():
                skill_args.append(vip_info[f"recep_any_targets|{obj_name[-1]}"])
                query = f"{skill_name}: {skill_args[0]}, which is placed on {skill_args[2]}, you need to explore, find the most suitable next-step nav points."
            else:
                query = f"{skill_name}: {skill_args[0]}, you need to explore, find the most suitable next-step nav points."
        elif skill_name == "nav_to_obj":
            query = f"{skill_name}: {skill_args[0]}, you need to explore, find the most suitable next-step nav points."
        elif skill_name == "pick":
            query = f"{skill_name}: {skill_args[0]}, find the most suitable grasp points."
        elif skill_name == "place":
            query = f"{skill_name}: {skill_args[0]}, which should be placed on {skill_args[1]}, find the most suitable place points."
        query = query + " The objects you need to find are all masked and annotated with its name."
        return query


    def filter_envs(self, curr_envs_to_keep_active):
        """
        Cleans up stateful variables of the policy so that
        they match with the active environments
        """
        self._next_sol_idxs = self._next_sol_idxs[curr_envs_to_keep_active]
        parse_solution_actions = [
            self._parse_solution_actions() for _ in range(self._num_envs)
        ]
        self._update_solution_actions(
            [
                parse_solution_actions[i]
                for i in range(curr_envs_to_keep_active.shape[0])
                if curr_envs_to_keep_active[i]
            ]
        )

    def visual_prompting(self, observations, **kwargs):
        camera_info = json.loads(kwargs['vip_sim_info'][0]['camera_info'])
        if self.should_prompt:
            # should use vlm to iteratively sample
            vlm_obs = self.obs
            self.should_prompt = False
        else:
            # keep the last sampled point
            if self.skill_name == "pick" or self.skill_name == "place":
                rgb = observations[0]['arm_workspace_rgb'].copy()
                points = observations[0]['arm_workspace_points'].copy()
            elif self.skill_name == "nav_to_obj" or self.skill_name == "nav_to_goal":
                rgb = observations[0]['nav_workspace_rgb'].copy()
                points = observations[0]['nav_workspace_points'].copy()
            else:
                rgb = observations[0]['arm_workspace_rgb'].copy()
                points = observations[0]['arm_workspace_points'].copy()

            if len(points[0]) > 0 and len(points[1]) > 0 and self.samples is not None:
                vlm_obs = run_prompter(im=rgb, points=points, samples=self.samples,
                                       camera_info=camera_info, skill_name=self.skill_name)
            else:
                vlm_obs = rgb

        return vlm_obs
