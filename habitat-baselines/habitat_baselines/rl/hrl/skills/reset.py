# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os.path as osp
from typing import List, Tuple
from dataclasses import dataclass

from gym import spaces
import numpy as np
import torch

from habitat.core.spaces import ActionSpace

from habitat_baselines.common.logging import baselines_logger
from habitat_baselines.rl.hrl.skills.nn_skill import NnSkillPolicy
from habitat_baselines.rl.hrl.utils import find_action_range
from habitat_baselines.rl.ppo.policy import PolicyActionData


class ResetArmSkill(NnSkillPolicy):
    """
    Skill to reset the arm to its initial position.
    """
    RESET_ID = 3

    @dataclass
    class ResetArmActionArgs:
        """
        :property action_idx: The index of the oracle action we want to execute
        :property grab_release: Whether we want to grab (1) or drop an object (0)
        """
        action_idx: int

    def __init__(
        self,
        wrap_policy,
        config,
        action_space,
        filtered_obs_space,
        filtered_action_space,
        batch_size,
    ):
        super().__init__(
            wrap_policy,
            config,
            action_space,
            filtered_obs_space,
            filtered_action_space,
            batch_size,
        )
        self.is_init = False
        action_name = "arm_reset_action"
        self._reset_srt_idx, self._reset_end_idx = find_action_range(action_space, action_name)

    def on_enter(
        self,
        skill_arg: List[str],
        batch_idxs: List[int],
        observations,
        rnn_hidden_states,
        prev_actions,
        skill_name,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ret = super().on_enter(
            skill_arg,
            batch_idxs,
            observations,
            rnn_hidden_states,
            prev_actions,
            skill_name,
        )

        return ret

    @classmethod
    def from_config(
        cls, config, observation_space, action_space, batch_size, full_config
    ):
        filtered_action_space = ActionSpace(
            {config.action_name: action_space[config.action_name]}
        )
        baselines_logger.debug(
            f"Loaded action space {filtered_action_space} for skill {config.skill_name}"
        )
        return cls(
            None,
            config,
            action_space,
            observation_space,
            filtered_action_space,
            batch_size,
        )

    def _parse_skill_arg(self, skill_name: str, skill_arg: str):
        return None

    @property
    def required_obs_keys(self) -> List[str]:
        return super().required_obs_keys + ["joint"]

    def _is_skill_done(
        self, observations, rnn_hidden_states, prev_actions, masks, batch_idx
    ):
        current_joint_pos = observations["joint"].cpu().numpy()

        return (
            torch.as_tensor(
                np.abs(current_joint_pos - self._init_joint_pos).max(-1),
                dtype=torch.float32,
            )
            < 0.01
        )

    def _internal_act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        cur_batch_idx,
        deterministic=False,
        new_action=None,
    ):
        current_joint_pos = observations["joint"].cpu().numpy()
        if not self.is_init:
            self._init_joint_pos = current_joint_pos
            self.is_init = True

        full_action = torch.zeros(
            (masks.shape[0], self._full_ac_size), device=masks.device
        )

        full_action[0][self._reset_srt_idx:self._reset_end_idx] = torch.tensor([self.RESET_ID])

        return PolicyActionData(
            actions=full_action, rnn_hidden_states=rnn_hidden_states
        )
