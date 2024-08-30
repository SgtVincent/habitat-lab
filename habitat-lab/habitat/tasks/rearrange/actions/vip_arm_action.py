# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Union
import magnum as mn
import numpy as np
from gym import spaces

import habitat_sim
from habitat.articulated_agent_controllers import HumanoidRearrangeController
from habitat.core.registry import registry
from habitat.tasks.rearrange.actions.actions import (
    ArticulatedAgentAction,
    ArmEEAction,
    ArmRelPosKinematicReducedActionStretch
)
from habitat.tasks.rearrange.utils import (
    coll_name_matches,
    coll_link_name_matches,
    get_match_link
)
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim
from habitat_sim.physics import MotionType



@registry.register_task_action
class VIPPickAction(ArmEEAction, ArticulatedAgentAction):
    """
    Pick/drop action for the articulated_agent given the object/receptacle index.
    Uses inverse kinematics (requires pybullet) to apply end-effector position control for the articulated_agent's arm.
    """

    def __init__(self, *args, sim: RearrangeSim, task, **kwargs):
        ArmEEAction.__init__(self, *args, sim=sim, **kwargs)
        self._sim = sim
        self._task = task
        self._entities = self._task.pddl_problem.get_ordered_entities_list()
        self._prev_ep_id = None
        # Agent not initialized
        # self._init_cord = self.cur_articulated_agent.ee_transform().translation
        self._hand_pose_iter = 0
        self.is_reset = False

    def reset(self, *args, **kwargs):
        super().reset()

    @property
    def action_space(self):
        return spaces.Box(
                    shape=(6,),
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    dtype=np.float32,
                )
    # NOTE: get_rigid_object_manager has different object idx than that of pddl planner
    # def _get_coord_for_idx(self, object_target_idx):
    #     obj_pos = (
    #         self._sim.get_rigid_object_manager()
    #         .get_object_by_id(object_target_idx)
    #         .translation
    #     )
    #     return obj_pos

    def _get_coord_for_pddl_idx(self, object_target_idx):
        pick_obj_entity = self._entities[int(object_target_idx)]
        obj_pos = self._task.pddl_problem.sim_info.get_entity_pos(
            pick_obj_entity
        )
        return obj_pos

    def get_scene_index_obj(self, object_target_idx):
        pick_obj_entity = self._entities[int(object_target_idx)]
        entity_name = pick_obj_entity.name
        obj_id = self._task.pddl_problem.sim_info.obj_ids[entity_name]
        return self._sim.scene_obj_ids[obj_id]

    def _suction_grasp(self):
        """
        Grasp object using suction grasp, snap to object if contact is detected.
        """
        attempt_snap_entity: Optional[Union[str, int]] = None
        match_coll = None
        contacts = self._sim.get_physics_contact_points()

        # TODO: the two arguments below should be part of args
        ee_index = 0
        index_grasp_manager = 0

        robot_id = self._sim.articulated_agent.sim_obj.object_id
        all_gripper_links = list(
            self._sim.articulated_agent.params.gripper_joints
        )
        robot_contacts = [
            c
            for c in contacts
            if coll_name_matches(c, robot_id)
            and any(coll_link_name_matches(c, l) for l in all_gripper_links)
        ]

        if len(robot_contacts) == 0:
            return

        # Contacted any objects?
        for scene_obj_id in self._sim.scene_obj_ids:
            for c in robot_contacts:
                if coll_name_matches(c, scene_obj_id):
                    match_coll = c
                    break
            if match_coll is not None:
                attempt_snap_entity = scene_obj_id
                break

        if attempt_snap_entity is not None:
            rom = self._sim.get_rigid_object_manager()
            ro = rom.get_object_by_id(attempt_snap_entity)

            ee_T = self.cur_articulated_agent.ee_transform()
            obj_in_ee_T = ee_T.inverted() @ ro.transformation

            # here we need the link T, not the EE T for the constraint frame
            ee_link_T = self.cur_articulated_agent.sim_obj.get_link_scene_node(
                self.cur_articulated_agent.params.ee_links[ee_index]
            ).absolute_transformation()

            self._sim.grasp_mgr.snap_to_obj(
                int(attempt_snap_entity),
                force=False,
                # rel_pos is the relative position of the object COM in link space
                rel_pos=ee_link_T.inverted().transform_point(ro.translation),
                keep_T=obj_in_ee_T,
                should_open_gripper=False,
            )
            return

    def set_desired_ee_pos(self, ee_pos: np.ndarray) -> None:
        self.ee_target += np.array(ee_pos)

        self.apply_ee_constraints()

        joint_pos = np.array(self.cur_articulated_agent.arm_joint_pos)
        joint_vel = np.zeros(joint_pos.shape)

        self._ik_helper.set_arm_state(joint_pos, joint_vel)

        des_joint_pos = self._ik_helper.calc_ik(self.ee_target)
        des_joint_pos = list(des_joint_pos)
        if self.cur_articulated_agent.sim_obj.motion_type == MotionType.DYNAMIC:
            self.cur_articulated_agent.arm_motor_pos = des_joint_pos
        if self.cur_articulated_agent.sim_obj.motion_type == MotionType.KINEMATIC:
            self.cur_articulated_agent.arm_joint_pos = des_joint_pos
            self.cur_articulated_agent.fix_joint_values = des_joint_pos

    def step(self, pick_action, **kwargs):
        object_pick_pddl_idx = pick_action[0]
        should_pick = pick_action[1]
        has_pick_action = pick_action[2]
        if has_pick_action > 0:
            object_coord = pick_action[3:]
        else:
            object_coord = self._get_coord_for_pddl_idx(object_pick_pddl_idx)

        if object_pick_pddl_idx > len(self._entities):
            return self.ee_target

        if should_pick == 1:
            # or self.cur_grasp_mgr.snap_idx is None
            cur_ee_pos = self.cur_articulated_agent.ee_transform().translation
            if not self.is_reset:
                self.ee_target = self._ik_helper.calc_fk(self.cur_articulated_agent.arm_joint_pos)
                # Note(zxz): ee_target is under transformation of ik_help,
                # it should be transformed to the world base to be equal to cur_ee_pos
                # cur_ee_pos = self.cur_articulated_agent.base_transformation.transform_point(self.ee_target)
                self.is_reset = True
            translation = object_coord - cur_ee_pos

            # translation from object to end effector in base frame
            translation_base = self.cur_articulated_agent.base_transformation.inverted().transform_vector(translation)
            should_rest = False
            # if self.hand_state == HandState.APPROACHING:  # Approaching
            # Only move the hand to object if has to drop or object is not grabbed
            translation_base = np.clip(translation_base, -1, 1)
            self._ee_ctrl_lim = 0.03
            translation_base *= self._ee_ctrl_lim
            self.set_desired_ee_pos(translation_base)

            # DEBUG VISUALIZATION
            if self._render_ee_target:
                global_pos = self.cur_articulated_agent.base_transformation.transform_point(
                    self.ee_target
                )
                self._sim.viz_ids["ee_target"] = self._sim.visualize_position(
                    global_pos, self._sim.viz_ids["ee_target"]
                )

            # TODO: should we add retracting action here?

            # Grasp & Ungrasp when we are close to the target
            # if np.linalg.norm(translation_base) < self._config.grasp_thresh_dist:
            #     self.cur_grasp_mgr.snap_to_obj(
            #         self.get_scene_index_obj(object_pick_pddl_idx),
            #     )
            #     return self.ee_target
        else:
            self.is_reset = False

        return self.ee_target


@registry.register_task_action
class VIPPlaceAction(VIPPickAction):
    """
    Pick/drop action for the articulated_agent given the object/receptacle index.
    Uses inverse kinematics (requires pybullet) to apply end-effector position control for the articulated_agent's arm.
    """

    def __init__(self, *args, sim: RearrangeSim, task, **kwargs):
        ArmEEAction.__init__(self, *args, sim=sim, **kwargs)
        self._sim = sim
        self._task = task
        self._entities = self._task.pddl_problem.get_ordered_entities_list()
        self._prev_ep_id = None
        self.is_reset = False

    def reset(self, *args, **kwargs):
        ArmEEAction.reset(self, *args, **kwargs)

    @property
    def action_space(self):
        return spaces.Box(
            shape=(6,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def step(self, place_action, **kwargs):

        recep_place_pddl_idx = place_action[0]
        should_place = place_action[1]
        has_place_action = place_action[2]
        if has_place_action > 0:
            recep_coord = place_action[3:]
        else:
            recep_coord = self._get_coord_for_pddl_idx(recep_place_pddl_idx)

        if recep_place_pddl_idx-1 > len(self._entities):
            return self.ee_target

        if should_place == 2:
            # get recep coordinates
            cur_ee_pos = self.cur_articulated_agent.ee_transform().translation
            if not self.is_reset:
                self.ee_target = self._ik_helper.calc_fk(self.cur_articulated_agent.arm_joint_pos)
                self.is_reset = True
            translation = recep_coord - cur_ee_pos

            # translation from object to end effector in base frame
            translation_base = self.cur_articulated_agent.base_transformation.inverted().transform_vector(translation)

            should_rest = False
            # if self.hand_state == HandState.APPROACHING:  # Approaching
            # Only move the hand to object if has to drop or object is not grabbed
            # or self.cur_grasp_mgr.snap_idx is None
            translation_base = np.clip(translation_base, -1, 1)
            self._ee_ctrl_lim = 0.03
            translation_base *= self._ee_ctrl_lim
            self.set_desired_ee_pos(translation_base)

            # DEBUG VISUALIZATION
            if self._render_ee_target:
                global_pos = self.cur_articulated_agent.base_transformation.transform_point(
                    self.ee_target
                )
                self._sim.viz_ids["ee_target"] = self._sim.visualize_position(
                    global_pos, self._sim.viz_ids["ee_target"]
                )

            # Grasp & Ungrasp when we are close to the target
            # if np.linalg.norm(translation_base) < self._config.grasp_thresh_dist:
            #     self.cur_grasp_mgr.desnap()
        else:
            self.is_reset = False

        return self.ee_target


@registry.register_task_action
class StretchVIPPickAction(ArmRelPosKinematicReducedActionStretch, VIPPickAction):
    """
    Pick/drop action for the articulated_agent given the object/receptacle index.
    Uses inverse kinematics (requires pybullet) to apply end-effector position control for the articulated_agent's arm.
    """

    def __init__(self, *args, config, sim: RearrangeSim, task, **kwargs):
        super().__init__(self, *args, config=config, sim=sim, task=task, **kwargs)
        self._sim = sim
        self._task = task
        self._entities = self._task.pddl_problem.get_ordered_entities_list()
        self._prev_ep_id = None
        # Agent not initialized
        # self._init_cord = self.cur_articulated_agent.ee_transform().translation
        self._init_cord = np.array([0.0, 0.0, 0.0])
        self._hand_pose_iter = 0

        self._delta_pos_limit = self._config.delta_pos_limit
        self._should_clip = self._config.get("should_clip", True)
        self._arm_joint_mask = self._config.arm_joint_mask
        self._arm_joint_limit = self._config.arm_joint_limit

    def set_desired_ee_pos(self, ee_pos: np.ndarray) -> np.array:
        self.ee_target += np.array(ee_pos)

        self.apply_ee_constraints()

        joint_pos = np.array(self.cur_articulated_agent.arm_joint_pos)
        joint_vel = np.zeros(joint_pos.shape)

        self._ik_helper.set_arm_state(joint_pos, joint_vel)

        des_joint_pos = self._ik_helper.calc_ik(self.ee_target)

        return np.array(des_joint_pos)

    def step(self, pick_action, **kwargs):
        object_pick_pddl_idx = pick_action[0]
        should_pick = pick_action[1]
        has_pick_action = pick_action[2]
        if has_pick_action > 0:
            object_coord = pick_action[3:]
        else:
            object_coord = self._get_coord_for_pddl_idx(object_pick_pddl_idx)

        if object_pick_pddl_idx > len(self._entities):
            return self.ee_target

        if should_pick == 1:
            # or self.cur_grasp_mgr.snap_idx is None
            cur_ee_pos = self.cur_articulated_agent.ee_transform().translation
            if not self.is_reset:
                self.ee_target = self._ik_helper.calc_fk(self.cur_articulated_agent.arm_joint_pos)
                self.is_reset = True
            translation = object_coord - cur_ee_pos

            # translation from object to end effector in base frame
            translation_base = self.cur_articulated_agent.base_transformation.inverted().transform_vector(translation)
            should_rest = False
            # if self.hand_state == HandState.APPROACHING:  # Approaching
            # Only move the hand to object if has to drop or object is not grabbed
            translation_base = np.clip(translation_base, -1, 1)
            self._ee_ctrl_lim = 0.03
            translation_base *= self._ee_ctrl_lim
            delta_pos = self.set_desired_ee_pos(translation_base)
            if self._should_clip:
                # clip from -1 to 1
                delta_pos = np.clip(delta_pos, -1, 1)
            delta_pos *= self._delta_pos_limit
            self._sim: RearrangeSim

            # Expand delta_pos based on mask
            expanded_delta_pos = np.zeros(len(self._arm_joint_mask))
            src_idx = 0
            tgt_idx = 0
            for mask in self._arm_joint_mask:
                if mask == 0:
                    tgt_idx += 1
                    src_idx += 1
                    continue
                expanded_delta_pos[tgt_idx] = delta_pos[src_idx]
                tgt_idx += 1
                src_idx += 1

            min_limit, max_limit = self.cur_articulated_agent.arm_joint_limits

            if self.cur_articulated_agent.sim_obj.motion_type == MotionType.DYNAMIC:
                set_arm_pos = (
                    expanded_delta_pos + self.cur_articulated_agent.arm_motor_pos
                )
                # Perform roll over to the joints so that the user cannot control
                # the motor 2, 3, 4 for the arm.
                if expanded_delta_pos[0] >= 0:
                    for i in range(3):
                        if set_arm_pos[i] > max_limit[i]:
                            set_arm_pos[i + 1] += set_arm_pos[i] - max_limit[i]
                            set_arm_pos[i] = max_limit[i]
                else:
                    for i in range(3):
                        if set_arm_pos[i] < min_limit[i]:
                            set_arm_pos[i + 1] -= min_limit[i] - set_arm_pos[i]
                            set_arm_pos[i] = min_limit[i]
                set_arm_pos = np.clip(set_arm_pos, min_limit, max_limit)

                self.cur_articulated_agent.arm_joint_pos = set_arm_pos
            if self.cur_articulated_agent.sim_obj.motion_type == MotionType.KINEMATIC:
                set_arm_pos = (
                    expanded_delta_pos + self.cur_articulated_agent.arm_joint_pos
                )
                # Perform roll over to the joints so that the user cannot control
                # the motor 2, 3, 4 for the arm.
                if expanded_delta_pos[0] >= 0:
                    for i in range(3):
                        if set_arm_pos[i] > max_limit[i]:
                            set_arm_pos[i + 1] += set_arm_pos[i] - max_limit[i]
                            set_arm_pos[i] = max_limit[i]
                else:
                    for i in range(3):
                        if set_arm_pos[i] < min_limit[i]:
                            set_arm_pos[i + 1] -= min_limit[i] - set_arm_pos[i]
                            set_arm_pos[i] = min_limit[i]
                set_arm_pos = np.clip(set_arm_pos, min_limit, max_limit)

                self.cur_articulated_agent.arm_joint_pos = set_arm_pos
                self.cur_articulated_agent.fix_joint_values = set_arm_pos


            # DEBUG VISUALIZATION
            if self._render_ee_target:
                global_pos = self.cur_articulated_agent.base_transformation.transform_point(
                    self.ee_target
                )
                self._sim.viz_ids["ee_target"] = self._sim.visualize_position(
                    global_pos, self._sim.viz_ids["ee_target"]
                )
        else:
            self.is_reset = False

        return self.ee_target
