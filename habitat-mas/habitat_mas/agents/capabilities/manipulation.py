from typing import Tuple
import os
import numpy as np
from habitat.config.default_structured_configs import ThirdRGBSensorConfig, HeadRGBSensorConfig
from habitat.config.default_structured_configs import AgentConfig
from habitat.config.default_structured_configs import (   
    ArmActionConfig,
    BaseVelocityNonCylinderActionConfig,
    OracleNavActionConfig
)
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim
from habitat.articulated_agents.robots.fetch_robot import FetchRobot
from habitat.articulated_agents.robots.spot_robot import SpotRobot
from habitat.articulated_agents.robots.stretch_robot import StretchRobot
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim
from habitat.tasks.rearrange.utils import IkHelper

from habitat_mas.agents.sim_utils import make_cfg, default_sim_settings
from habitat_mas.agents.robots.defaults import (
    data_dir,
    robot_urdf_paths,
    robot_arm_urdf_paths
)
from habitat_mas.test.data_utils import init_rearrange_env

def sample_ee_positions(
    sim: RearrangeSim,
    agent_id: int,
    num_bins=5,
    subtract_base=True,
) -> np.ndarray:
    """
    Sample end effector positions of the robot arm.
    """
    end_effector_positions = []
    ik_helper: IkHelper = sim.agents_mgr[agent_id].ik_helper
    agent = sim.agents_mgr[agent_id].articulated_agent
    joint_limits_lower, joint_limits_uppper = ik_helper.get_joint_limits()

    # Generate all joint position combinations, which is an N-dimensional grid, N is the number of joints
    joint_positions = np.meshgrid(
        *[
            np.linspace(
                joint_limits_lower[i], joint_limits_uppper[i], num_bins
            )
            for i in range(len(joint_limits_lower))
        ]
    )

    # Flatten the joint positions
    joint_positions = np.array(
        [joint_position.flatten() for joint_position in joint_positions]
    ).T

    # Iterate over the joint positions
    for joint_position in joint_positions:

        # Calculate the end effector position
        # end_effector_position = ik_helper.calc_fk(joint_position) # offset from real position
        agent.arm_joint_pos = joint_position
        agent.fix_joint_values = joint_position
        end_effector_position = agent.ee_transform().translation

        # Append the end effector position to the list
        end_effector_positions.append(end_effector_position)

    # Convert the end effector positions to a numpy array
    end_effector_positions = np.array(end_effector_positions)
    
    # get robot feet position 
    if subtract_base:
        base_pos = sim.agents_mgr[agent_id].articulated_agent.base_transformation.translation + \
            sim.agents_mgr[agent_id].articulated_agent.params.base_offset
        end_effector_positions -= base_pos
        
    return end_effector_positions

def get_arm_workspace(
    sim: RearrangeSim,
    agent_id: int,
    num_bins=5,
    geometry: str = "box",
    visualize=False,
    subtract_base=True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the workspace of the arm of the agent.

    Args:
        sim: The rearrange simulator.
        agent_id: The agent id.
        joint_delta: The delta value for joint position sampling.
        geometry: The geometry of the workspace.
    """
    # TODO: Do we need ellipsoid culling for better accuracy?
    assert geometry in ["sphere", "box"]

    # Initialize an empty list to store the end effector positions
    end_effector_positions = sample_ee_positions(
        sim, agent_id, num_bins=num_bins, subtract_base=subtract_base
    )

    # Visualize the end effector positions
    if visualize:
        for i, end_effector_position in enumerate(end_effector_positions):
            marker_name = f"end_effector_{i}"
            sim.viz_ids[marker_name] = sim.visualize_position(
                end_effector_position, r=0.01
            )

    if geometry == "sphere":
        # Calculate the center of the sphere
        center = np.mean(end_effector_positions, axis=0)

        # Calculate the radius of the sphere
        radius = np.max(
            np.linalg.norm(end_effector_positions - center, axis=1)
        )

        return center, radius

    elif geometry == "box":

        # Calculate the minimum and maximum x, y, and z coordinates
        min_coords = np.min(end_effector_positions, axis=0)
        max_coords = np.max(end_effector_positions, axis=0)

        return min_coords, max_coords

    else: # pragma: no cover
        raise ValueError("Invalid geometry")


def get_all_robot_arm_workspace():
    # Define the settings for the simulator
    produce_debug_video = False
    observations = []
    cfg_settings = default_sim_settings.copy()
    cfg_settings["scene"] = "NONE"
    cfg_settings["enable_physics"] = True

    # Create the simulator configuration
    hab_cfg = make_cfg(cfg_settings)
    # Define the robots to initialize

    
    robot_arm_workspace = {}
    
    agent_configs = {
        "FetchRobot": {
            "arm_len": 7,
            "pb_link_idx": 7
        },
        "SpotRobot": {
            "arm_len": 7,
            "pb_link_idx": 7
        },
        "StretchRobot": {
            "arm_len": 8,
            "pb_link_idx": 14
        }
    }
    
    for robot_type, robot_arm_urdf_path in robot_arm_urdf_paths.items():
        # Get the robot class            
            
        # Define the agent configuration
        agent_config = AgentConfig()
        agent_config.articulated_agent_urdf = robot_urdf_paths[robot_type]
        agent_config.articulated_agent_type = robot_type
        agent_config.ik_arm_urdf = robot_arm_urdf_path
        agent_config.arm_len = agent_configs[robot_type]["arm_len"]
        agent_config.pb_link_idx = agent_configs[robot_type]["pb_link_idx"]


        # Define sensors that will be attached to this agent, here a third_rgb sensor and a head_rgb.
        # We will later talk about why we are giving the sensors these names
        agent_config.sim_sensors = {
            "third_rgb": ThirdRGBSensorConfig(),
            "head_rgb": HeadRGBSensorConfig(),
        }
        agent_dict = {"main_agent": agent_config}
        action_dict = {
            "arm_ee_action": ArmActionConfig(
                arm_controller="ArmEEAction",
                # grip_controller="MagicGraspAction",
                grip_controller="SuctionGraspAction",
                # render_ee_target=True
            ),
            "base_velocity_non_cylinder_action": BaseVelocityNonCylinderActionConfig(),
            "oracle_coord_action": OracleNavActionConfig(type="OracleNavCoordinateAction", 
                spawn_max_dist_to_obj=1.0,
                motion_control="base_velocity_non_cylinder",
                navmesh_offset=[[0.0, 0.0], [0.25, 0.0], [-0.25, 0.0]]
            )
        }
        
        env = init_rearrange_env(agent_dict, action_dict)
        # Initialize pybullet ikhelper
        env.reset()
        
        # compute the both the  sphere and box arm workspace of the robot
        end_effector_positions = sample_ee_positions(
           env.sim, 0, num_bins=5, subtract_base=True
        )
        # Calculate the center of the sphere
        center = np.mean(end_effector_positions, axis=0)

        # Calculate the radius of the sphere
        radius = np.max(
            np.linalg.norm(end_effector_positions - center, axis=1)
        )
        # Calculate the minimum and maximum x, y, and z coordinates
        min_bound = np.min(end_effector_positions, axis=0)
        max_bound = np.max(end_effector_positions, axis=0)
        
        robot_arm_workspace[robot_type] = {
            "center": center,
            "radius": radius,
            "min_bound": min_bound,
            "max_bound": max_bound,
        }
            
    return robot_arm_workspace

if __name__ == "__main__":
    robot_arm_workspace = get_all_robot_arm_workspace()
    print(robot_arm_workspace)
