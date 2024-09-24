from __future__ import annotations
import os
import datetime
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
from hydra.core.hydra_config import HydraConfig

import gym.spaces as spaces
import numpy as np
import torch
from habitat_mas.agents.actions.discussion_actions import *
from habitat_mas.utils import AgentArguments
from habitat_mas.utils.models import OpenAIModel

from habitat_baselines.common.storage import Storage
from habitat_baselines.common.tensor_dict import TensorDict
from habitat_baselines.rl.multi_agent.pop_play_wrappers import (
    MultiAgentPolicyActionData,
    MultiPolicy,
    MultiStorage,
    MultiUpdater,
    _merge_list_dict,
)
from habitat_baselines.rl.hrl.hierarchical_policy import HierarchicalPolicy
from habitat_baselines.rl.hrl.hl.llm_policy import LLMHighLevelPolicy
from habitat_baselines.rl.multi_agent.utils import (
    add_agent_names,
    add_agent_prefix,
    update_dict_with_agent_prefix,
)


ROBOT_RESUME_TEMPLATE = (
    '"{robot_key}" is a  "{robot_type}" agent.'
    " It has the following capabilities:\n\n"
    '"""\n{capabilities}\n"""\n\n'
)


def create_leader_prompt(robot_resume):
    LEADER_SYSTEM_PROMPT_TEMPLATE = (
        "You are a group discussion leader."
        " Your have to discuss with real robots (agents) to break an overall"
        " task to smaller subtasks and assign a subtask to each agent."
        "The agents are described below:\n\n"
        '"""\n{robot_resume}\n"""\n\n'
    )
    FORMAT_INSTRUCTION = (
        "You should assign subtasks to each agent based on their capabilities, following this format:\n\n"
        r"{robot_id||subtask_description}\n\n"
        'For example, if you want to assign a subtask "Navigate to targets_0" to agent_0, you should type:\n'
        r"{agent_0||Navigate to targets_0}\n\n"
        "Remember you must include the brackets and you MUST include all the robots in your response.\n"
        'If you think a robot should not have a subtask, you can assign it "Nothing to do".\n'
        "Remember the subtask target should always be objects and robots, not a region.\n"
    )

    return (
        LEADER_SYSTEM_PROMPT_TEMPLATE.format(robot_resume=robot_resume)
        + FORMAT_INSTRUCTION
    )


def create_leader_start_message(task_description, scene_description):
    LEADER_MESSAGE_TEMPLATE = (
        " The task is described below:\n\n"
        '"""\n{task_description}\n"""\n\n'
        "All agents are in a scene, the scene description is as follows:\n\n"
        '"""\n{scene_description}\n"""\n\n'
        "Now you should assign subtasks to each agent based on their capabilities, following the format in the system prompt."
    )

    return LEADER_MESSAGE_TEMPLATE.format(
        task_description=task_description, scene_description=scene_description
    )  # + FORMAT_INSTRUCTION


def create_robot_prompt(robot_type, robot_key, capabilities):
    ROBOT_GROUP_DISCUSS_SYSTEM_PROMPT_TEMPLATE = (
        # 'You are a "{robot_type}" robot with id "{robot_id}".'
        'You are a "{robot_type}" agent called "{robot_key}".'
        "You have the following capabilities:\n\n"
        '"""\n{capabilities}\n"""\n\n'
        # "The physics capabilities include its mobility, perception, and manipulation capabilities."
        ' You will receive a subtask from the leader. If you don\'t have any task to do, you will receive "Nothing to do". '
        " Your task is to check if you are able to complete the assigned task by common sense reasoning and if targets is within the range of your sensors and workspace."
        " You can generate python code to check if task is feasible numerically."
        " If you are going to pick or place objects, please pay attention to the shape type of the robot workspace when your generate code to check if a location is within the 3D space bounded by the shape."
        " If you are going to navigate in multi-floor environments, please pay attention to your mobility capabilities and floors."
        " If you need to perceive objects, you only need to pay attention to the fov, height, horizontal dist, etc. when you generate code to check if a location in within vertical perception range."
        " When considering manipulation task, you only need pay attention to center point, radius, bounding box, etc."
        " Notably, ALL coordinates are in [x, z, y] format, where the second coordinate represents height, and the first and third coordinates represent horizontal positions."
        " I will execute the code and give your the result to help you make decisions."
    )
    FORMAT_INSTRUCTION = (
        r" Please put all your code in a single python code block wrapped within ```python and ```."
        r' You MUST print the varible with "<name>: <value>" format you want to know in the code.'
        r" Finally, if you think the task is incorrect, you can explain the reason and ask the leader to modify it,"
        r' following this format: "{{no||<the reason>}}".'
        r' If you think the task is correct, you should confirm it by typing "{{yes}}".'
        r" Example responses: {{yes}}, {{no||I have no moving ability}}, {{no||The object is out of my arm workspace}}."
    )
    return (
        ROBOT_GROUP_DISCUSS_SYSTEM_PROMPT_TEMPLATE.format(
            robot_type=robot_type, robot_key=robot_key, capabilities=capabilities
        )
        + FORMAT_INSTRUCTION
    )


def create_robot_start_message(
    task_description, scene_description, compute_path: bool = False
):
    ROBOT_GROUP_DISCUSS_MESSAGE_TEMPLATE = (
        " Your task is to work with other agents to complete the assigned subtask described below:\n\n"
        '"""\n{task_description}\n"""\n\n'
        "The scene description is as follows:\n\n"
        '"""\n{scene_description}\n"""\n\n'
        " Notably, ALL coordinates are in[x, z, y] format, where the second coordinate represents height, and the first and third coordinates represent horizontal positions."
    )
    COMPUTE_PATH = (
        "Please infer the navigation path based on the region descriptions and determine if the path requires crossing floors. "
        "You should determine whether you can succeed based on your capabilities."
    )
    if compute_path:
        ROBOT_GROUP_DISCUSS_MESSAGE_TEMPLATE += COMPUTE_PATH

    return ROBOT_GROUP_DISCUSS_MESSAGE_TEMPLATE.format(
        task_description=task_description, scene_description=scene_description
    )


NO_MANIPULATION = "There are no explicit manipulation components"
NO_MOBILITY = "The provided URDF does not include specific joints"
NO_PERCEPTION = "UNKNOWN"


def get_text_capabilities(robot: dict):
    capabilities = ""
    if "mobility" in robot:
        mobility = json.dumps(robot["mobility"]["summary"])
        capabilities += f"Mobility capbility: {mobility}\n"
    if "perception" in robot:
        perception = json.dumps(robot["perception"]["summary"])
        capabilities += f"Perception capbility: {perception}\n"
    if "manipulation" in robot:
        manipulation = json.dumps(robot["manipulation"]["summary"])
        capabilities += f"Manipulation capbility: {manipulation}\n"
    return capabilities


def get_full_capabilities(robot: dict):
    capabilities = "The followinfg is a list of python dicts describing the robot's capabilities:\n"
    if "mobility" in robot:
        mobility = json.dumps(robot["mobility"])
        capabilities += f" - Mobility capability: {mobility}\n"
    if "perception" in robot:
        perception = json.dumps(robot["perception"])
        capabilities += f" - Perception capability: {perception}\n"
    if "manipulation" in robot:
        manipulation = json.dumps(robot["manipulation"])
        capabilities += f" - Manipulation capability: {manipulation}\n"
    return capabilities


def parse_leader_response(text):
    # Define the regular expression pattern
    pattern = r"\{(.*?)\|\|(.*?)\}"

    # Find all matches in the text
    matches = re.findall(pattern, text)

    # Create a dictionary from the matches
    robot_tasks = {
        robot_id: subtask_description for robot_id, subtask_description in matches
    }

    return robot_tasks


def parse_agent_response(text):
    # Define the regular expression pattern
    pattern = r"\{(yes|no)(?:\|\|(.*?))?\}"

    # Find all matches in the text
    matches = re.findall(pattern, text)

    if len(matches) == 0:
        print("No match found in agent response: ", text)
        return "no", text
    response, reason = matches[0]
    if response == "yes":
        reason = None  # Use None to indicate no reason
    elif response == "no":
        reason = (
            reason.strip() if reason else ""
        )  # Store the reason, strip spaces, or empty if none

    return response, reason


# DISCUSSION_TOOLS = [eval_python_code, add, subtract, multiply, divide]
DISCUSSION_TOOLS = []

ROBOT_DESCRIPTION = {
    "SpotRobot": "Spot is a legged base robot with 7-DOF arm of revolute joints.",
    "FetchRobot": "Fetch is a wheeled base robot with a 7-DOF arm of revolute joints",
    "StretchRobot": "Stretch is a wheeled base robot and a telescoping arm of prismatic joints.",
    "DJIDrone": "Drone is a DJI M100 drone with an RGBD sensor for its model credit",
}

def agent_centric_discussion(
    leader,
    agents,
    task_description,
    scene_description,
    should_agent_reflection,
    max_discussion_rounds=3,
) -> dict[str, str]:
    pass

def leader_centric_discussion(
    leader,
    agents,
    task_description,
    scene_description,
    should_agent_reflection,
    max_discussion_rounds=3,
) -> dict[str, str]:
    compute_path = "regions_description" in scene_description

    leader_start_message = create_leader_start_message(
        task_description=task_description, scene_description=scene_description
    )

    response = leader.chat(leader_start_message)
    robot_tasks = parse_leader_response(response)
    print("===============Task Description==============")
    print(task_description)
    print("===============Leader Response==============")
    print(response)
    print("===========================================")

    agent_response = {}
    if not should_agent_reflection:
        return robot_tasks

    for robot_id in robot_tasks:
        robot_model = agents[robot_id]
        robot_start_message = create_robot_start_message(
            task_description=robot_tasks[robot_id],
            scene_description=scene_description,
            compute_path=compute_path,
        )
        response = robot_model.chat(robot_start_message)
        agent_response[robot_id] = parse_agent_response(response)
        print("===============Robot Response==============")
        print(f"Robot {robot_id} response: {response}")
        print("===========================================")

    for _ in range(max_discussion_rounds):
        all_yes = True
        prompt = ""
        for robot_id, (response, reason) in agent_response.items():
            if response == "no":
                prompt += f"Robot {robot_id} response: {response}, reason: {reason}\n"
                all_yes = False
        if all_yes:
            return robot_tasks
        prompt += "Please modify the task and reassign the subtasks."

        response = leader.chat(prompt)
        robot_tasks = parse_leader_response(response)

        print("===============Leader Response==============")
        print(response)
        print("===========================================")
        for robot_id in robot_tasks:
            robot_model = agents[robot_id]
            robot_start_message = create_robot_start_message(
                task_description=robot_tasks[robot_id],
                scene_description=scene_description,
                compute_path=compute_path,
            )
            response = robot_model.chat(robot_start_message)
            agent_response[robot_id] = parse_agent_response(response)
            print("===============Robot Response==============")
            print(f"Robot {robot_id} response: {response}")
            print("===========================================")

    return robot_tasks


def group_discussion(
    robot_resume: dict,
    scene_description: str,
    task_description: str,
    save_chat_history=True,
    save_chat_history_dir="",
    episode_id=-1,
    should_group_discussion: bool = True,
    should_agent_reflection: bool = True,
    should_robot_resume: bool = True,
    should_numerical: bool = True,
) -> dict[str, AgentArguments]:
    if save_chat_history:
        episode_save_dir = os.path.join(save_chat_history_dir, str(episode_id))
        if not os.path.exists(episode_save_dir):
            os.makedirs(episode_save_dir)

    # Get robot resume from the beginning of the group discussion
    robot_resume = json.loads(robot_resume)
    robot_resume_prompt = ""
    capabilities_list = {}

    for robot_key in robot_resume:
        resume = robot_resume[robot_key]

        # For EMOS benchmark with numerical capabilities and robot resume
        if should_numerical and should_robot_resume:
            capabilities_list[robot_key] = get_full_capabilities(resume)
        # For EMOS benchmark with robot resume but without numerical capabilities
        elif should_robot_resume:
            capabilities_list[robot_key] = get_text_capabilities(resume)
        # For EMOS benchmark without robot resume
        else:
            capabilities_list[robot_key] = ROBOT_DESCRIPTION[resume["robot_type"]]

        robot_resume_prompt += ROBOT_RESUME_TEMPLATE.format(
            robot_key=robot_key,
            robot_type=resume["robot_type"],
            capabilities=capabilities_list[robot_key],
        )

    if not should_group_discussion:
        results = {}
        for robot_key in robot_resume:
            results[robot_key] = AgentArguments(
                robot_id=robot_key,
                robot_type=robot_resume[robot_key]["robot_type"],
                task_description=task_description,
                subtask_description="",
                chat_history=None,
            )

        return results

    leader_prompt = create_leader_prompt(robot_resume_prompt)

    leader = OpenAIModel(
        leader_prompt,
        DISCUSSION_TOOLS,
        discussion_stage=True,
        code_execution=False,
        enable_logging=save_chat_history,
        logging_file=os.path.join(episode_save_dir, "leader_group_chat_history.json"),
    )
    agents: Dict[str, OpenAIModel] = {}
    for robot_key in robot_resume:
        robot_prompt = create_robot_prompt(
            robot_resume[robot_key]["robot_type"],
            robot_key,
            capabilities_list[robot_key],
        )
        agents[robot_key] = OpenAIModel(
            robot_prompt,
            DISCUSSION_TOOLS,
            discussion_stage=True,
            code_execution=True,
            enable_logging=save_chat_history,
            logging_file=os.path.join(
                episode_save_dir, f"{robot_key}_group_chat_history.json"
            ),
        )

    robot_tasks = leader_centric_discussion(
        leader,
        agents,
        task_description,
        scene_description,
        should_agent_reflection,
    )

    results = {}
    for agent in agents:
        results[agent] = AgentArguments(
            robot_id=agent,
            robot_type=robot_resume[agent]["robot_type"],
            task_description=task_description,
            subtask_description=robot_tasks[agent],
            chat_history=agents[agent].chat_history,
        )
    leader_tokens = leader.token_usage
    robot_tokens = sum([agent.token_usage for agent in agents.values()])
    total_tokens = leader_tokens + robot_tokens
    print("===============Task Assignment Result==============")
    print(robot_tasks)
    print("===============group discussion token usage========================")
    print(f"Leader token usage: {leader_tokens}")
    print(f"Robots token usage: {robot_tokens}")
    print(f"Total token usage: {total_tokens}")
    print("========================================================")

    return results


class MultiLLMPolicy(MultiPolicy):
    """
    Wraps a set of LLM policies. Add group discussion stage before individual policy actions.
    """

    def __init__(self, update_obs_with_agent_prefix_fn, **kwargs):
        self._active_policies = []
        if update_obs_with_agent_prefix_fn is None:
            update_obs_with_agent_prefix_fn = update_dict_with_agent_prefix
        self._update_obs_with_agent_prefix_fn = update_obs_with_agent_prefix_fn

        # if True, the episode will terminate when all the agent choose to wait
        self.should_terminate_on_wait = kwargs.get("should_terminate_on_wait", False)

        # config for ablation study
        self.should_group_discussion = kwargs.get("should_group_discussion", True)
        self.should_agent_reflection = kwargs.get("should_agent_reflection", True)
        self.should_robot_resume = kwargs.get("should_robot_resume", True)
        self.should_numerical = kwargs.get("should_numerical", True)

    def set_active(self, active_policies):
        self._active_policies = active_policies

    def on_envs_pause(self, envs_to_pause):
        for policy in self._active_policies:
            policy.on_envs_pause(envs_to_pause)

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
        envs_text_context=[{}],
        **kwargs,
    ):
        # debug info
        # TODO: disable saving chat history for batch experiments
        save_chat_history = kwargs.get("save_chat_history", True)
        save_chat_history_dir = kwargs.get(
            "save_chat_history_dir", "./chat_history_output"
        )
        # Create a directory to save chat history
        if save_chat_history:
            # save dir format: chat_history_output/<date>/<config>/<episode_id>
            date_str = datetime.datetime.now().strftime("%Y-%m-%d")
            hydra_cfg = HydraConfig.get()
            config_str = hydra_cfg.job.config_name.split("/")[-1].replace(".yaml", "")
            # episode_save_dir = os.path.join(save_chat_history_dir, date_str, config_str, str(episode_id))
            save_chat_history_dir = os.path.join(
                save_chat_history_dir, date_str, config_str
            )

            if not os.path.exists(save_chat_history_dir):
                os.makedirs(save_chat_history_dir)

        n_agents = len(self._active_policies)
        split_index_dict = self._build_index_split(
            rnn_hidden_states, prev_actions, kwargs
        )
        agent_rnn_hidden_states = rnn_hidden_states.split(
            split_index_dict["index_len_recurrent_hidden_states"], -1
        )
        agent_prev_actions = prev_actions.split(
            split_index_dict["index_len_prev_actions"], -1
        )
        agent_masks = masks.split([1] * n_agents, -1)
        n_envs = prev_actions.shape[0]
        assert (
            n_envs == 1
        ), "Currently, the stage 2 only supports single environment with multiple agents."

        text_goal = observations["pddl_text_goal"][0].tolist()
        text_goal = "".join([chr(code) for code in text_goal]).strip()
        # text_goal = "Fill in a tank with water. The tank is 80cm high, 50cm wide, and 50cm deep. The tank is empty. agent_0 has a 1L backet and agent_1 has a 2L bucket. Use function call to compute how many time agent_1 and agent_2 needs to fill the tank to full."
        # Stage 1: If all prev_actions are zero, which means it is the first step of the episode, then we need to do group discussion
        # Given: Robot resume + Scene description + task instruction
        # Output: (Subtask decomposition) + task assignment
        envs_agent_arguments = []
        for i in range(n_envs):
            env_prev_actions = prev_actions[i]
            env_text_context = envs_text_context[i]
            # if no previous actions, then it is the first step of the episode
            if not env_prev_actions.any():
                if "robot_resume" in env_text_context:
                    robot_resume = env_text_context["robot_resume"]
                if "scene_description" in env_text_context:
                    scene_description = env_text_context["scene_description"]
                if "episode_id" in env_text_context:
                    episode_id = env_text_context["episode_id"]
                # print("===============Group Discussion===============")
                # print(robot_resume)
                # print("=============================================")
                # print(scene_description)
                # print("=============================================")
                # print(text_goal)
                # print("=============================================")

                agent_arguments = group_discussion(
                    robot_resume,
                    scene_description,
                    text_goal,
                    should_group_discussion=self.should_group_discussion,
                    should_agent_reflection=self.should_agent_reflection,
                    should_robot_resume=self.should_robot_resume,
                    should_numerical=self.should_numerical,
                    save_chat_history=save_chat_history,
                    save_chat_history_dir=save_chat_history_dir,
                    episode_id=episode_id,
                )
                envs_agent_arguments.append(agent_arguments)

                # Invalidate all action policies and flag them to be reinitialized
                for agent_i, policy in enumerate(self._active_policies):
                    policy._high_level_policy.llm_agent.initilized = False

        # Stage 2: Individual policy actions
        agent_actions = []
        for agent_i, policy in enumerate(self._active_policies):
            # collect assigned tasks for agent_i across all envs
            agent_i_handle = f"agent_{agent_i}"
            select_agent_arguments = [
                agent_arguments[agent_i_handle] if agent_i_handle in arguments else None
                for arguments in envs_agent_arguments
            ]
            agent_obs = self._update_obs_with_agent_prefix_fn(observations, agent_i)

            # TODO: Currently, the stage 2 only supports single environment with multiple agents.
            # TODO: Please update the code of agent initialization and policy action to support vectorized environment.
            # Initialize action execution agent with new context information
            policy: HierarchicalPolicy
            llm_policy: LLMHighLevelPolicy = policy._high_level_policy
            if not llm_policy.llm_agent.initilized:
                print("=================agent_task_assignment===================")
                args = select_agent_arguments[0]
                print(args)
                episode_id = envs_text_context[0]["episode_id"]
                episode_save_dir = os.path.join(save_chat_history_dir, str(episode_id))
                logging_path = os.path.join(
                    episode_save_dir, f"{agent_i_handle}_action_history.json"
                )

                llm_policy.llm_agent.init_agent(
                    robot_type=args.robot_type,
                    task_description=args.task_description,
                    subtask_description=args.subtask_description,
                    chat_history=args.chat_history,
                    enable_logging=save_chat_history,
                    logging_file=logging_path,
                )

            # Run the policy
            agent_actions.append(
                policy.act(
                    agent_obs,
                    agent_rnn_hidden_states[agent_i],
                    agent_prev_actions[agent_i],
                    agent_masks[agent_i],
                    deterministic,
                    envs_text_context=envs_text_context,
                    agent_arguments=select_agent_arguments,  # pass the task planning result to the policy
                )
            )

        if self.should_terminate_on_wait:
            should_terminate = True
            for agent_id, agent_action in enumerate(agent_actions):
                wait_id = self._active_policies[agent_id]._name_to_idx["wait"]
                if agent_action.skill_id != wait_id:
                    should_terminate = False
                    break
            if should_terminate:
                print("=================Terminate=================")
                print("All agents are waiting for next action.")
                print("===========================================")
                for agent_i, policy in enumerate(self._active_policies):
                    agent_actions[agent_i].actions[i, policy._stop_action_idx] = 1.0

        policy_info = _merge_list_dict([ac.policy_info for ac in agent_actions])
        batch_size = masks.shape[0]
        device = masks.device

        action_dims = split_index_dict["index_len_prev_actions"]

        # We need to split the `take_actions` if they are being assigned from
        # `actions`. This will be the case if `take_actions` hasn't been
        # assigned, like in a monolithic policy where there is no policy
        # hierarchicy.
        if any(ac.take_actions is None for ac in agent_actions):
            length_take_actions = action_dims
        else:
            length_take_actions = None

        def _maybe_cat(get_dat, feature_dims, dtype):
            all_dat = [get_dat(ac) for ac in agent_actions]
            # Replace any None with dummy data.
            all_dat = [
                torch.zeros((batch_size, feature_dims[ind]), device=device, dtype=dtype)
                if dat is None
                else dat
                for ind, dat in enumerate(all_dat)
            ]
            return torch.cat(all_dat, -1)

        rnn_hidden_lengths = [ac.rnn_hidden_states.shape[-1] for ac in agent_actions]
        return MultiAgentPolicyActionData(
            rnn_hidden_states=torch.cat(
                [ac.rnn_hidden_states for ac in agent_actions], -1
            ),
            actions=_maybe_cat(lambda ac: ac.actions, action_dims, prev_actions.dtype),
            values=_maybe_cat(
                lambda ac: ac.values, [1] * len(agent_actions), torch.float32
            ),
            action_log_probs=_maybe_cat(
                lambda ac: ac.action_log_probs,
                [1] * len(agent_actions),
                torch.float32,
            ),
            take_actions=torch.cat(
                [
                    ac.take_actions if ac.take_actions is not None else ac.actions
                    for ac in agent_actions
                ],
                -1,
            ),
            policy_info=policy_info,
            should_inserts=np.concatenate(
                [
                    ac.should_inserts
                    if ac.should_inserts is not None
                    else np.ones(
                        (batch_size, 1), dtype=bool
                    )  # None for monolithic policy, the buffer should be updated
                    for ac in agent_actions
                ],
                -1,
            ),
            length_rnn_hidden_states=rnn_hidden_lengths,
            length_actions=action_dims,
            length_take_actions=length_take_actions,
            num_agents=n_agents,
        )

    def _build_index_split(self, rnn_hidden_states, prev_actions, kwargs):
        """
        Return a dictionary with rnn_hidden_states lengths and action lengths that
        will be used to split these tensors into different agents. If the lengths
        are already in kwargs, we return them as is, if not, we assume agents
        have the same action/hidden dimension, so the tensors will be split equally.
        Therefore, the lists become [dimension_tensor // num_agents] * num_agents
        """
        n_agents = len(self._active_policies)
        index_names = [
            "index_len_recurrent_hidden_states",
            "index_len_prev_actions",
        ]
        split_index_dict = {}
        for name_index in index_names:
            if name_index not in kwargs:
                if name_index == "index_len_recurrent_hidden_states":
                    all_dim = rnn_hidden_states.shape[-1]
                else:
                    all_dim = prev_actions.shape[-1]
                split_indices = int(all_dim / n_agents)
                split_indices = [split_indices] * n_agents
            else:
                split_indices = kwargs[name_index]
            split_index_dict[name_index] = split_indices
        return split_index_dict

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks, **kwargs):
        split_index_dict = self._build_index_split(
            rnn_hidden_states, prev_actions, kwargs
        )
        agent_rnn_hidden_states = torch.split(
            rnn_hidden_states,
            split_index_dict["index_len_recurrent_hidden_states"],
            dim=-1,
        )
        agent_prev_actions = torch.split(
            prev_actions, split_index_dict["index_len_prev_actions"], dim=-1
        )
        agent_masks = torch.split(masks, [1, 1], dim=-1)
        all_value = []
        for agent_i, policy in enumerate(self._active_policies):
            agent_obs = self._update_obs_with_agent_prefix_fn(observations, agent_i)
            all_value.append(
                policy.get_value(
                    agent_obs,
                    agent_rnn_hidden_states[agent_i],
                    agent_prev_actions[agent_i],
                    agent_masks[agent_i],
                )
            )
        return torch.stack(all_value, -1)

    def get_extra(
        self, action_data: MultiAgentPolicyActionData, infos, dones
    ) -> List[Dict[str, float]]:
        all_extra = []
        for policy in self._active_policies:
            all_extra.append(policy.get_extra(action_data, infos, dones))
        # The action_data is shared across all policies, so no need to reutrn multiple times
        inputs = all_extra[0]
        ret: List[Dict] = []
        for env_d in inputs:
            ret.append(env_d)

        return ret

    @property
    def policy_action_space(self):
        # TODO: Hack for discrete HL action spaces.
        all_discrete = np.all(
            [
                isinstance(policy.policy_action_space, spaces.MultiDiscrete)
                for policy in self._active_policies
            ]
        )
        if all_discrete:
            return spaces.MultiDiscrete(
                tuple(
                    [policy.policy_action_space.n for policy in self._active_policies]
                )
            )
        else:
            return spaces.Dict(
                {
                    policy_i: policy.policy_action_space
                    for policy_i, policy in enumerate(self._active_policies)
                }
            )

    @property
    def policy_action_space_shape_lens(self):
        lens = []
        for policy in self._active_policies:
            if isinstance(policy.policy_action_space, spaces.Discrete):
                lens.append(1)
            elif isinstance(policy.policy_action_space, spaces.Box):
                lens.append(policy.policy_action_space.shape[0])
            else:
                raise ValueError(
                    f"Action distribution {policy.policy_action_space}" "not supported."
                )
        return lens

    @classmethod
    def from_config(
        cls,
        config,
        observation_space,
        action_space,
        update_obs_with_agent_prefix_fn: Optional[Callable] = None,
        **kwargs,
    ):
        kwargs["should_terminate_on_wait"] = (
            config.habitat.dataset.should_terminate_on_wait
        )
        kwargs["should_group_discussion"] = (
            config.habitat.dataset.should_group_discussion
        )
        kwargs["should_agent_reflection"] = (
            config.habitat.dataset.should_agent_reflection
        )
        kwargs["should_robot_resume"] = config.habitat.dataset.should_robot_resume
        kwargs["should_numerical"] = config.habitat.dataset.should_numerical

        return cls(update_obs_with_agent_prefix_fn, **kwargs)


class MultiLLMStorage(MultiStorage):
    def __init__(self, update_obs_with_agent_prefix_fn, **kwargs):
        super().__init__(update_obs_with_agent_prefix_fn, **kwargs)

    @classmethod
    def from_config(
        cls,
        config,
        observation_space,
        action_space,
        update_obs_with_agent_prefix_fn: Optional[Callable] = None,
        **kwargs,
    ):
        return cls(update_obs_with_agent_prefix_fn, **kwargs)


class MultiLLMUpdater(MultiUpdater):
    def __init__(self):
        self._active_updaters = []

    @classmethod
    def from_config(cls, config, observation_space, action_space, **kwargs):
        return MultiLLMUpdater()
