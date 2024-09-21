"""VIP."""

import json
import re

import cv2
import habitat_mas.prompt.vip as vip
# TODO(zxz)
# 1. 将sample的点选做是传进去为true的点
# 2. sample服从传进去为true的点的正态分布
# 3. 设置返回是一堆点的坐标
# 4. 可视化visual prompting以及返回的结果（在最高层加一个新的sensor）


def make_prompt(description, top_n=3):
    return f"""
INSTRUCTIONS:
You are tasked to locate an object, region, or point in space in the given annotated image according to a description.
The image is annoated with numbered circles.
Choose the top {top_n} circles that have the most overlap with and/or is closest to what the description is describing in the image.
You are a five-time world champion in this game.
Give a one sentence analysis of why you chose those points.
Provide your answer at the end in a valid JSON of this format:

{{"points": []}}

DESCRIPTION: {description}
IMAGE:
""".strip()


def extract_json(response, key):
    json_part = re.search(r"\{.*\}", response, re.DOTALL)
    parsed_json = {}
    if json_part:
        json_data = json_part.group()
        # Parse the JSON data
        parsed_json = json.loads(json_data)
    else:
        print("No JSON data found ******\n", response)
    return parsed_json[key]


def vip_perform_selection(skill_name, prompter, vlm, im, desc, arm_coord, samples, camera_info, top_n):
    """Perform one selection pass given samples."""
    image_circles_np = prompter.add_arrow_overlay_plt(
        skill_name=skill_name, image=im, samples=samples, arm_xy=arm_coord, camera_info=camera_info
    )

    _, encoded_image_circles = cv2.imencode(".png", image_circles_np)

    prompt_seq = [make_prompt(desc, top_n=top_n), encoded_image_circles]
    response = vlm.query(prompt_seq)

    try:
        arrow_ids = extract_json(response, "points")
    except Exception as e:
        print(e)
        arrow_ids = []
    return arrow_ids, image_circles_np


def vip_runner(
    vlm,
    im,
    desc,
    style,
    action_spec,
    actions=None,
    n_samples_init=25,
    n_samples_opt=10,
    n_iters=3,
    n_parallel_trials=1,
    camera_info=None,
    skill_name=None,
):
    """VIP."""

    prompter = vip.VisualIterativePrompter(
        style, action_spec, vip.SupportedEmbodiments.HF_DEMO
    )

    output_ims = []
    arm_coord = (int(im.shape[1] / 2), int(im.shape[0] / 2))

    for i in range(n_parallel_trials):
        center_mean = action_spec["loc"]
        center_std = action_spec["scale"]
        for itr in range(n_iters):
            if itr == 0:
                style["num_samples"] = n_samples_init
            else:
                style["num_samples"] = n_samples_opt
            samples = prompter.sample_actions(skill_name, im, arm_coord, center_mean, center_std,
                                              camera_info, true_action=actions)
            arrow_ids, image_circles_np = vip_perform_selection(
                skill_name, prompter, vlm, im, desc, arm_coord, samples, camera_info, top_n=1
            )

            # plot sampled circles as red
            selected_samples = []
            for sample in samples:
                if int(sample.label) in arrow_ids:
                    sample.coord.color = (255, 0, 0)
                    selected_samples.append(sample)
            image_circles_marked_np = prompter.add_arrow_overlay_plt(
                skill_name, image_circles_np, selected_samples, arm_coord, camera_info
            )
            output_ims.append(image_circles_marked_np)

            # if at last iteration, pick one answer out of the selected ones
            if itr == n_iters - 1:
                arrow_ids, _ = vip_perform_selection(
                    skill_name, prompter, vlm, im, desc, arm_coord, selected_samples, camera_info, top_n=1
                )

                selected_samples = []
                for sample in samples:
                    if int(sample.label) in arrow_ids:
                        sample.coord.color = (255, 0, 0)
                        selected_samples.append(sample)
                image_circles_marked_np = prompter.add_arrow_overlay_plt(
                    skill_name, im, selected_samples, arm_coord, camera_info
                )
                output_ims.append(image_circles_marked_np)
            center_mean, center_std = prompter.fit(arrow_ids, samples)

    return output_ims, selected_samples
