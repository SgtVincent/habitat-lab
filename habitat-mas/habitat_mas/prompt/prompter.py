import numpy as np
import habitat_mas.prompt.vip as vip
import scipy

# Adjust radius of annotations based on size of the image
radius_per_pixel = 0.05


def run_prompter(im, points, samples, camera_info, skill_name):
    img_size = np.min(im.shape[:2])

    style = {
        'num_samples': 1,
        'circle_alpha': 0.6,
        'alpha': 0.8,
        'arrow_alpha': 0.0,
        'radius': int(img_size * radius_per_pixel),
        'thickness': 2,
        'fontsize': int(img_size * radius_per_pixel),
        'rgb_scale': 255,
        'focal_offset': 1,  # camera distance / std of action in z
    }

    action_spec = {
        'loc': [0, 0, 0],
        'scale': [100, 100, 100],
        'min_scale': [0.0, 0.0, 0.0],
        'min': [-30.0, -30.0, -30],
        'max': [30, 30, 30],
        'action_to_coord': 250,
        'robot': None,
    }
    actions = points[0]
    # Fit Gaussian distributions to the points' coordinates.
    loc_scale = [
        scipy.stats.norm.fit(
            [action[d] for action in actions])
        for d in range(3)
    ]
    action_spec['loc'] = [loc_scale[d][0] for d in range(3)]
    action_spec['min'] = [actions[:, d].min() for d in range(3)]
    action_spec['max'] = [actions[:, d].max() for d in range(3)]
    action_spec['scale'] = np.clip(
        [loc_scale[d][1] for d in range(3)],
        action_spec['min_scale'],
        None,
    )

    prompter = vip.VisualIterativePrompter(
        style, action_spec, vip.SupportedEmbodiments.HF_DEMO
    )

    arm_coord = (int(im.shape[1] / 2), int(im.shape[0] / 2))
    # center_mean = action_spec["loc"]
    # center_std = action_spec["scale"]
    # samples = prompter.sample_actions(im, arm_coord, center_mean, center_std, camera_info)
    image_circles_np = prompter.add_arrow_overlay_plt(
        skill_name, image=im, samples=samples, arm_xy=arm_coord, camera_info=camera_info
    )

    return image_circles_np


