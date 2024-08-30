import numpy as np
from habitat_mas.prompt.vip_runner import vip_runner
from habitat_mas.prompt.vlms import GPT4V
import scipy

# Adjust radius of annotations based on size of the image
radius_per_pixel = 0.05


def run_vip(
    im,
    query,
    n_samples_init,
    n_samples_opt,
    n_iters,
    n_parallel_trials,
    openai_api_key,
    points,
    camera_info,
    skill_name,
):
    if not openai_api_key:
        print('Must provide OpenAI API Key')
        return []
    if im is None:
        print('Must specify image')
        return []
    if not query:
        print('Must specify description')
        return []

    img_size = np.min(im.shape[:2])
    # add some action spec
    style = {
        'num_samples': 12,
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

    vlm = GPT4V(openai_api_key=openai_api_key)
    return vip_runner(
        vlm,
        im,
        query,
        style,
        action_spec,
        actions,
        n_samples_init=n_samples_init,
        n_samples_opt=n_samples_opt,
        n_iters=n_iters,
        n_parallel_trials=n_parallel_trials,
        camera_info=camera_info,
        skill_name=skill_name
    )
