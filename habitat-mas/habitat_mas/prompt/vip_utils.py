"""Utils for visual iterative prompting.

A number of utility functions for VIP.
"""

import re

import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.distance as distance


def min_dist(coord, coords):
    if not coords:
        return np.inf
    xys = np.asarray([[coord.xy] for coord in coords])
    return np.linalg.norm(xys - np.asarray(coord.xy), axis=-1).min()


def coord_outside_image(coord, image, radius):
    (height, image_width, _) = image.shape
    x, y = coord.xy
    x_outside = x > image_width - 2 * radius or x < 2 * radius
    y_outside = y > height - 2 * radius or y < 2 * radius
    return x_outside or y_outside


def is_invalid_coord(coord, coords, radius, image):
    # invalid if too close to others or outside of the image
    pos_overlaps = min_dist(coord, coords) < 1.5 * radius
    return pos_overlaps or coord_outside_image(coord, image, radius)


def angle_mag_2_x_y(angle, mag, arm_coord, is_circle=False, radius=40):
    x, y = arm_coord
    x += int(np.cos(angle) * mag)
    y += int(np.sin(angle) * mag)
    if is_circle:
        x += int(np.cos(angle) * radius * np.sign(mag))
        y += int(np.sin(angle) * radius * np.sign(mag))
    return x, y


def coord_to_text_coord(coord, arm_coord, radius):
    return (
        int(coord.xy[0]),
        int(coord.xy[1]),
    )


def parse_response(response, answer_key='Arrow: ['):
    values = []
    if answer_key in response:
        print('parse_response from answer_key')
        arrow_response = response.split(answer_key)[-1].split(']')[0]
        for val in map(int, re.findall(r'\d+', arrow_response)):
            values.append(val)
    else:
        print('parse_response for all ints')
        for val in map(int, re.findall(r'\d+', response)):
            values.append(val)
    return values


def compute_errors(action, true_action, verbose=False):
    """Compute errors between a predicted action and true action."""
    l2_error = np.linalg.norm(action - true_action)
    cos_sim = 1 - distance.cosine(action, true_action)
    l2_xy_error = np.linalg.norm(action[-2:] - true_action[-2:])
    cos_xy_sim = 1 - distance.cosine(action[-2:], true_action[-2:])
    z_error = np.abs(action[0] - true_action[0])
    errors = {
        'l2': l2_error,
        'cos_sim': cos_sim,
        'l2_xy_error': l2_xy_error,
        'cos_xy_sim': cos_xy_sim,
        'z_error': z_error,
    }

    if verbose:
        print('action: \t', [f'{a:.3f}' for a in action])
        print('true_action \t', [f'{a:.3f}' for a in true_action])
        print(f'l2: \t\t{l2_error:.3f}')
        print(f'l2_xy_error: \t{l2_xy_error:.3f}')
        print(f'cos_sim: \t{cos_sim:.3f}')
        print(f'cos_xy_sim: \t{cos_xy_sim:.3f}')
        print(f'z_error: \t{z_error:.3f}')

    return errors


def plot_errors(all_errors, error_types=None):
    """Plot errors across iterations."""
    if error_types is None:
        error_types = [
            'l2',
            'l2_xy_error',
            'z_error',
            'cos_sim',
            'cos_xy_sim',
        ]

    _, axs = plt.subplots(2, 3, figsize=(15, 8))
    for i, error_type in enumerate(error_types):  # go through each error type
        all_iter_errors = {}
        for error_by_iter in all_errors:  # go through each call
            for itr in error_by_iter:  # go through each iteration
                if itr in all_iter_errors:  # add error to the iteration it happened
                    all_iter_errors[itr].append(error_by_iter[itr][error_type])
                else:
                    all_iter_errors[itr] = [error_by_iter[itr][error_type]]

        mean_iter_errors = [
            np.mean(all_iter_errors[itr]) for itr in all_iter_errors
        ]

        axs[i // 3, i % 3].plot(all_iter_errors.keys(), mean_iter_errors)
        axs[i // 3, i % 3].set_title(error_type)
    plt.show()
