from gym import spaces
import numpy as np
import pygame
import random

from mm_lfd.envs.sim_pusht.pusht_pc_env import PushTPCEnv


def find_and_sample_colors(image, color_values, K):
    """
    Find all positions in the image that match any of the colors in the list and randomly sample K points.

    :param image: np.ndarray, shape [H, W, C] with uint8 colors
    :param color_values: np.ndarray, shape [M, 3] with uint8 color values
    :param K: int, number of points to sample
    :return: list of sampled points
    """
    # Ensure color_values is a numpy array
    color_values = np.array(color_values, dtype=np.uint8)

    # Create a mask that is True where the image matches any color in color_values
    mask = np.zeros(image.shape[:2], dtype=bool)

    for color in color_values:
        color_mask = np.all(image == color, axis=-1)
        mask = np.logical_or(mask, color_mask)

    # Get the positions of all matching pixels
    matching_positions = np.argwhere(mask)

    # Randomly sample K points from the matching positions
    if len(matching_positions) >= K:
        sampled_positions = random.sample(list(matching_positions), K)
    else:
        sampled_positions = matching_positions.tolist()

    return np.array(sampled_positions)


class PushTDensePCEnv(PushTPCEnv):
    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10}

    def __init__(self, args, rng=None, rng_act=None):
        super().__init__(args, rng=rng, rng_act=rng_act)
        self.num_points = args.num_points

    def render(self, mode="rgb_array"):
        result = super().render(mode=mode)

        image = result["images"][0]

        # get a set of points that represent the current state
        pts = []
        object_colors = np.array([[119, 136, 153], [143, 163, 184]])
        sampled_points = find_and_sample_colors(image, object_colors, self.num_points)
        result["pc"] = np.concatenate(
            [sampled_points[:, [1, 0]], np.zeros_like(sampled_points[:, [0]])], axis=1
        )

        return result
