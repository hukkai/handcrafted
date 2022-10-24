from typing import List

import numpy as np

num_key_points = 17

left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]


def flip(input_keypoint):
    keypoint = input_keypoint.copy()
    kp_x = keypoint[..., 0]
    kp_x[kp_x != 0] = kp_x.max() - kp_x[kp_x != 0]

    new_order = list(range(num_key_points))
    for left, right in zip(left_kp, right_kp):
        new_order[left] = right
        new_order[right] = left

    keypoint = keypoint[new_order]
    return keypoint


def resize(input_keypoint, scale):
    keypoint = input_keypoint.copy()
    keypoint[..., 0] *= scale
    return keypoint


def rotate(input_keypoint, rotation_matrix):
    return input_keypoint @ rotation_matrix


def batch_augment(trainX: List[np.array],
                  trainY: np.array,
                  flip_aug: bool = True,
                  scale_aug: bool = True,
                  rotate_aug: bool = False):
    """
    Args:
        trainX (List[np.array]): the batch training inputs. A list of
            key-points. Each element in the list should be a numpy array of
            size (P, T, N, 2) where `P` equals to `num_key_points`, `T` is the
            number of frames and `N` is the number of persons.
        trainY (np.array): the batch training labels. Numpy array of size [B]
            where B equals to the length of `trainX`.
    """
    if flip_aug:
        X_flip = [flip(keypoint) for keypoint in trainX]
        trainX = trainX + X_flip
        trainY = np.concatenate([trainY] * 2)

    if scale_aug:
        augmented = []
        for scale in 4 / 3, 3 / 2, 2 / 3, 3 / 4:
            newX = [resize(keypoint, scale) for keypoint in trainX]
            augmented += newX

        trainX = trainX + augmented
        trainY = np.concatenate([trainY] * 5)

    if rotate_aug:
        augmented = []
        for theta in np.pi / 12, -np.pi / 12:
            M = [[np.cos(theta), np.sin(theta)],
                 [-np.sin(theta), np.cos(theta)]]
            M = np.array(M)
            newX = [rotate(keypoint, M) for keypoint in trainX]
            augmented += newX

        trainX = trainX + augmented
        trainY = np.concatenate([trainY] * 3)

    return trainX, trainY
