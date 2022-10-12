import numpy as np

num_key_points = 14


def batch_augment(trainX: np.array,
                  trainY: np.array,
                  flip_aug: bool = True,
                  scale_aug: bool = True,
                  rotate_aug: bool = True):
    """
    Args:
        trainX (np.array): the batch training inputs. Numpy array of size
            [B, num_key_points, T, 2] where `B` is the number of samples,
            `num_key_points` is the number of key points, fixed as 14 in this
            task, and `T` is the number of frames.
        trainY (np.array): the batch training labels. Numpy array of size [B].
    """
    if flip_aug:
        left_kp = [0, 1, 2, 6, 7, 8]
        right_kp = [5, 4, 3, 11, 10, 9]

        X_flip = trainX.copy()
        kp_x = X_flip[..., 0]
        kp_x[kp_x != 0] = kp_x.max() - kp_x[kp_x != 0]

        new_order = list(range(num_key_points))
        for left, right in zip(left_kp, right_kp):
            new_order[left] = right
            new_order[right] = left

        X_flip = X_flip[:, new_order]

        trainX = np.concatenate([trainX, X_flip])
        trainY = np.concatenate([trainY] * 2)

    if scale_aug:
        augmented = [trainX]
        for scale in 4 / 3, 3 / 2, 2 / 3, 3 / 4:
            newX = trainX.copy()
            newX[..., 0] *= scale
            augmented.append(newX)

        trainX = np.concatenate(augmented)
        trainY = np.concatenate([trainY] * 5)

    if rotate_aug:
        augmented = [trainX]
        for theta in np.pi / 12, -np.pi / 12:
            M = [[np.cos(theta), np.sin(theta)],
                 [-np.sin(theta), np.cos(theta)]]
            M = np.array(M)
            newX = trainX @ M
            augmented.append(newX)

        trainX = np.concatenate(augmented)
        trainY = np.concatenate([trainY] * 3)

    return trainX, trainY
