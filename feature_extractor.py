import numpy as np
from numba import jit

# Define some constants
BINS = 9

num_key_points = 17
indexes = [(i, j) for i in range(num_key_points)
           for j in range(i + 1, num_key_points)]
num_pairs = len(indexes)
indexes = np.array(indexes).T


@jit(nopython=True, fastmath=True)
def helper(array: np.ndarray, bins: int, num_hist: int, max_val: float):
    """Given a numpy array of size (num_hist, N), returns a histogram array of
    size (num_hist, bins). Each row of the output array is the histogram of the
    corresponding row of the input array.

    Args:
        array (np.ndarray): numpy array of size (num_hist, N).
        bins (int): number of bins.
        num_hist (int): number of histograms.
        max_val (float): The maximum value in the array.
            The mininmum value in the array should be (no smaller than) 0. It
            is the users' responsibility to check this.

    Returns:
        np.ndarray: histograms of size (num_hist, bins).
    """
    array = np.round_(array / (max_val / (bins - 1)), 0, array)
    array = array.astype(np.uint8)
    all_histograms = np.zeros((num_hist, bins))
    for i in range(num_hist):
        for j in array[i]:
            all_histograms[i, j] += 1
    all_histograms /= array.shape[1]
    all_histograms -= 1 / bins
    return all_histograms


@jit(nopython=True, fastmath=True)
def get_spatial_features(keypoint: np.array, bins: int):
    """Given the keypoint skeleton, a numpy array of size (P, T, N, 2), returns
    the spatial features, a one-dimensional array.

    Args:
        keypoint (np.array): numpy array of size of (P, T, N, 2) where `P` is
            the number of key points, `T` is the number of frames and `N` is
            the number of persons.
        bins (int): number of bins.
    Returns:
        np.ndarray: the one-dimensional spatial features.
    """

    # shape of vectors: P(P-1)/2, T, N, 2
    vectors = keypoint[indexes[0]] - keypoint[indexes[1]]
    # shape of vectors: P(P-1)/2, T * N, 2
    vectors = vectors.reshape(num_pairs, -1, 2)

    distance = np.sqrt(np.square(vectors).sum(2))  # shape: P(P-1)/2, T * N

    avg_distance = distance.sum(1)  # shape: P(P-1) / 2
    avg_distance /= avg_distance.max() + 1e-8

    distance_stats = np.zeros((num_pairs, 2))
    for i in range(num_pairs):
        max_ = distance[i].max() + 1e-8
        array_ = distance[i] / max_
        distance_stats[i, 0] = array_.mean()
        distance_stats[i, 1] = array_.std()

    angle = np.arctan2(vectors[:, :, 1], vectors[:, :, 0]) + np.pi
    # shape: P(P-1) / 2, T * N

    angle_histograms = helper(angle, bins, num_pairs, max_val=np.pi * 2)
    angle_histograms = angle_histograms.reshape(-1)  # shape: P(P-1) / 2 * bins

    feature = np.concatenate(
        (avg_distance, distance_stats.reshape(-1), angle_histograms))
    return feature


@jit(nopython=True, fastmath=True)
def get_temporal_features(keypoint: np.array, steps: np.array, bins: int):
    """Given the keypoint skeleton, a numpy array of size (P, T, N, 2), returns
    the temporal features, a one-dimensional array.

    Args:
        keypoint (np.array): numpy array of size of (P, T, N, 2) where `P` is
            the number of key points, `T` is the number of frames and `N` is
            the number of persons.
        steps (np.array): one-dimensional integer array of the temporal steps.
        bins (int): number of bins.
    Returns:
        np.ndarray: the one-dimensional spatial features.
    """

    num_steps = len(steps)
    all_histograms = np.zeros((num_steps, num_key_points, bins))
    distance_feature = np.zeros((3, num_steps, num_key_points))

    for i in range(num_steps):
        step = steps[i]
        move = keypoint[:, :-step] - keypoint[:, step:]
        # shape: P, T - step, N, 2

        angle = np.arctan2(move[..., 1], move[..., 0]) + np.pi
        # shape: P, T - step, N
        angle = angle.reshape(num_key_points, -1)

        histograms = helper(angle, bins, num_key_points, max_val=np.pi * 2)
        all_histograms[i] = histograms

        for j in range(num_key_points):
            move_vectors = move[j]  # shape:  T - step, N, 2
            distance = np.sqrt(np.square(move_vectors).sum(-1))
            distance_feature[0, i, j] = distance.mean()
            distance /= distance.max() + 1e-8
            distance_feature[1, i, j] = distance.mean()
            distance_feature[2, i, j] = distance.std()

    distance_feature[0] /= distance_feature[0].max() + 1e-8

    feature = np.concatenate(
        (distance_feature.reshape(-1), all_histograms.reshape(-1)))
    return feature


triplet = np.array([[0, 5, 7], [0, 5, 11], [0, 6, 8], [0, 6, 12], [5, 7, 9],
                    [6, 8, 10], [5, 11, 13], [5, 11, 12], [11, 13, 15],
                    [6, 12, 14], [12, 14, 16], [11, 12, 14]]).T

triplet0, triplet1, triplet2 = triplet


def get_angle(x: np.array, y: np.array):
    """Given two numpy arrays of the same size [N1, N2, ..., Nk, 2], returns a
    numpy array of size the [N1, N2, ..., Nk], the angle between the two
    arrays."""
    dot = (x * y).sum(-1)
    norm_x = np.linalg.norm(x, axis=-1)
    norm_y = np.linalg.norm(y, axis=-1)
    cos = dot / (norm_x * norm_y + 1e-7)
    theta = np.arccos(np.clip(cos, -1.0, 1.0))
    return theta


def get_triplet_features(keypoint: np.array, bins: int = BINS):
    """Given the keypoint skeleton, a numpy array of size (P, T, N, 2), returns
    the triplet features, a one-dimensional array.

    Suppose keypoint p1 and p2 are connected with p3. The angle between
        p3 -> p1 and p2 -> p3 is a triplet feature.
    Args:
        keypoint (np.array): numpy array of size of (P, T, N, 2) where `P` is
            the number of key points, `T` is the number of frames and `N` is
            the number of persons.
        steps (np.array): one-dimensional integer array of the temporal steps.
        bins (int): number of bins.
    Returns:
        np.ndarray: the one-dimensional spatial features.
    """
    mid = keypoint[triplet1]  # shape: num_of_triplets, T, N, 2
    vector1 = mid - keypoint[triplet0]  # shape: num_of_triplets, T, N, 2
    vector2 = keypoint[triplet2] - mid  # shape: num_of_triplets, T, N, 2
    theta = get_angle(vector1, vector2)  # shape: num_of_triplets, T, N
    L = theta.shape[0]  # num_of_triplets
    theta = theta.reshape(L, -1)
    all_histograms = helper(theta, bins=bins, num_hist=L, max_val=np.pi)
    return all_histograms.reshape(-1)


def get_fit_sample(keypoint: np.array, bins: float = BINS):
    x = []
    t = round(keypoint.shape[1] * 0.6)
    for j in range(2):
        if j == 0:
            keypoint_clip = keypoint[:, :t]
        elif j == 1:
            keypoint_clip = keypoint[:, -t:]
        x.append(get_spatial_features(keypoint_clip, bins=bins))
        num_frames = keypoint_clip.shape[1]
        steps = [1, 2, 4, 8]
        steps += [num_frames // segment for segment in [7, 14, 21, 28]]
        steps = [max(min(step, num_frames - 2), 1) for step in steps]
        steps = np.array(steps, dtype=np.int64)
        x.append(get_temporal_features(keypoint_clip, bins=bins, steps=steps))
        x.append(get_triplet_features(keypoint_clip, bins=bins))
    x = np.concatenate(x).astype(np.float32)
    return x
