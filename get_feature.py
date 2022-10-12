import argparse
import multiprocessing
import time

import numpy as np

from feature_extractor import get_fit_sample
from data_augmentation import batch_augment


def parser_args():
    parser = argparse.ArgumentParser(
        description='feature extraction from skeleton data.'
        )
    parser.add_argument('--path_to_data', type=str, 
                        default='/home/eddiej/'
                                'oosto_action_skeleton_v0.2.0.npz')
    parser.add_argument('--num_key_points', type=int, default=14)
    parser.add_argument('--num_frames', type=int, default=32)
    parser.add_argument('--num_cpus', type=int, default=-1)
    parser.add_argument('--saved_folder', type=str, default='./data/')
    args = parser.parse_args()
    return args


def check_num_key_points(num_key_points):
    if num_key_points != 14:
        raise ValueError('The number of num_key_points is not 14.'
                         'You need to make changes to `feature_extractor.py`'
                         'as the number is pre-define in the jit for '
                         'acceleration. If you have done this, you can skip '
                         'this check.')


def main():
    args = parser_args()
    num_key_points = args.num_key_points
    check_num_key_points(num_key_points)

    x = np.load(args.path_to_data)
    trainX = x['x_train'].reshape(-1, args.num_frames, num_key_points, 3)
    # confidence score is not used
    trainX = trainX[..., :2].transpose(0, 2, 1, 3)
    # the shape of trainX should be [batch, num_key_points, num_frames, 2]
    trainY = x['y_train'].argmax(1)

    if args.num_cpus > 0:
        num_cpus = args.num_cpus
    else:
        num_cpus = multiprocessing.cpu_count()

    trainX, trainY = batch_augment(trainX, trainY)

    testX = x['x_test'].reshape(-1, 32, num_key_points, 3)[..., :2]
    testX = testX.transpose(0, 2, 1, 3)
    testY = x['y_test'].argmax(1)

    pool = multiprocessing.Pool(num_cpus)

    # Test if the function works well before multiprocessing.
    _ = get_fit_sample(trainX[0])
    del _

    t = time.time()
    out = pool.map(get_fit_sample, trainX)
    pool.close()
    speed = time.time() - t

    speed = speed / trainX.shape[0] * 1000 * num_cpus / 8
    print('Speed %.2f ms per sample per 8 cpu' % speed)

    X, Y = [], []
    for idx, feat in enumerate(out):
        X.append(out[idx])
        Y.append(trainY[idx])

    np.save('%s/trainX' % args.saved_folder, np.array(X))
    np.save('%s/trainY' % args.saved_folder, np.array(Y))

    pool = multiprocessing.Pool(num_cpus)
    out = pool.map(get_fit_sample, testX)

    X, Y = [], []
    for idx, feat in enumerate(out):
        if feat is not None:
            X.append(out[idx])
            Y.append(testY[idx])

    np.save('%s/testX' % args.saved_folder, np.array(X))
    np.save('%s/testY' % args.saved_folder, np.array(Y))


if __name__ == '__main__':
    main()
