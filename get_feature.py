import argparse
import multiprocessing
import os
import pickle
import time

import numpy as np

from data_augmentation import batch_augment
from feature_extractor import get_fit_sample


def parser_args():
    parser = argparse.ArgumentParser(
        description='feature extraction from skeleton data.')
    parser.add_argument('--train_pickle',
                        type=str,
                        default='./data/ntu120_xsub_train.pkl')
    parser.add_argument('--val_pickle',
                        type=str,
                        default='./data/ntu120_xsub_val.pkl')
    parser.add_argument('--num_cpus', type=int, default=-1)
    parser.add_argument('--saved_folder', type=str, default='./data/')
    args = parser.parse_args()
    return args


def load_data(pickle_path):
    X, Y = [], []
    with open(pickle_path, 'rb') as f:
        x = pickle.load(f)
    for item in x:
        keypoint = item['keypoint']
        if keypoint.max() > 0:
            keypoint = keypoint.astype(np.float32).transpose(2, 1, 0, 3)
            X.append(keypoint)
            Y.append(item['label'])
    return X, np.array(Y)


def main():
    args = parser_args()

    trainX, trainY = load_data(args.train_pickle)
    trainX, trainY = batch_augment(trainX, trainY)

    valX, valY = load_data(args.val_pickle)

    if args.num_cpus > 0:
        num_cpus = args.num_cpus
    else:
        num_cpus = multiprocessing.cpu_count()

    pool = multiprocessing.Pool(num_cpus)

    # Test if the function works well before multiprocessing.
    _ = get_fit_sample(trainX[0])
    print(_.shape)
    del _

    t = time.time()
    out = pool.map(get_fit_sample, trainX)
    pool.close()
    speed = time.time() - t

    speed = speed / len(out) * 1000 * num_cpus / 8
    print('Speed %.2f ms per sample per 8 cpu' % speed)

    if not os.path.isdir(args.saved_folder):
        os.mkdir(args.saved_folder)

    np.save('%s/trainX' % args.saved_folder, np.array(out))
    np.save('%s/trainY' % args.saved_folder, trainY)

    pool = multiprocessing.Pool(num_cpus)
    out = pool.map(get_fit_sample, valX)

    np.save('%s/valX' % args.saved_folder, np.array(out))
    np.save('%s/valY' % args.saved_folder, valY)


if __name__ == '__main__':
    main()
