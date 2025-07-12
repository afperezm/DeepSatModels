import os

import argparse
import torch

import numpy as np
import pandas as pd
import pickle as pkl

from itertools import chain
from tqdm import tqdm

from src.datasets.pastis import load_patch, load_target
from data.PASTIS24.data2windows import get_doy, unfold_reshape

PARAMS = None


def main():
    root_dir = PARAMS.root_dir
    save_dir = PARAMS.save_dir
    width, height = PARAMS.image_size, PARAMS.image_size
    window_size = PARAMS.window_size

    meta_patch = pd.read_json(os.path.join(root_dir, 'metadata.json'))
    meta_patch.index = meta_patch['Patch'].astype(int)
    meta_patch.sort_index(inplace=True)

    num_patches = meta_patch.shape[0]
    patches_indices = meta_patch.index

    folds_dir = os.path.join(save_dir, 'fold-paths')
    patches_dir = os.path.join(save_dir, 'pickle24x24')

    if folds_dir and (not os.path.exists(folds_dir)):
        os.makedirs(folds_dir)

    if patches_dir and (not os.path.exists(patches_dir)):
        os.makedirs(patches_dir)

    folds_files = {}

    for idx in tqdm(range(num_patches), 'Processing patches'):
        patch_idx = patches_indices[idx]

        fold = meta_patch.loc[patch_idx, 'Fold']

        patch = load_patch(meta_patch, root_dir, height, width, 'S2_10m', patch_idx)
        patch = patch.transpose((0, 3, 1, 2)).astype(np.float32)

        label = load_target(meta_patch, root_dir, height, width, patch_idx).astype(int)
        label = np.expand_dims(label, 0)

        dates = meta_patch[f'Dates-S2_10m'].iloc[idx]
        doy = np.array([get_doy(d.replace('-', '')[:8]) for d in dates])

        unfolded_patches = unfold_reshape(torch.tensor(patch), window_size).numpy()
        unfolded_labels = unfold_reshape(torch.tensor(label.astype(float)), window_size).numpy()

        for subpatch_idx in range(unfolded_patches.shape[0]):

            subpatch_path = os.path.join(patches_dir, f'{patch_idx}_{subpatch_idx:03d}.pickle')

            if not os.path.exists(subpatch_path):
                with open(subpatch_path, 'wb') as f:
                    pkl.dump({'img': unfolded_patches[subpatch_idx], 'labels': unfolded_labels[subpatch_idx],
                              'doy': doy}, f)

            if fold in folds_files:
                folds_files[fold].append(f'{os.path.basename(patches_dir)}/{patch_idx}_{subpatch_idx:03d}.pickle')
            else:
                folds_files[fold] = [f'{os.path.basename(patches_dir)}/{patch_idx}_{subpatch_idx:03d}.pickle']

    folds = sorted(list(folds_files.keys()))
    num_folds = len(folds)

    fold_sequence = []

    for idx in range(num_folds):
        rotated = folds[idx:] + folds[:idx]
        train = rotated[:num_folds - 2]
        val = [rotated[num_folds - 2]]
        test = [rotated[num_folds - 1]]
        fold_sequence.append([train, val, test])

    for fold, (train_folds, val_fold, test_fold) in enumerate(tqdm(fold_sequence, desc="Processing folds"), start=1):
        for split_name, split_folds in zip(['train', 'val', 'test'], [train_folds, val_fold, test_fold]):
            if split_name == 'train':
                split_path = os.path.join(folds_dir, f"folds_{''.join([str(f) for f in split_folds])}.csv")
            else:
                split_path = os.path.join(folds_dir, f"fold_{''.join([str(f) for f in split_folds])}.csv")
            if os.path.exists(split_path):
                continue
            split_files = list(chain.from_iterable(folds_files[f] for f in split_folds))
            split_df = pd.DataFrame(split_files)
            split_df.to_csv(split_path, header=None, index=False)


def parse_args():
    parser = argparse.ArgumentParser(description='Script for S2TS dataset splitting')
    parser.add_argument('--root_dir', help='Data root directory', required=True)
    parser.add_argument('--save_dir', help='Directory where to store unfold data', required=True)
    parser.add_argument('--image_size', help='Size of input images', type=int, required=True)
    parser.add_argument('--window_size', help='Size of extracted windows', type=int, default=24)
    return parser.parse_args()


if __name__ == '__main__':
    PARAMS = parse_args()
    main()
