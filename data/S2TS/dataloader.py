import os
import sys
import torch

import pandas as pd
import pickle as pkl

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


def get_distributed_dataloader(root_dir, paths_file, rank, world_size, transform=None, batch_size=32, num_workers=4,
                               shuffle=True, return_paths=False):
    dataset = S2TSWindowedDataset(root_dir=root_dir, csv_file=paths_file, transform=transform,
                                  return_paths=return_paths)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                            pin_memory=True, sampler=sampler)
    return dataloader


def get_dataloader(root_dir, paths_file, transform=None, batch_size=32, num_workers=4, shuffle=True,
                   return_paths=False, collate_fn=None):
    dataset = S2TSWindowedDataset(root_dir=root_dir, csv_file=paths_file, transform=transform,
                                  return_paths=return_paths)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                            collate_fn=collate_fn)
    return dataloader


class S2TSWindowedDataset(Dataset):
    """
    PyTorch Dataset class to load samples from a windowed Sentinel-2 satellite images dataset.
    """

    def __init__(self, root_dir, csv_file, transform=None, return_paths=False):
        """
        Args:
            root_dir (string): Directory with all the images.
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
            return_paths (boolean): Flag to indicate return of images paths.
        """
        self.root_dir = root_dir
        if type(csv_file) == str:
            self.data_paths = pd.read_csv(csv_file, header=None)
        elif type(csv_file) in [list, tuple]:
            self.data_paths = pd.concat([pd.read_csv(csv_file_, header=None) for csv_file_ in csv_file],
                                        axis=0).reset_index(drop=True)
        self.transform = transform
        self.return_paths = return_paths

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):

        subpatch_name = self.data_paths.iloc[index, 0]
        subpatch_path = os.path.join(self.root_dir, subpatch_name)

        with open(subpatch_path, 'rb') as handle:
            sample = pkl.load(handle, encoding='latin1')

        if self.transform:
            sample = self.transform(sample)

        if self.return_paths:
            return sample, subpatch_path

        return sample


def my_collate(batch):
    """
    Filter out sample where mask is zero everywhere.
    """
    index = [b['unk_masks'].sum(dim=(0, 1, 2)) != 0 for b in batch]
    batch = [b for i, b in enumerate(batch) if index[i]]
    return torch.utils.data.dataloader.default_collate(batch)


if __name__ == "__main__":

    data_dir = sys.argv[1]
    train_paths = sys.argv[2]

    train_dataloader = get_dataloader(data_dir, train_paths, batch_size=1)

    for batch_idx, train_batch in enumerate(train_dataloader):
        data, targets, dates = train_batch['img'], train_batch['labels'], train_batch['doy']

        print(f"batch - {batch_idx} - ", data.shape, targets.shape, dates.shape)
