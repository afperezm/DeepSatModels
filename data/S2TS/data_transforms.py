import json
import random
import torch

import numpy as np
import torch.nn.functional as F

from copy import deepcopy
from scipy import ndimage
from torchvision import transforms

from utils.config_files_utils import get_params_values


def S2TS_segmentation_transform(model_config, stats_file, is_training):
    dataset_img_res = 24

    input_img_res = model_config['img_res']
    ground_truths = ['labels']
    max_seq_len = model_config['max_seq_len']

    with open(stats_file, "r") as file:
        norm_values = json.loads(file.read())

    transforms_list = []
    transforms_list.append(ToTensor())  # data from numpy arrays to torch.float32
    transforms_list.append(Normalize(norm_values['mean'], norm_values['std']))  # normalize all inputs individually

    if dataset_img_res != input_img_res:
        transforms_list.append(Crop(img_size=dataset_img_res, crop_size=input_img_res, random=is_training,
                                    ground_truths=ground_truths))  # random crop

    transforms_list.append(TileDates(H=model_config['img_res'], W=model_config['img_res'],
                                     doy_bins=None))  # tile day and year to shape TxWxHx1
    transforms_list.append(CutOrPad(max_seq_len=max_seq_len, random_sample=False,
                                    from_start=True))  # pad with zeros to maximum sequence length
    transforms_list.append(PermuteDimensions())

    return transforms.Compose(transforms_list)


# 1
class ToTensor(object):
    """
    Convert NumPy arrays in sample to Tensors.
    items in  : img, labels, doy
    items out : img, labels, doy
    """

    def __call__(self, sample):
        tensor_sample = {'inputs': torch.tensor(sample['img']).to(torch.float32),
                         'labels': torch.tensor(sample['labels'].astype(np.float32)).to(torch.float32).unsqueeze(-1),
                         'doy': torch.tensor(np.array(sample['doy'])).to(torch.float32)}
        return tensor_sample


# 2
class Normalize(object):
    """
    Normalize inputs as in https://arxiv.org/pdf/1802.02080.pdf
    items in  : img, labels, doy
    items out : img, labels, doy
    """

    def __init__(self, mean, std):
        self.mean = np.expand_dims(np.array(mean), axis=(0, 2, 3)).astype(np.float32)
        self.std = np.expand_dims(np.array(std), axis=(0, 2, 3)).astype(np.float32)

    def __call__(self, sample):
        sample['inputs'] = (sample['inputs'] - self.mean) / self.std
        sample['doy'] = sample['doy'] / 365.0001
        return sample


# 3
class Crop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, img_size, crop_size, random=False, ground_truths=[]):
        self.img_size = img_size
        self.crop_size = crop_size
        self.random = random
        if not random:
            self.top = int((img_size - crop_size) / 2)
            self.left = int((img_size - crop_size) / 2)
        self.ground_truths = ground_truths

    def __call__(self, sample):
        if self.random:
            top = torch.randint(self.img_size - self.crop_size, (1,))[0]
            left = torch.randint(self.img_size - self.crop_size, (1,))[0]
        else:  # center
            top = self.top
            left = self.left
        sample['inputs'] = sample['inputs'][:, :, top:top + self.crop_size, left:left + self.crop_size]
        for gt in self.ground_truths:
            sample[gt] = sample[gt][top:top + self.crop_size, left:left + self.crop_size]
        return sample


# 4
class TileDates(object):
    """
    Tile a 1d array to height (H), width (W) of an image.
    items in  : img, labels, doy
    items out : img, labels, doy
    """

    def __init__(self, H, W, doy_bins=None):
        assert isinstance(H, (int,))
        assert isinstance(W, (int,))
        self.H = H
        self.W = W
        self.doy_bins = doy_bins

    def __call__(self, sample):
        doy = self.repeat(sample['doy'], binned=self.doy_bins is not None)
        sample['inputs'] = torch.cat((sample['inputs'], doy), dim=1)
        del sample['doy']
        return sample

    def repeat(self, tensor, binned=False):
        if binned:
            out = tensor.unsqueeze(1).unsqueeze(1).repeat(1, self.H, self.W, 1)  # .permute(0, 2, 3, 1)
        else:
            out = tensor.repeat(1, self.H, self.W, 1).permute(3, 0, 1, 2)
        return out


# 5
class CutOrPad(object):
    """
    Pad series with zeros (matching series elements) to a max sequence length or cut sequential parts
    items in  : inputs, *inputs_backward, labels
    items out : inputs, *inputs_backward, labels, seq_lengths

    REMOVE DEEPCOPY OR REPLACE WITH TORCH FUN
    """

    def __init__(self, max_seq_len, random_sample=False, from_start=False):
        assert isinstance(max_seq_len, (int, tuple))
        self.max_seq_len = max_seq_len
        self.random_sample = random_sample
        self.from_start = from_start
        assert int(random_sample) * int(
            from_start) == 0, "choose either one of random, from start sequence cut methods but not both"

    def __call__(self, sample):
        seq_len = deepcopy(sample['inputs'].shape[0])
        sample['inputs'] = self.pad_or_cut(sample['inputs'])
        if "inputs_backward" in sample:
            sample['inputs_backward'] = self.pad_or_cut(sample['inputs_backward'])
        if seq_len > self.max_seq_len:
            seq_len = self.max_seq_len
        sample['seq_lengths'] = seq_len
        return sample

    def pad_or_cut(self, tensor, dtype=torch.float32):
        seq_len = tensor.shape[0]
        diff = self.max_seq_len - seq_len
        if diff > 0:
            # Pad
            tsize = list(tensor.shape)
            if len(tsize) == 1:
                pad_shape = [diff]
            else:
                pad_shape = [diff] + tsize[1:]
            tensor = torch.cat((tensor, torch.zeros(pad_shape, dtype=dtype)), dim=0)
        elif diff < 0:
            # Cut
            if self.random_sample:
                return tensor[self.random_subseq(seq_len)]
            elif self.from_start:
                start_idx = 0
            else:
                start_idx = torch.randint(seq_len - self.max_seq_len, (1,))[0]
            tensor = tensor[start_idx:start_idx + self.max_seq_len]
        return tensor

    def random_subseq(self, seq_len):
        return torch.randperm(seq_len)[:self.max_seq_len].sort()[0]


# 6
class PermuteDimensions(object):
    """
    Transform that permutes a tensor from TCHW to THWC format.
    """

    def __call__(self, sample):
        sample['inputs'] = sample['inputs'].permute(0, 2, 3, 1)
        return sample
