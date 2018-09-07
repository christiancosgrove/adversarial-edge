import os
import torch
from typing import List, Dict, Tuple

import numpy as np
from scipy.misc import imresize

from torch.utils.data.dataset import Dataset
from bsds_wrapper import BSDSWrapper
from scipy.ndimage.filters import gaussian_filter


class BSDSDataset(Dataset):

    images: Dict[str, np.ndarray] = {}
    labels: Dict[str, np.ndarray] = {}
    sample_names: List[str]
    duplicated_names: List[str]
    indices: List[int]
    wrapper: BSDSWrapper

    def __init__(self, is_test: bool, data_dir: str, output_size=None):
        self.images_dir = os.path.join(data_dir, "BSDS500/data/images")
        self.ground_truth_dir = os.path.join(data_dir, "BSDS500/data/groundTruth")

        self.wrapper = BSDSWrapper(data_dir)
        self.output_size = output_size

        self.sample_names = self.wrapper.test_sample_names if is_test else self.wrapper.train_sample_names
        self.duplicated_names = []
        self.indices = []
        for image in self.sample_names:
            self.add_image(image)

    def add_image(self, image):
        x = self.wrapper.read_image(image)
        x = x.transpose(2, 0, 1)
        x = np.array(x, dtype=np.float)
        ys = self.wrapper.load_boundaries(os.path.join(self.ground_truth_dir, image))
        ys = np.array(ys, dtype=np.float)
        self.duplicated_names.extend([image for _ in range(ys.shape[0])])
        self.indices.extend(list(range(ys.shape[0])))
        self.images[image] = x
        self.labels[image] = ys

    def random_crop(self, sample):
        image, label = sample
        h, w = image.shape[1:]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        return image[:, top: top + new_h, left: left + new_w], gaussian_filter(label[:, top: top + new_h, left: left + new_w], sigma=(0, 0, 0))

    def __getitem__(self, index):
        out = self.images[self.duplicated_names[index]], self.labels[self.duplicated_names[index]][self.indices[index]:self.indices[index] + 1]
        if self.output_size is not None:
            out = self.random_crop(out)
        return out

    def __len__(self):
        return len(self.duplicated_names)