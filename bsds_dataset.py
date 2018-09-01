import os
import torch
from typing import List, Dict

import numpy as np
from scipy.misc import imresize

from torch.utils.data.dataset import Dataset
from bsds_wrapper import BSDSWrapper


class BSDSDataset(Dataset):

    images: Dict[str, np.ndarray] = {}
    labels: Dict[str, np.ndarray] = {}
    sample_names: List[str]
    wrapper: BSDSWrapper

    def __init__(self, is_test: bool, data_dir: str, output_size=None):
        self.images_dir = os.path.join(data_dir, "BSDS500/data/images")
        self.ground_truth_dir = os.path.join(data_dir, "BSDS500/data/groundTruth")

        self.wrapper = BSDSWrapper(data_dir)
        self.output_size = output_size

        self.sample_names = self.wrapper.test_sample_names if is_test else self.wrapper.train_sample_names

        for image in self.sample_names:
            self.add_image(image)

    def add_image(self, image):
        x = self.wrapper.read_image(image)
        x = x.transpose(2, 0, 1)
        x = np.array(x, dtype=np.float) / 256.0
        y = self.wrapper.load_boundaries(os.path.join(self.ground_truth_dir, image))[0]
        y = np.array(y, dtype=np.float)
        y = np.expand_dims(y, 0)
        self.images[image] = x
        self.labels[image] = y

    def random_crop(self, sample):
        image, label = sample
        h, w = image.shape[1:]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        return image[:, top: top + new_h, left: left + new_w], label[:, top: top + new_h, left: left + new_w]

    def __getitem__(self, index):
        out = self.images[self.sample_names[index]], self.labels[self.sample_names[index]]
        if self.output_size is not None:
            out = self.random_crop(out)
        return out

    def __len__(self):
        return len(self.sample_names)