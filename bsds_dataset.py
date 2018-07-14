import os
from typing import List

import numpy as np
from scipy.misc import imresize
from tensorflow.python.data import Dataset

from bsds_wrapper import BSDSWrapper


class BSDSDataset(Dataset):

    labels: List[np.ndarray]
    images: List[np.ndarray]
    wrapper: BSDSWrapper

    def __init__(self, data_dir: str):
        images_dir = os.path.join(data_dir, "BSDS500/data/images")
        ground_truth_dir = os.path.join(data_dir, "BSDS500/data/groundTruth")

        self.wrapper = BSDSWrapper(data_dir)
        self.images = []
        self.labels = []

        for image in self.wrapper.train_sample_names:
            x = np.array(self.wrapper.read_image(image), dtype=np.float) / 256.0
            x = imresize(x, (320, 320)).reshape((320, 320, 3))
            x = x.transpose(2, 0, 1)
            y = np.array(self.wrapper.load_boundaries(os.path.join(ground_truth_dir, image))[0], dtype=np.float)
            y = imresize(y, (320, 320)).reshape((320, 320))
            self.images.append(x)
            self.labels.append(y)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return len(self.images)
