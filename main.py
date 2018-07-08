import os
import numpy as np
import torch
from scipy.misc import imresize
from torch.autograd import Variable

from bsds_dataset import BSDSDataset
import matplotlib.pyplot as plt

from unet import UNet

if __name__ == "__main__":
    """
    testing
    """
    model = UNet(3, depth=5, merge_mode='concat')
    data_dir = "./data/BSR"
    images_dir = os.path.join(data_dir, "BSDS500/data/images")
    groundtruth_dir = os.path.join(data_dir, "BSDS500/data/groundTruth")

    dataset = BSDSDataset(data_dir)

    for image in dataset.train_sample_names:
        x = dataset.read_image(image)
        y = dataset.load_boundaries(os.path.join(groundtruth_dir, image))

        x = imresize(x, (320, 320)).reshape((1, 320, 320, 3))
        x = x.transpose(0, 3, 1, 2)

        tensor = Variable(torch.FloatTensor(np.random.random((1, 3, 320, 320))))
        out = model(tensor)
        # loss = torch.sum(out)
        # loss.backward()

        out_cpu = out.detach().numpy().transpose(0, 2, 3, 1).reshape(320, 320, 3)

        plt.imshow(out_cpu)
        plt.show()
        plt.imshow(y[0])
        plt.show()

