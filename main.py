import argparse
import torch

from torch import Tensor
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from bsds_dataset import BSDSDataset
from bsds_wrapper import BSDSWrapper
from unet import UNet
import numpy as np


def get_training_batch(dataset: BSDSWrapper, batch_size: int):
    dataset.load_boundaries()


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_dir", type=str, default="checkpoint")
parser.add_argument("--load", help="load or not", action="store_true")
args = parser.parse_args()

def load_checkpoint(checkpoint_dir: str, mod: torch.nn.Module, optim: torch.optim.Optimizer):
    checkpoint = torch.load(checkpoint_dir)
    mod.load_state_dict(checkpoint['state_dict'])
    optim.load_state_dict(checkpoint['optimizer'])


""" Divides an image into 320x320 patches to be fed to the U-Net model."""
def divide_image(image: np.ndarray, num_channels: int):
    expanded = np.tile(image, (1, 2, 2))
    h = expanded.shape[1]
    w = expanded.shape[2]

    assert(w >= 320 and h >= 320)

    nh = int(np.ceil(image.shape[1] / 320)) * 320
    nw = int(np.ceil(image.shape[2] / 320)) * 320

    expanded = expanded[:, :nh, :nw]

    return np.reshape(
        np.transpose(
            np.reshape(expanded, (num_channels, nh // 320, 320, nw // 320, 320)),
                (1, 3, 0, 2, 4)
        ), (-1, num_channels, 320, 320)), nh // 320, nw // 320


""" Combines the output of a U-Net model (multiple 320x320 patches) back into the original-sized image. """
def combine_images(repeats_x: int, repeats_y: int, original_width: int, original_height: int, image: np.ndarray, num_channels: int):
    return np.reshape(
        np.transpose(
            np.reshape(image,
                       (num_channels, repeats_y, repeats_x, 320, 320)),
            (0, 1, 3, 2, 4)),
        (num_channels, repeats_y * 320, repeats_x * 320))[:, :original_height, :original_width]

""" Gets the output of the model on an arbitrarily-sized image.
    Uses the divide_image routine to process the image in patches.
 """
def get_model_output(model: UNet, image: np.ndarray):
    original_height = image.shape[1]
    original_width = image.shape[2]

    tiled_x, rx, ry = divide_image(image, 3)
    tensor_x = torch.Tensor(tiled_x).float().cuda()
    out = model(tensor_x)
    combined_out = combine_images(rx, ry, original_width, original_height, out.cpu().detach().numpy(), 1)
    return combined_out


def evaluate(model: UNet, dset: BSDSDataset):
    for image in dset.wrapper.test_sample_names:
        x = dset.images[image]
        y = dset.labels[image]

        combined_out = get_model_output(model, x)

        disp_image_output(x, y, combined_out)


def disp_image_output(x: np.ndarray, y: np.ndarray, out: np.ndarray):
    out_reshaped = np.squeeze(out)
    x_reshaped = np.transpose(x, (1, 2, 0))
    y_reshaped = np.squeeze(y)

    plt.imshow(x_reshaped * 256)
    plt.show()
    plt.imshow(y_reshaped)
    plt.show()
    plt.imshow(out_reshaped * 256)
    plt.show()

def disp_output(x: torch.Tensor, y: torch.Tensor, out: torch.Tensor):
    out_model = out.cpu().detach().numpy().transpose(0, 2, 3, 1).reshape(-1, 320, 320)
    out_cpu = x.cpu().detach().numpy().transpose(0, 2, 3, 1).reshape(-1, 320, 320, 3)
    out_cpu_y = y.cpu().detach().numpy().reshape(-1, 320, 320)


if __name__ == "__main__":
    """
    testing
    """
    model = UNet(num_classes=1, depth=5, merge_mode='concat').cuda()

    data_dir: str = "../BSR"

    dset = BSDSDataset(False, data_dir, (320, 320))
    mb_size = 8
    loader = DataLoader(dset, batch_size=mb_size)

    test_loader = DataLoader(BSDSDataset(True, data_dir))

    optimizer = Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)

    if args.load:
        load_checkpoint(checkpoint_dir=args.checkpoint_dir, mod=model, optim=optimizer)

    iteration = 0

    while True:
        for x, y in loader:

            x = x.float().cuda()
            y = y.float().cuda()

            out = model(x).cuda()

            optimizer.zero_grad()
            loss = BCEWithLogitsLoss()(out, y)
            loss.backward()

            optimizer.step()

            print('Loss: ', loss.cpu().detach().numpy())

            if iteration % 1000 == 999:
                disp_output(x, y, out)
                torch.save(model.state_dict(), args.checkpoint_dir)

            iteration += 1
            evaluate(model, dset)

