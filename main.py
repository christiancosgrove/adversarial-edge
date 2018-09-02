import argparse
import os

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
from bsds import evaluate_boundaries
import tqdm
from torch.nn import functional as F


def get_training_batch(dataset: BSDSWrapper, batch_size: int):
    dataset.load_boundaries()


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_dir", type=str, default="checkpoint")
parser.add_argument("--load", help="load or not", action="store_true")
args = parser.parse_args()


def load_checkpoint(checkpoint_dir: str, mod: torch.nn.Module, optim: torch.optim.Optimizer):
    checkpoint = torch.load(checkpoint_dir)
    mod.load_state_dict(checkpoint['checkpoint'])
    optim.load_state_dict(checkpoint['optimizer'])


def save_checkpoint(filename: str, model: torch.nn.Module, optim: torch.optim.Optimizer):
    check = {}
    check['state_dict'] = model.state_dict()
    check['optimizer'] = optim.state_dict()
    torch.save(check, filename)


""" Divides an image into 320x320 patches to be fed to the U-Net model."""


def divide_image(image: np.ndarray, num_channels: int):
    expanded = np.tile(image, (1, 2, 2))
    h = expanded.shape[1]
    w = expanded.shape[2]

    assert (w >= 320 and h >= 320)

    nh = int(np.ceil(image.shape[1] / 320)) * 320
    nw = int(np.ceil(image.shape[2] / 320)) * 320

    expanded = expanded[:, :nh, :nw]

    return np.reshape(
        np.transpose(
            np.reshape(expanded, (num_channels, nh // 320, 320, nw // 320, 320)),
            (1, 3, 0, 2, 4)
        ), (-1, num_channels, 320, 320)), nh // 320, nw // 320


""" Combines the output of a U-Net model (multiple 320x320 patches) back into the original-sized image. """


def combine_images(repeats_x: int, repeats_y: int, original_width: int, original_height: int, image: np.ndarray,
                   num_channels: int):
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
    out = F.sigmoid(model(tensor_x))
    combined_out = combine_images(rx, ry, original_width, original_height, out.cpu().detach().numpy(), 1)
    return combined_out


def print_results(sample_results, threshold_results, overall_result):
    print('Per image:')
    for sample_index, res in enumerate(sample_results):
        print('{:<10d} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(
            sample_index + 1, res.threshold, res.recall, res.precision, res.f1))

    print('')
    print('Per threshold:')
    for thresh_i, res in enumerate(threshold_results):
        print('{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(
            res.threshold, res.recall, res.precision, res.f1))

    print('')
    print('Summary:')
    print('{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(
        overall_result.threshold, overall_result.recall, overall_result.precision, overall_result.f1,
        overall_result.best_recall, overall_result.best_precision, overall_result.best_f1,
        overall_result.area_pr))


def precision_recall_chart(threshold_results, title: str):
    prec = []
    rec = []

    for res in threshold_results:
        prec.append(res.precision)
        rec.append(res.recall)

    fig, ax = plt.subplots(1, 1)
    ax.plot(rec, prec)
    fig.savefig('figs/{}.png'.format(title))
    plt.close(fig)


def evaluate(model: UNet, dset: BSDSDataset, iteration: int):
    def load_prediction(image: str):
        x = dset.images[image]
        return np.squeeze(get_model_output(model, x))

    def load_gt_boundary(image: str):
        return dset.labels[image]

    sample_results, threshold_results, overall_result = \
        evaluate_boundaries.pr_evaluation(20, dset.wrapper.test_sample_names, load_gt_boundary, load_prediction,
                                          progress=tqdm.tqdm)

    precision_recall_chart(threshold_results, 'iteration_{}'.format(iteration))

    # print_results(sample_results, threshold_results, overall_result)


def disp_image_output(x: np.ndarray, y: np.ndarray, out: np.ndarray):
    disp_image_single(x)
    disp_edge_single(y)
    disp_edge_single(y)


def disp_image_single(x: np.ndarray, title=None):
    x_reshaped = np.transpose(x, (1, 2, 0))
    plt.imshow(x_reshaped * 256)
    if title is not None:
        plt.title(title)
    plt.show()


def disp_edge_single(y: np.ndarray, title=None):
    y_reshaped = np.squeeze(y)
    plt.imshow(y_reshaped)
    if title is not None:
        plt.title(title)
    plt.show()


def attack(model: UNet, x: np.ndarray, target_y: torch.Tensor, groundtruth_y: torch.Tensor):
    x_adv = np.copy(x)
    step_size = 0.00001
    epsilon = 0.000001

    iterations = 50

    for i in range(iterations):
        x_adv = torch.Tensor(x_adv).cuda()
        x_adv.requires_grad = True
        out = model(x_adv).cuda()

        #adversarial target
        adv_target = 1 - groundtruth_y

        loss = BCEWithLogitsLoss()(out, adv_target)
        loss.backward()
        grad = x_adv.grad
        grad = grad.cpu().detach().numpy()
        x_adv = x_adv.cpu().detach().numpy()
        x_adv = np.add(x_adv, step_size * np.sign(grad))
        x_adv = np.clip(x_adv, x - epsilon, x + epsilon)
        x_adv = np.clip(x_adv, 0, 1)

    return x_adv


def attack_and_display(model: UNet, x: torch.Tensor, y: np.ndarray, target_y: torch.Tensor, model_out: torch.Tensor):
    plt.close(plt.figure())
    attack_out = attack(model, x[:1], target_y[:1], y[:1])
    disp_image_single(x[0], 'Real Input')
    disp_image_single(attack_out[0], 'Attacked Input')

    # Compute model output on adversarial input

    adv_out = model(torch.Tensor(attack_out).cuda())

    disp_edge_single(adv_out[0].cpu().detach().numpy(), 'Adversarial Output')
    disp_edge_single(model_out[0].cpu().detach().numpy(), 'Model Output')
    disp_edge_single(target_y[0].cpu().detach().numpy(), 'Target Output')
    disp_edge_single(y[0].cpu().detach().numpy(), 'Groundtruth Output')


def train():
    model = UNet(num_classes=1, depth=1, start_filts=16, merge_mode='concat').cuda()

    data_dir: str = "../BSR"

    dset = BSDSDataset(False, data_dir, (320, 320))

    test_dset = BSDSDataset(True, data_dir)
    mb_size = 4
    loader = DataLoader(dset, batch_size=mb_size)

    optimizer = Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8)

    if args.load:
        load_checkpoint(checkpoint_dir=args.checkpoint_dir, mod=model, optim=optimizer)

    iteration = 0

    while True:
        for x, y in loader:

            x = x.float().cuda()
            y = y.float().cuda()

            if iteration == 0:
                target_y = y[:1]

            out = model(x).cuda()

            optimizer.zero_grad()
            loss = BCEWithLogitsLoss()(out, y)
            loss.backward()

            optimizer.step()

            if iteration % 1000 == 999:
                save_checkpoint(os.path.join(args.checkpoint_dir, "checkpoint_{}".format(iteration)), model, optimizer)

            iteration += 1

            if iteration % 1000 == 250:
                attack_and_display(model, x, y, target_y, out)
            # if iteration % 1000 == 10:
            #     print('iteration {}; loss {}'.format(iteration, loss.cpu().detach().numpy()))
            #     evaluate(model, test_dset, iteration)


if __name__ == "__main__":
    train()
