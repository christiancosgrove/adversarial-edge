from torch import Tensor
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from bsds_dataset import BSDSDataset
from bsds_wrapper import BSDSWrapper
from unet import UNet


def get_training_batch(dataset: BSDSWrapper, batch_size: int):
    dataset.load_boundaries()


if __name__ == "__main__":
    """
    testing
    """
    model = UNet(num_classes=1, depth=5, merge_mode='concat').float()
    data_dir = "./data/BSR"

    dset = BSDSDataset(data_dir)
    mb_size = 4
    loader = DataLoader(dset, batch_size=mb_size)

    optimizer = Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)

    for x, y in loader:

        x = x.float()
        y = y.float()

        out = model(x).float()

        optimizer.zero_grad()
        loss = BCEWithLogitsLoss()(out, y)
        loss.backward()

        optimizer.step()

        out_model = out.detach().numpy().transpose(0, 2, 3, 1).reshape(mb_size, 320, 320)
        out_cpu = x.detach().numpy().transpose(0, 2, 3, 1).reshape(mb_size, 320, 320, 3)
        out_cpu_y = y.detach().numpy().reshape(mb_size, 320, 320)

        plt.imshow(out_cpu[0])
        plt.show()
        plt.imshow(out_cpu_y[0])
        plt.show()
        plt.imshow(out_model[0])
        plt.show()
