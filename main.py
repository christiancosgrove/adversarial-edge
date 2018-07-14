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
    model = UNet(3, depth=5, merge_mode='concat')
    data_dir = "./data/BSR"

    dset = BSDSDataset(data_dir)
    mb_size = 4
    loader = DataLoader(dset, batch_size=mb_size)

    for x, y in loader:

        out_cpu = x.detach().numpy().transpose(0, 2, 3, 1).reshape(mb_size, 320, 320, 3)
        out_cpu_y = y.detach().numpy().reshape(mb_size, 320, 320)

        plt.imshow(out_cpu[0])
        plt.show()

        plt.imshow(out_cpu_y[0])
        plt.show()

    # for image in dataset.train_sample_names:
    #     x = dataset.read_image(image)
    #     y = dataset.load_boundaries(os.path.join(groundtruth_dir, image))
    #
    #     x = imresize(x, (320, 320)).reshape((1, 320, 320, 3))
    #     x = x.transpose(0, 3, 1, 2)
    #
    #     tensor = Variable(torch.FloatTensor(np.random.random((1, 3, 320, 320))))
    #     out = model(tensor)
    #     # loss = torch.sum(out)
    #     # loss.backward()
    #
    #     out_cpu = out.detach().numpy().transpose(0, 2, 3, 1).reshape(320, 320, 3)
    #
    #     plt.imshow(out_cpu)
    #     plt.show()
    #     plt.imshow(y[0])
    #     plt.show()

