from PIL import Image
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
import torchvision.transforms.functional as TF
class Custom_ImageFolder(ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target, index
