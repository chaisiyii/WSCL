"""
Data Augmentation For Training Feature Extractor and Classifier.
"""
import random
import torchvision.transforms.functional as F

from torchvision import transforms
from typing import Sequence
from .gaussian_blur import GaussianBlur


class ChoiceRotateTransform:
    def __init__(self, angles: Sequence[int]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return F.rotate(x, angle)


data_transform = {
    "train": transforms.Compose(
        [transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    ),
    "val": transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    ),
    "test": transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )
}

data_transform_simclr = transforms.Compose([
    ChoiceRotateTransform([0, 90, 180, 270]),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([GaussianBlur(kernel_size=int(0.1 * 64))], p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class TwoTransform:
    """
        Create two views of the same image.
    """
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]