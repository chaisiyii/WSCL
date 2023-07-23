"""
DataLoader For Feature Extractor and Classifier.
"""
import torch
from torchvision.datasets import ImageFolder
from .augmentation import TwoTransform, data_transform, data_transform_simclr


def load_data_simclr(batch_size=16, num_workers=8):
    path = ""  # your path
    dataset = ImageFolder(path, transform=TwoTransform(data_transform_simclr))
    
    sampler = None
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=(sampler is None),
        num_workers=num_workers, pin_memory=True, sampler=sampler
    )
    return train_loader


def load_data_simclr_val(batch_size=16, num_workers=8):
    path = ""  # your path
    dataset = ImageFolder(path, transform=TwoTransform(data_transform_simclr))
    
    sampler = None
    val_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=(sampler is None),
        num_workers=num_workers, pin_memory=True, sampler=sampler
    )
    return val_loader


def load_train_data(batch_size=16, num_workers=8, path=None):
    if path is None:
        path = ""  # your path
    dataset = ImageFolder(path, transform=data_transform['train'])

    sampler = None
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=(sampler is None),
        num_workers=num_workers, pin_memory=True, sampler=sampler
    )
    return train_loader


def load_val_data(batch_size=16, num_workers=8):
    path = ""  # your path
    dataset = ImageFolder(path, transform=data_transform['val'])
    val_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return val_loader


def load_test_data(batch_size=16, num_workers=8):
    path = ""  # your path
    dataset = ImageFolder(path, transform=data_transform['test'])
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return test_loader
