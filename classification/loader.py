import torchvision
from torchvision import transforms
import torch
import os

def get_loader(config,train=True):
    assert config.data.lower() == 'cifar10'
    if not os.path.exists("./data"):
        os.mkdir("./data")
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.RandomCrop(227),
                                    transforms.RandomHorizontalFlip(0.5),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])
    dset = torchvision.datasets.CIFAR10(root='data',train=train, transform = transform, download = True)
    loader = torch.utils.data.DataLoader(dset, batch_size = config.batch_size, shuffle = True)
    return loader
