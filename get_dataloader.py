from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def get_loader(directory, transform):
    dataset = ImageFolder(directory, transform=transform)
    return DataLoader(dataset, batch_size=32, shuffle=True)
