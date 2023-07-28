import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder


class DeviceDataLoader:
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)



def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def preprocess():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation(50),
        transforms.Resize(size=(128, 128), antialias=True)
    ])

    torch.manual_seed(0)
    batch_size = 8
    training_data = DataLoader(ImageFolder("train", transform=transform), batch_size, shuffle=True, num_workers=10,
                               pin_memory=True)
    validation_data = DataLoader(ImageFolder("valid", transform=transform), batch_size, num_workers=10,
                                 pin_memory=True)



    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("using: " + str(device))

    return DeviceDataLoader(training_data, device), DeviceDataLoader(validation_data, device)