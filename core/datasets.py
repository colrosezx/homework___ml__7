from torchvision import datasets, transforms

def get_dataset():
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    train = datasets.FakeData(transform=transform)
    val = datasets.FakeData(transform=transform)
    return train, val