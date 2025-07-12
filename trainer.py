import os
import torch
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from core.datasets import get_dataset
from core.model import prepare_model

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = prepare_model().to(device)

    train_ds, val_ds = get_dataset()
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(5):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
    os.makedirs("weights", exist_ok=True)
    torch.save(model.state_dict(), "weights/resnet18.pth")

if __name__ == "__main__":
    train_model()