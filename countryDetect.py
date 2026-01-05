import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.models as models
import os
import numpy as np

def main():
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
        device = torch.device("mps")
    print(device)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    TRAIN_COUNTRIES_DIR = os.path.join(BASE_DIR, "trainCountries")
    TEST_COUNTRIES_DIR = os.path.join(BASE_DIR, "testCountries")

    weights = models.ResNet18_Weights.DEFAULT
    transform = weights.transforms()

    train_dataset = ImageFolder(root=TRAIN_COUNTRIES_DIR, transform=transform)
    test_dataset  = ImageFolder(root=TEST_COUNTRIES_DIR,  transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    test_loader  = DataLoader(test_dataset,  batch_size=16, shuffle=False, num_workers=2)

    num_classes = len(train_dataset.classes)

    net = models.resnet18(weights=weights)
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    net = net.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-4)

    epochs = 10
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"[{epoch+1}/{epochs}] loss: {running_loss/len(train_loader):.4f}")

    correct = 0
    total = 0
    net.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            predicted = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy of the network: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    main()
