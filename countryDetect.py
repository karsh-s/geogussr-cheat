import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os
import numpy as np

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
    device = torch.device("mps")
print(device)

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_COUNTRIES_DIR = os.path.join(BASE_DIR, "trainCountries")
TEST_COUNTRIES_DIR = os.path.join(BASE_DIR, "testCountries")

train_dataset = ImageFolder(
    root = TRAIN_COUNTRIES_DIR,
    transform = transform
)

test_dataset = ImageFolder(
    root = TEST_COUNTRIES_DIR,
    transform = transform
)

train_loader = DataLoader(train_dataset, batch_size = 4, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = 4, shuffle = False)

classes = ("Albania","Andorra","Austria","Belgium","Bulgaria","Croatia",
    "Czech Republic","Denmark","Estonia","Finland","France",
    "Germany","Greece","Hungary","Iceland","Ireland","Italy",
    "Latvia","Liechtenstein","Lithuania","Luxembourg","Malta",
    "Monaco","Montenegro","Netherlands","North Macedonia",
    "Norway","Poland","Portugal","Romania","Serbia",
    "Slovakia","Slovenia","Spain","Sweden","Switzerland",
    "United Kingdom","Bangladesh","Cambodia","Hong Kong","India","Indonesia",
    "Israel","Japan","Kyrgyzstan","Malaysia",
    "Mongolia","Nepal","Pakistan","Philippines","Singapore",
    "South Korea","Sri Lanka","Taiwan","Thailand","Vietnam","Bhutan")

class ConvNeuralNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 64, 3, padding = 1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding = 1)

        self.pool = nn.MaxPool2d(2, stride = 2)
        self.gap = nn.AdaptiveAvgPool2d((1,1))

        self.fc1 = nn.Linear(128 * 1 * 1, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim = 1)
        return x

num_classes = len(train_dataset.classes)
net = ConvNeuralNet(num_classes)
net.to(device)

loss_function = nn.NLLLoss()
optimizer = optim.Adam(net.parameters(), lr = 0.001)

epochs = 10
for epoch in range(epochs):

    running_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = loss_function(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 2000 == 1999:
            print(f'[{epoch + 1}/{epochs}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

def view_classification(image, probabilities):
    probabilities = probabilities.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 9))

    image = image.permute(1,2,0)
    denormalized_image = image / 2 + 0.5
    ax1.imshow(denormalized_image)
    ax1.axis('off')
    k = len(train_dataset.classes)
    ax2.barh(np.arange(k), probabilities[:k])
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(k))
    ax2.set_yticklabels(train_dataset.classes)
    ax2.set_title('Prob')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()

images, _ = next(iter(test_loader))

image = images[0]
batched_image = image.unsqueeze(0).to(device)
with torch.no_grad():
    log_probabilities = net(batched_image)

probabilities = torch.exp(log_probabilities).squeeze().cpu()
view_classification(image, probabilities)

correct = 0
total = 0

with torch.no_grad():
    for data in test_loader:
        images, labels = data[0].to(device), data[1].to(device)

        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network: {100 * correct // total}%')
plt.show()