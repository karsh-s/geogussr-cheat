import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import random
import numpy as np

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 1
IMG_SIZE = 224
NUM_WORKERS = 4

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "trainEurope")

# Load CSV
csv_path = os.path.join(image_dir, 'inputs.csv')
df = pd.read_csv(csv_path, sep=";").dropna(axis=1, how="all")
print(f"Loaded {len(df)} images")

# Create label mapping
countries = sorted(df['country'].unique())
country_to_idx = {country: idx for idx, country in enumerate(countries)}
idx_to_country = {idx: country for country, idx in country_to_idx.items()}

print(f"\nFound {len(countries)} countries:")
print(countries[:10], "..." if len(countries) > 10 else "")

# Add numeric labels
df['label'] = df['country'].map(country_to_idx)

# Train/val split
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)  # Shuffle
n = int(0.85 * len(df))
df_train = df.iloc[:n]
df_val = df.iloc[n:]

print(f"\nTraining samples: {len(df_train)}")
print(f"Validation samples: {len(df_val)}")

# Dataset class
class CountryDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get image path
        img_file = self.df.loc[idx, 'file']
        if img_file.startswith('/'):
            img_file = img_file[1:]
        img_path = os.path.join(self.image_dir, img_file)
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get label
        label = self.df.loc[idx, 'label']
        
        return image, label

# Data augmentation for training
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# No augmentation for validation
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create datasets
train_dataset = CountryDataset(df_train, image_dir, transform=train_transform)
val_dataset = CountryDataset(df_val, image_dir, transform=val_transform)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                         num_workers=0, drop_last=True)  # num_workers=0 for compatibility
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                       num_workers=0)

# Create model - ResNet50 pretrained on ImageNet
print("\nLoading pretrained ResNet50...")
model = models.resnet50(pretrained=True)

# Freeze early layers (optional - comment out to train all layers)
for param in list(model.parameters())[:-20]:
    param.requires_grad = False

# Replace final layer for our number of countries
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(countries))

model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2)

# Training function
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    loop = tqdm(loader, desc="Training")
    for images, labels in loop:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        loop.set_postfix(loss=loss.item(), acc=100.*correct/total)
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

# Validation function
@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    loop = tqdm(loader, desc="Validation")
    for images, labels in loop:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        loop.set_postfix(loss=loss.item(), acc=100.*correct/total)
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

# Training loop
print("\n" + "="*60)
print("Starting training...")
print("="*60)

best_acc = 0.0
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc = validate(model, val_loader, criterion)
    
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    
    # Learning rate scheduling
    scheduler.step(val_acc)
    
    # Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'country_to_idx': country_to_idx,
            'idx_to_country': idx_to_country,
        }, os.path.join(BASE_DIR, 'cnn_model_best.pth'))
        print(f"✓ Saved new best model with accuracy: {val_acc:.2f}%")

# Save final model
torch.save({
    'epoch': EPOCHS,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'val_acc': val_acc,
    'country_to_idx': country_to_idx,
    'idx_to_country': idx_to_country,
}, os.path.join(BASE_DIR, 'cnn_model_final.pth'))

print("\n" + "="*60)
print("Training complete!")
print(f"Best validation accuracy: {best_acc:.2f}%")
print("="*60)
print(f"\n✓ Best model saved to: cnn_model_best.pth")
print(f"✓ Final model saved to: cnn_model_final.pth")
print("\nYou can now use test_cnn.py to test the model!")
