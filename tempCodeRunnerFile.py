import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

BATCH_SIZE = 64
EPOCHS = 15
LEARINING_RATE = 0.001
IMAGE_SIZE = (48, 48)
MODEL_FILENAME = "emotion_cnn_model.pth"

"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")
"""

device = torch.device("cpu")
print("Using CPU for training.")

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
train_dataset = datasets.load_data_placeholder if False else datasets.ImageFolder(root='./train', transform=train_transform)
test_dataset = datasets.ImageFolder(root="./test", transform=test_transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

class_names = train_dataset.classes
print(f"Detected emotion classes: {class_names}")

class EmotionCNN(nn.Module):
    def __init__(self, num_classes=len(class_names)):
        super(EmotionCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(128 * 6 * 6, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        
        x = x.view(x.size(0), -1)
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = EmotionCNN().to(device)
if os.path.exists(MODEL_FILENAME):
    model.load_state_dict(torch.load(MODEL_FILENAME, map_location=device))
    print(f"Loaded existing weights from {MODEL_FILENAME}. Continuing training from previous state...")
else:
    print("No existing weights found. Training from scratch.")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARINING_RATE)

print("Starting training process...")
for epoch in range(EPOCHS):
    # --- TRAINING ---
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total_train += labels.size(0)
        correct_train += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100.0 * correct_train / total_train

    # --- TESTING / VALIDATION ---
    model.eval()
    test_loss = 0.0
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total_test += labels.size(0)
            correct_test += predicted.eq(labels).sum().item()

    test_epoch_loss = test_loss / len(test_loader)
    test_epoch_acc = 100.0 * correct_test / total_test

    print(f"Epoch [{epoch+1}/{EPOCHS}] | "
          f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}% || "
          f"Test Loss: {test_epoch_loss:.4f} | Test Acc: {test_epoch_acc:.2f}%")

torch.save(model.state_dict(), MODEL_FILENAME)
print(f"Model successfully trained and saved to {MODEL_FILENAME}")