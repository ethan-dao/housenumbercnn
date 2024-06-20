import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import SVHN
from torchvision.transforms import ToTensor, Normalize, Compose
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter

# Define transformations with normalization
transform = Compose([
    ToTensor(),
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize images; optional but increases accuracy slightly
])

# Download our data with transformations
training_dataset = SVHN(root='data/', split='train', download=True, transform=transform)
validation_size = 12500  # Adjust if needed
training_size = len(training_dataset) - validation_size
training_data, validation_data = random_split(training_dataset, [training_size, validation_size])
training_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
validation_dataloader = DataLoader(validation_data, batch_size=64, shuffle=True)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

class CNNModel(nn.Module):
    def __init__(self): # Hidden layers
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x): # Forward (pooling)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
model = CNNModel().to(device)

loss_fn = nn.CrossEntropyLoss()
learning_rate = 1e-3
epochs = 5

# Initialize TensorBoard SummaryWriter
writer = SummaryWriter('runs/SVHN_CNN')


def fit(epochs, learning_rate, model, training_dataloader, validation_dataloader, optimizer_func=torch.optim.Adam):
    optimizer = optimizer_func(model.parameters(), learning_rate)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for X, y in training_dataloader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for X, y in validation_dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                val_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        
        val_loss /= len(validation_dataloader)
        correct /= len(validation_dataloader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {running_loss/len(training_dataloader)}, Validation Loss: {val_loss}, Accuracy: {100*correct:.2f}%")

        # Log metrics to TensorBoard
        writer.add_scalar('Loss/Train', running_loss / len(training_dataloader), epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Accuracy/Validation', 100 * correct, epoch)
            

def evaluate(dataloader, model, loss_fn):  # Testing loop
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    # Log test metrics to TensorBoard
    writer.add_scalar('Loss/Test', test_loss, 0)
    writer.add_scalar('Accuracy/Test', 100 * correct, 0)

fit(epochs, learning_rate, model, training_dataloader, validation_dataloader)

# Evaluate model on test set
testing_dataset = SVHN(root='data/', split='test', download=True, transform=transform)
testing_dataloader = DataLoader(testing_dataset, batch_size=64, shuffle=False)
evaluate(testing_dataloader, model, loss_fn)

# Close TensorBoard
writer.close()
