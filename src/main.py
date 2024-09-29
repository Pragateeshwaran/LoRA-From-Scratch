# Import dependencies
import torch
import numpy as np 
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets 
import time

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"This model runs on {device}")

# Define class names for FashionMNIST
class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

# Define the Model
class Model(nn.Module):
    """
    Model that contains 3 Linear layers.
    """
    def __init__(self):
        super().__init__()
        # Layers
        self.Layer1 = nn.Linear(28*28, 10000)
        self.Layer2 = nn.Linear(10000, 10000)
        self.Layer3 = nn.Linear(10000, 10)

        # Activation Functions
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)   

    def forward(self, X):
        X = X.view(-1, 28*28)
        
        res = self.Layer1(X)
        res = self.relu(res)

        res = self.Layer2(res)
        res = self.tanh(res)

        res = self.Layer3(res)
        res = self.softmax(res)

        return res 

# Prepare Data
def Prepare_Data(batch_size_train=64, batch_size_test=1024):
    """
    Prepares the FashionMNIST dataset loaders.

    Args:
        batch_size_train (int): Batch size for training.
        batch_size_test (int): Batch size for testing.

    Returns:
        tuple: Training and testing DataLoaders.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Training dataset and loader
    mnist_trainset = datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=transform
    )
    train_loader = DataLoader(mnist_trainset, batch_size=batch_size_train, shuffle=True)
    
    # Testing dataset and loader
    mnist_testset = datasets.FashionMNIST(
        root='./data', train=False, download=True, transform=transform
    )
    test_loader = DataLoader(mnist_testset, batch_size=batch_size_test, shuffle=False)
    
    return train_loader, test_loader

# Initialize Model
model = Model().to(device=device)
print(model)
 
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training Loop
def train(model, device, train_loader, optimizer, criterion, epoch):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The neural network model.
        device (torch.device): Device to run the training on.
        train_loader (DataLoader): DataLoader for training data.
        optimizer (torch.optim.Optimizer): Optimizer.
        criterion (nn.Module): Loss function.
        epoch (int): Current epoch number.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()            
        outputs = model(data)           
        loss = criterion(outputs, target)   
        loss.backward()                
        optimizer.step()                 
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        if (batch_idx + 1) % 100 == 0:
            print(f'Epoch [{epoch}/100], Step [{batch_idx+1}/{len(train_loader)}], '
                  f'Loss: {running_loss / (batch_idx+1):.4f}, '
                  f'Accuracy: {100 * correct / total:.2f}%')
    
    epoch_time = time.time() - start_time
    print(f'Epoch [{epoch}/100] completed in {epoch_time:.2f} seconds. '
          f'Average Loss: {running_loss / len(train_loader):.4f}, '
          f'Accuracy: {100 * correct / total:.2f}%')

# Testing Loop with Per-Class Accuracy
def test(model, device, test_loader, criterion, class_names):
    """
    Evaluates the model on the test dataset and reports per-class accuracy.

    Args:
        model (nn.Module): The neural network model.
        device (torch.device): Device to run the evaluation on.
        test_loader (DataLoader): DataLoader for test data.
        criterion (nn.Module): Loss function.
        class_names (list): List of class names corresponding to labels.

    Returns:
        float: Test accuracy in percentage.
    """
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    # Initialize lists to keep track of correct predictions and total samples per class
    class_correct = list(0. for _ in range(len(class_names)))
    class_total = list(0. for _ in range(len(class_names)))

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            c = (predicted == target).squeeze()

            # Iterate through the batch
            for i in range(len(target)):
                label = target[i]
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1

            correct += (predicted == target).sum().item()
            total += target.size(0)

    average_loss = test_loss / len(test_loader)
    accuracy = 100 * correct / total
    print(f'Test Loss: {average_loss:.4f}, Test Accuracy: {accuracy:.2f}%\n')

    # Print per-class accuracy
    for i in range(len(class_names)):
        if class_total[i] > 0:
            print(f'Class: {class_names[i]:15s} - Correct: {int(class_correct[i])}/{int(class_total[i])} '
                  f'({100 * class_correct[i] / class_total[i]:.2f}%)')
        else:
            print(f'Class: {class_names[i]:15s} - No samples.')

    print("\n")  # Add an extra newline for better readability
    return accuracy

# Main Execution
if __name__ == "__main__":
    train_loader, test_loader = Prepare_Data()
    
    best_accuracy = 0.0  # To keep track of the best accuracy
    num_epochs = 1  # Set to 100 as needed

    for epoch in range(1, num_epochs + 1):
        train(model, device, train_loader, optimizer, criterion, epoch)
        accuracy = test(model, device, test_loader, criterion, class_names)
        
        # Save the model checkpoint if it's the best so far
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'Best model saved with accuracy: {best_accuracy:.2f}%\n')
    
    print(f'Training completed. Best Test Accuracy: {best_accuracy:.2f}%')
