import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import time
from LoRA import apply_LoRA
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEED = 8
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

class Mnist(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(28 * 28, 1000)
        self.layer2 = nn.Linear(1000, 2000)
        self.layer3 = nn.Linear(2000, 2000)
        self.layer4 = nn.Linear(2000, 10)
        self.relu = nn.ReLU()

    def forward(self, X):
        X = X.view(-1, 28 * 28)
        X = self.relu(self.layer1(X))
        X = self.relu(self.layer2(X))
        X = self.relu(self.layer3(X))
        X = self.layer4(X) 
        return X

    def train_model(self, train_loader, criterion, optimizer, device, epochs):
        self.train()
        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            digit_correct = np.zeros(10, dtype=int)
            digit_total = np.zeros(10, dtype=int)

            start_time = time.time()
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                outputs = self(data)
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

                for i in range(10):
                    digit_total[i] += (target == i).sum().item()
                    digit_correct[i] += ((predicted == i) & (target == i)).sum().item()

                if (batch_idx + 1) % 100 == 0:
                    print(f'Epoch [{epoch+1}/{epochs}], Step [{batch_idx + 1}/{len(train_loader)}], '
                          f'Loss: {running_loss / (batch_idx + 1):.4f}, '
                          f'Accuracy: {100 * correct / total:.2f}%')

            epoch_time = time.time() - start_time
            print(f'Epoch [{epoch+1}/{epochs}] completed in {epoch_time:.2f} seconds. '
                  f'Average Loss: {running_loss / len(train_loader):.4f}, '
                  f'Accuracy: {100 * correct / total:.2f}%')
            self.print_digit_correctness(digit_correct, digit_total)

    def test(self, test_loader, criterion, device):
        self.eval()
        test_loss = 0
        correct = 0
        total = 0
        digit_correct = np.zeros(10, dtype=int)
        digit_total = np.zeros(10, dtype=int)

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = self(data)
                loss = criterion(outputs, target)
                test_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

                for i in range(10):
                    digit_total[i] += (target == i).sum().item()
                    digit_correct[i] += ((predicted == i) & (target == i)).sum().item()

        accuracy = 100 * correct / total
        print(f'Test Loss: {test_loss/len(test_loader):.4f}, Test Accuracy: {accuracy:.2f}%')
        self.print_digit_correctness(digit_correct, digit_total)
        return accuracy

    def print_digit_correctness(self, digit_correct, digit_total):
        print("Per-digit correctness:")
        for i in range(10):
            print(f"Digit {i}: Correctly predicted {digit_correct[i]} / Total {digit_total[i]} "
                  f"({100 * digit_correct[i] / digit_total[i]:.2f}%)")

def prepare_data(batch_size_train=64, batch_size_test=1024):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    return mnist_trainset, mnist_testset

def get_digit_subset(dataset, digit):
    indices = (dataset.targets == digit).nonzero().squeeze()
    return Subset(dataset, indices)

if __name__ == "__main__":
    # Initialize and train the original model
    model = Mnist().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    mnist_trainset, mnist_testset = prepare_data()
    train_loader = DataLoader(mnist_trainset, batch_size=64, shuffle=True)
    test_loader = DataLoader(mnist_testset, batch_size=1024, shuffle=False)
    
    print("Training original model:")
    model.train_model(train_loader, criterion, optimizer, device, epochs=1)
    original_accuracy = model.test(test_loader, criterion, device)

    # Ask user which digit to fine-tune
    digit_to_finetune = int(input("Enter the digit you want to fine-tune (0-9): "))

    # Apply LoRA and fine-tune for the specific digit
    print(f"\nApplying LoRA and fine-tuning for digit {digit_to_finetune}:")
    model = apply_LoRA(model, rank=4, enable=True, device=device)
    model = model.to(device)  # Ensure the model is on the correct device after applying LoRA
    lora_optimizer = optim.AdamW(model.parameters(), lr=0.0001)

    # Create a subset of the training data for the specific digit
    digit_trainset = get_digit_subset(mnist_trainset, digit_to_finetune)
    digit_train_loader = DataLoader(digit_trainset, batch_size=64, shuffle=True)

    model.train_model(digit_train_loader, criterion, lora_optimizer, device, epochs=3)
    lora_accuracy = model.test(test_loader, criterion, device)

    print(f"\nOriginal model accuracy: {original_accuracy:.2f}%")
    print(f"LoRA fine-tuned model accuracy: {lora_accuracy:.2f}%")

    # Test accuracy for the specific digit
    digit_testset = get_digit_subset(mnist_testset, digit_to_finetune)
    digit_test_loader = DataLoader(digit_testset, batch_size=1024, shuffle=False)
    digit_accuracy = model.test(digit_test_loader, criterion, device)
    print(f"Accuracy for digit {digit_to_finetune}: {digit_accuracy:.2f}%")
