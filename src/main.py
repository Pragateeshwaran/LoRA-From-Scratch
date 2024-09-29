# main.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from model import SimpleNN, apply_lora, Enable_lora
import time



device = "cuda" if torch.cuda.is_available() else "cpu"

def count_parameters(model): 
    """
    Counts the total number of trainable parameters in the model.

    Args:
        model (nn.Module): The neural network model.

    Returns:
        int: Total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def Prepare_Data(batch_size_train=64, batch_size_test=1024, fine_tune_digit=None):
    """
    Prepares the MNIST dataset loaders.

    Args:
        batch_size_train (int): Batch size for training.
        batch_size_test (int): Batch size for testing.
        fine_tune_digit (int, optional): If specified, filters the training data to include only this digit.

    Returns:
        tuple: Training and testing DataLoaders.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize with MNIST mean and std
    ])
    
    # Training dataset and loader
    mnist_trainset = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    
    if fine_tune_digit is not None:
        # Filter the dataset to include only the specified digit
        include_indices = mnist_trainset.targets == fine_tune_digit
        mnist_trainset.data = mnist_trainset.data[include_indices]
        mnist_trainset.targets = mnist_trainset.targets[include_indices]
        print(f"Training on digit: {fine_tune_digit}")
    
    train_loader = DataLoader(mnist_trainset, batch_size=batch_size_train, shuffle=True)
    
    # Testing dataset and loader
    mnist_testset = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    test_loader = DataLoader(mnist_testset, batch_size=batch_size_test, shuffle=False)
    
    return train_loader, test_loader

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
        
        # Count correct predictions
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        if (batch_idx + 1) % 100 == 0:  # Print every 100 batches
            print(f'Epoch [{epoch}/100], Step [{batch_idx + 1}/{len(train_loader)}], '
                  f'Loss: {running_loss / (batch_idx + 1):.4f}, '
                  f'Accuracy: {100 * correct / total:.2f}%, '
                  f'Correct Predictions: {correct}/{total}')
    epoch_time = time.time() - start_time
    print(f'Epoch [{epoch}/100] completed in {epoch_time:.2f} seconds. '
          f'Average Loss: {running_loss / len(train_loader):.4f}, '
          f'Accuracy: {100 * correct / total:.2f}%\n')

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
    class_correct = list(0. for _ in range(len(class_names)))
    class_total = list(0. for _ in range(len(class_names)))
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            for i in range(len(target)):
                label = target[i].item()
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
    print("\n")
    return accuracy

def main():
    # Initialize the model
    model = SimpleNN().to(device)
    
    # Apply LoRA parametrization
    apply_lora(model, rank=4, lora_alpha=2)
    
    # Display parameter counts
    total_parameters_lora = 0
    total_parameters_non_lora = 0
    for index, layer in enumerate([model.layer1, model.layer2, model.layer3]):
        lora = layer.parametrizations["weight"][0]
        total_parameters_lora += lora.A.nelement() + lora.B.nelement()
        total_parameters_non_lora += layer.weight.numel() + (layer.bias.numel() if layer.bias is not None else 0)
        print(f'Layer {index+1}: W: {layer.weight.shape} + B: {layer.bias.shape if layer.bias is not None else "None"} + '
              f'Lora_A: {lora.A.shape} + Lora_B: {lora.B.shape}')
    
    print(f'\nTotal number of parameters (original): {total_parameters_non_lora:,}')
    print(f'Total number of parameters (original + LoRA): {total_parameters_lora + total_parameters_non_lora:,}')
    print(f'Parameters introduced by LoRA: {total_parameters_lora:,}')
    parameters_increment = (total_parameters_lora / total_parameters_non_lora) * 100 if total_parameters_non_lora > 0 else 0
    print(f'Parameters increment: {parameters_increment:.3f}%\n')
    
    # Freeze non-LoRA parameters
    for name, param in model.named_parameters():
        if 'parametrizations' not in name and 'bias' not in name:
            print(f'Freezing non-LoRA parameter {name}')
            param.requires_grad = False
    
    print("\nParameter counts after freezing non-LoRA parameters:")
    print(f'Total trainable parameters: {count_parameters(model):,}\n')
    
    # Prepare Data
    fine_tune_digit = 9  # Change this to fine-tune a different digit
    train_loader, test_loader = Prepare_Data(fine_tune_digit=fine_tune_digit)
    
    # Define optimizer and loss criterion
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Define class_names for MNIST
    class_names = [str(i) for i in range(10)]
    
    # Train and test
    best_accuracy = 0.0  # To keep track of the best accuracy during fine-tuning
    num_epochs = 5  # Set to 100 as needed

    for epoch in range(1, num_epochs + 1):
        Enable_lora(model, True)  # Enable LoRA during training
        # Re-initialize the optimizer to include only trainable parameters (LoRA parameters)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
        train(model, device, train_loader, optimizer, criterion, epoch)
        accuracy = test(model, device, test_loader, criterion, class_names)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), f'best_model_finetuned_digit_{fine_tune_digit}.pth')
            print(f'Best fine-tuned model saved with accuracy: {best_accuracy:.2f}%\n')
    print(f'Fine-Tuning completed. Best Test Accuracy: {best_accuracy:.2f}%')

if __name__ == "__main__":
    main()
