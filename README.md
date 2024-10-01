# LoRA-From-Scratch

This project implements a Low-Rank Adaptation (LoRA) technique from scratch for fine-tuning a neural network on the MNIST dataset. It allows for efficient adaptation of a pre-trained model to specific digits.

## Features

- Custom LoRA implementation
- MNIST classifier with fine-tuning capabilities
- Digit-specific adaptation
- Performance comparison between original and fine-tuned models

## Files

- `LoRA.py`: Core implementation of the LoRA technique
- `mnist_model.py`: MNIST model definition, training, and evaluation procedures

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/Pragateeshwaran/LoRA-From-Scratch.git
   cd LoRA-From-Scratch
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the `mnist_model.py` script:

```
python mnist_model.py
```

This script will:
1. Train an initial MNIST classifier
2. Prompt you to choose a digit for fine-tuning
3. Apply LoRA and fine-tune the model for the chosen digit
4. Display the results

## Implementation Details

### LoRA.py

- `LoRA_scratch`: Custom PyTorch module implementing the LoRA layer
- `linear_layer_parameterization`: Function to create LoRA parameterization for linear layers
- `apply_LoRA`: Function to apply LoRA to a given model

### mnist_model.py

- `Mnist`: Simple neural network for MNIST classification
- `train_model`: Method to train the model
- `test`: Method to evaluate the model
- `prepare_data`: Function to prepare the MNIST dataset
- `get_digit_subset`: Utility function to create a subset of data for a specific digit

## Customization

Modify the following parameters in `mnist_model.py`:
- Batch sizes
- Learning rates
- Number of epochs
- LoRA rank (in the `apply_LoRA` function call)

## Results

The script will output:
- Training progress
- Test accuracy for the original model
- Test accuracy after LoRA fine-tuning
- Accuracy for the specific digit chosen for fine-tuning

## Contact

For any questions or feedback, please reach out:

Email: geniuspekka1808@gmail.com

## Acknowledgements

- MNIST Dataset
- PyTorch
- LoRA: Low-Rank Adaptation of Large Language Models

---
 
