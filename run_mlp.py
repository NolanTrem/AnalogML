import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from ShapeMatrixGenerator import ShapeMatrixGenerator
# Not used in this example, but another option for simple matrices
from TextMatrixGenerator import TextMatrixGenerator

from mlp import MLP, predict, train

generator = ShapeMatrixGenerator(size=5)
shapes = ['circle', 'square', 'triangle', 'diamond']
shape_matrices = {
    'circle': generator.draw_circle(),
    'square': generator.draw_square(),
    'triangle': generator.draw_triangle(),
    'diamond': generator.draw_diamond(),
}

X = []
y = []

for index, shape in enumerate(shapes):
    # Print the shape
    print(f"Shape: {shape}")
    matrix = shape_matrices[shape]
    # print the matrix
    print(matrix)
    vector = matrix.flatten()
    # Print the flattened matrix
    print(vector)
    X.append(vector)
    y.append(index)

X = np.array(X)
y = np.array(y)

X_tensor = torch.Tensor(X)
y_tensor = torch.Tensor(y).long()

train_dataset = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

val_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)

input_size = X_tensor.size(1)
num_classes = 4

model = MLP(
    input_size=input_size,
    hidden_sizes=[4],
    num_classes=num_classes,
)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

swa_model, train_loss_history, val_loss_history, val_accuracy_history = train(
    model, criterion, optimizer, train_loader, val_loader, epochs=10000, quantization_warmup=500, quantization_steps=500
)

# Save the model state dictionary
model_path = "trained_model.pth"
torch.save(model.state_dict(), model_path)

# Save the weights and biases to CSV
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()
model.save_weights_biases_to_csv(num_decimals=4)

# Plot the training and validation loss
plt.plot(train_loss_history, label="Training Loss")
plt.plot(val_loss_history, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Over Time")
plt.legend()
plt.show()

# Plot the validation accuracy
plt.plot(val_accuracy_history, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Validation Accuracy Over Time")
plt.legend()
plt.show()
