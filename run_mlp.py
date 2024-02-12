import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from mlp import MLP, predict, train
from text_to_matrix import TextMatrixGenerator

generator = TextMatrixGenerator(size=10)
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
X = []
y = []

for index, letter in enumerate(alphabet):
    matrix = generator.text_to_matrix(letter)
    vector = matrix.flatten()
    X.append(vector)
    y.append(index)

X = np.array(X)
y = np.array(y)

X_tensor = torch.Tensor(X)
y_tensor = torch.Tensor(y).long()

X_train, X_val, y_train, y_val = train_test_split(
    X_tensor, y_tensor, test_size=0.2, random_state=42
)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

val_dataset = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

dataset = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

input_size = X_tensor.size(1)
num_classes = len(alphabet)
model = MLP(
    input_size=input_size,
    hidden_sizes=[8],
    num_classes=num_classes,
    num_decimals=3,
)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_loss_history, val_loss_history, val_accuracy_history = train(
    model, criterion, optimizer, train_loader, val_loader, epochs=2000
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
