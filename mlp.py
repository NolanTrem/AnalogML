import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_and_quantize_tensor(tensor, num_decimals):
    """
    Normalizes the tensor to the range [0, 1] and quantizes it to a specified number of decimal places.
    """
    if tensor.nelement() == 0:
        return tensor

    # Quantize
    scale = 10.0**num_decimals
    tensor = torch.round(tensor * scale) / scale

    # And normalize
    min_val = torch.min(tensor)
    tensor -= min_val
    max_val = torch.max(tensor)
    if max_val > 0:
        tensor /= max_val

    return tensor


def reapply_quantization_and_normalization(model):
    for m in model.modules():
        if hasattr(m, "weight"):
            m.weight.data = normalize_and_quantize_tensor(
                m.weight.data, model.num_decimals
            )
        if hasattr(m, "bias") and m.bias is not None:
            m.bias.data = normalize_and_quantize_tensor(m.bias.data, model.num_decimals)


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, num_decimals=None):
        super(MLP, self).__init__()
        self.num_decimals = num_decimals
        layers = []
        in_features = input_size
        for hidden_size in hidden_sizes:
            layers.extend([nn.Linear(in_features, hidden_size), nn.ReLU()])
            in_features = hidden_size
        layers.append(nn.Linear(in_features, num_classes))
        self.layers = nn.Sequential(*layers)

        if num_decimals is not None:
            self.apply(self.quantize_weights)

    def quantize_weights(self, m):
        if hasattr(m, "weight"):
            m.weight.data = normalize_and_quantize_tensor(
                m.weight.data, self.num_decimals
            )
        if hasattr(m, "bias") and m.bias is not None:
            m.bias.data = normalize_and_quantize_tensor(m.bias.data, self.num_decimals)

    def forward(self, x):
        """
        Forward pass through the model.
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def save_weights_biases_to_csv(self, num_decimals=4, prefix="layer"):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                weights = layer.weight.data.cpu().numpy()
                biases = layer.bias.data.cpu().numpy()
                weights_file = f"{prefix}_{i}_weights.csv"
                biases_file = f"{prefix}_{i}_biases.csv"
                # Use num_decimals to format the output
                np.savetxt(
                    weights_file, weights, delimiter=",", fmt=f"%.{num_decimals}f"
                )
                np.savetxt(biases_file, biases, delimiter=",", fmt=f"%.{num_decimals}f")
                print(f"Saved {weights_file} and {biases_file}")


def train(model, criterion, optimizer, train_loader, val_loader, epochs=10):
    model.train()
    train_loss_history = []
    val_loss_history = []
    val_accuracy_history = []

    for epoch in range(epochs):
        total_loss = 0

        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            reapply_quantization_and_normalization(model)
            total_loss += loss.item()

        epoch_loss = total_loss / len(train_loader)
        train_loss_history.append(epoch_loss)

        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        val_loss /= len(val_loader)
        val_loss_history.append(val_loss)

        accuracy = 100.0 * correct / len(val_loader.dataset)
        val_accuracy_history.append(accuracy)

        print(
            f"Epoch {epoch+1}, Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {accuracy:.2f}%"
        )

    return train_loss_history, val_loss_history, val_accuracy_history


def predict(model, data):
    model.eval()
    with torch.no_grad():
        output = model(data)
        return output.argmax(dim=1)
