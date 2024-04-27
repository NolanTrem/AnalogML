import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import copy


resistor_values = np.array([1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0, 3.3, 3.6, 3.9, 4.3, 4.7, 5.1, 5.6, 6.2, 6.8, 7.5, 8.2, 9.1]) * 1e3
conductance_values = 1 / resistor_values
normalized_conductance_values = (conductance_values - conductance_values.min()) / (conductance_values.max() - conductance_values.min())

def normalize_and_quantize_tensor(tensor):
    """
    Normalizes the tensor to the range [0, 1] and quantizes it to the nearest E24 resistor value.
    """
    if tensor.nelement() == 0:
        return tensor

    tensor = torch.clamp(tensor, 0, 1)
    quantized_tensor = torch.zeros_like(tensor)

    for value in normalized_conductance_values:
        mask = torch.abs(tensor - value) == torch.min(torch.abs(tensor - value))
        quantized_tensor[mask] = value

    return quantized_tensor

def reapply_quantization_and_normalization(model):
    for m in model.modules():
        if hasattr(m, "weight"):
            m.weight.data = normalize_weights(m.weight.data)
        if hasattr(m, "bias") and m.bias is not None:
            m.bias.data = normalize_and_quantize_tensor(m.bias.data)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        nn.init.zeros_(m.bias)

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, dropout_rate=0.5):
        super(MLP, self).__init__()
        layers = []
        in_features = input_size
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(in_features, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.LeakyReLU(),
                nn.Dropout(dropout_rate)
            ])
            in_features = hidden_size
        layers.append(nn.Linear(in_features, num_classes))
        self.layers = nn.Sequential(*layers)
        self.apply(init_weights)

    def forward(self, x):
        return self.layers(x)
    # def __init__(self, input_size, hidden_sizes, num_classes, num_decimals=None):
    #     super(MLP, self).__init__()
    #     self.num_decimals = num_decimals
    #     layers = []
    #     in_features = input_size
    #     for hidden_size in hidden_sizes:
    #         layers.extend([nn.Linear(in_features, hidden_size), nn.ReLU()])
    #         in_features = hidden_size
    #     layers.append(nn.Linear(in_features, num_classes))
    #     self.layers = nn.Sequential(*layers)

    #     if num_decimals is not None:
    #         self.apply(self.quantize_weights)

    # def quantize_weights(self, m):
    #     if hasattr(m, "weight"):
    #         m.weight.data = normalize_and_quantize_tensor(m.weight.data)
    #     if hasattr(m, "bias") and m.bias is not None:
    #         m.bias.data = normalize_and_quantize_tensor(m.bias.data)

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
    
def normalize_weights(weights, lut_assignment="layer"):
    if lut_assignment == "neuron":
        max_val = torch.max(torch.abs(weights), dim=1, keepdim=True)[0]
    elif lut_assignment == "layer":
        max_val = torch.max(torch.abs(weights))
    elif lut_assignment == "slice":
        max_val = torch.max(torch.abs(weights), dim=0, keepdim=True)[0]
    else:
        raise ValueError(f"Invalid LUT assignment: {lut_assignment}")

    return weights / max_val


def train(model, criterion, optimizer, train_loader, val_loader, scheduler, epochs=10, restart_interval=100, lut_assignment="layer"):
    model.train()
    train_loss_history = []
    val_loss_history = []
    val_accuracy_history = []
    swa_model = AveragedModel(model)
    val_loss = 0

    # Step 1: Train with continuous weights
    for epoch in range(epochs):
        total_loss = 0

        if (epoch + 1) % restart_interval == 0:
            model.apply(init_weights)

        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

            scheduler.step(val_loss)

        epoch_loss = total_loss / len(train_loader)
        train_loss_history.append(epoch_loss)

        model.eval()
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

        swa_model.update_parameters(model)
    
    # Step 2: Normalize weights based on LUT assignment
    for m in model.modules():
        if hasattr(m, "weight"):
            m.weight.data = normalize_weights(m.weight.data, lut_assignment)

    # Step 3: Quantize weights to powers-of-two or sums of powers-of-two
    for m in model.modules():
        if hasattr(m, "weight"):
            m.weight.data = optimizer.quantize_weights(m.weight.data)

    # Step 4: Fine-tune with quantized weights
    for epoch in range(epochs):
        total_loss = 0

        if (epoch + 1) % restart_interval == 0:
            model.apply(init_weights)

        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            reapply_quantization_and_normalization(model)
            total_loss += loss.item()

            scheduler.step(val_loss)

        epoch_loss = total_loss / len(train_loader)
        train_loss_history.append(epoch_loss)

        model.eval()
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

        swa_model.update_parameters(model)

    swa_model.eval()

    return swa_model, train_loss_history, val_loss_history, val_accuracy_history


def predict(model, data):
    model.eval()
    with torch.no_grad():
        output = model(data)
        return output.argmax(dim=1)


class AveragedModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.n_averaged = 0
        self.averaged_model = copy.deepcopy(model)

    def forward(self, x):
        return self.averaged_model(x)

    def update_parameters(self, model):
        self.n_averaged += 1
        for p_swa, p_model in zip(self.averaged_model.parameters(), model.parameters()):
            p_swa.data.mul_(self.n_averaged).add_(p_model.data).div_(self.n_averaged + 1)