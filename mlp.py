import copy

import numpy as np
import torch
import torch.nn as nn

# Standard E24 resistors
resistor_values = (
    np.array(
        [
            1.0,
            1.1,
            1.2,
            1.3,
            1.5,
            1.6,
            1.8,
            2.0,
            2.2,
            2.4,
            2.7,
            3.0,
            3.3,
            3.6,
            3.9,
            4.3,
            4.7,
            5.1,
            5.6,
            6.2,
            6.8,
            7.5,
            8.2,
            9.1,
        ]
    )
    * 1e3
)
conductance_values = 1 / resistor_values
normalized_conductance_values = (conductance_values - conductance_values.min()) / (
    conductance_values.max() - conductance_values.min()
)


def normalize_and_quantize_tensor(tensor, conductance_values):
    """
    Normalize the tensor values between 0 and 1 and quantize them to the nearest normalized conductance value.
    """
    tensor = torch.clamp(tensor, 0, 1)
    quantized_tensor = torch.zeros_like(tensor)
    for value in conductance_values:
        mask = torch.abs(tensor - value) == torch.min(torch.abs(tensor - value))
        quantized_tensor[mask] = value
    return quantized_tensor


def reapply_quantization_and_normalization(model):
    for m in model.modules():
        if hasattr(m, "weight"):
            m.weight.data = normalize_weights(m.weight.data)
        if hasattr(m, "bias") and m.bias is not None:
            m.bias.data = normalize_and_quantize_tensor(m.bias.data)


class MLP(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_sizes,
        num_classes,
        dropout_rate=0.05,
        quantization_parameter=0.0,
    ):
        super(MLP, self).__init__()
        self.quantization_parameter = quantization_parameter
        layers = []
        in_features = input_size
        for hidden_size in hidden_sizes:
            layers.extend(
                (
                    nn.Linear(in_features, hidden_size),
                    nn.BatchNorm1d(hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                )
            )
            in_features = hidden_size
        layers.append(nn.Linear(in_features, num_classes))
        self.layers = nn.Sequential(*layers)
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.uniform_(m.weight, 0, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.layers(x)

    def set_quantization_parameter(self, quantization_parameter):
        self.quantization_parameter = quantization_parameter

    def quantize_weights(self):
        for m in self.modules():
            if hasattr(m, "weight"):
                m.weight.data = self.quantize_tensor(m.weight.data)

    def quantize_tensor(self, tensor):
        if tensor.nelement() == 0:
            return tensor

        quantized_tensor = normalize_and_quantize_tensor(
            tensor, normalized_conductance_values
        )
        return (
            1 - self.quantization_parameter
        ) * tensor + self.quantization_parameter * quantized_tensor

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


def clamp_weights(model):
    for m in model.modules():
        if hasattr(m, "weight"):
            m.weight.data.clamp_(0, 1)
        if hasattr(m, "bias") and m.bias is not None:
            m.bias.data.clamp_(0, 1)


def train(
    model,
    criterion,
    optimizer,
    train_loader,
    val_loader,
    epochs=10,
    quantization_warmup=50,
    quantization_steps=50,
    patience=2000,
):
    model.train()
    train_loss_history = []
    val_loss_history = []
    val_accuracy_history = []

    best_model_weights = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    best_train_loss = float("inf")
    best_val_accuracy = 0.0

    best_finetune_weights = None
    best_finetune_train_loss = float("inf")
    best_finetune_val_loss = float("inf")
    best_finetune_val_accuracy = 0.0

    epochs_without_improvement = 0

    for epoch in range(epochs):
        total_train_loss = 0
        model.train()
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            clamp_weights(model)
            total_train_loss += loss.item()

        val_loss = 0
        correct = 0
        model.eval()
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        val_loss /= len(val_loader.dataset)
        accuracy = 100.0 * correct / len(val_loader.dataset)

        train_loss_history.append(total_train_loss / len(train_loader))
        val_loss_history.append(val_loss)
        val_accuracy_history.append(accuracy)

        if accuracy > best_val_accuracy or (
            accuracy == best_val_accuracy and val_loss < best_val_loss
        ):
            best_val_accuracy = accuracy
            best_val_loss = val_loss
            best_train_loss = total_train_loss / len(train_loader)
            best_model_weights = copy.deepcopy(model.state_dict())

        if accuracy < 100.0:
            epochs_without_improvement += 1
        else:
            epochs_without_improvement = 0

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        if epoch >= quantization_warmup:
            quantization_progress = min(
                1.0, (epoch - quantization_warmup + 1) / quantization_steps
            )
            model.set_quantization_parameter(quantization_progress)
            model.quantize_weights()

    model.load_state_dict(best_model_weights)

    print(
        f"Best pre-finetuning model -- Train Loss: {best_train_loss:.4f}, Val Loss: {best_val_loss:.4f}, Val Accuracy: {best_val_accuracy:.2f}%"
    )

    # Fine-tuning with quantized weights
    for epoch in range(epochs):
        total_train_loss = 0
        model.train()
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            clamp_weights(model)
            total_train_loss += loss.item()

        val_loss = 0
        correct = 0
        model.eval()
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        val_loss /= len(val_loader.dataset)
        accuracy = 100.0 * correct / len(val_loader.dataset)

        if accuracy > best_finetune_val_accuracy or (
            accuracy == best_finetune_val_accuracy
            and (
                val_loss < best_finetune_val_loss
                or total_train_loss / len(train_loader) < best_finetune_train_loss
            )
        ):
            best_finetune_val_accuracy = accuracy
            best_finetune_val_loss = val_loss
            best_finetune_train_loss = total_train_loss / len(train_loader)
            best_finetune_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_finetune_weights or best_model_weights)

    print(
        f"Best finetuning model -- Train Loss: {best_finetune_train_loss:.4f}, Val Loss: {best_finetune_val_loss:.4f}, Val Accuracy: {best_finetune_val_accuracy:.2f}%"
    )

    return model, train_loss_history, val_loss_history, val_accuracy_history


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
            p_swa.data.mul_(self.n_averaged).add_(p_model.data).div_(
                self.n_averaged + 1
            )
