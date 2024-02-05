import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super(MLP, self).__init__()
        layers = []
        in_features = input_size
        # Create N hidden layers
        for hidden_size in hidden_sizes:
            layers.extend((nn.Linear(in_features, hidden_size), nn.ReLU()))
            in_features = hidden_size
        layers.append(nn.Linear(in_features, num_classes))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


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