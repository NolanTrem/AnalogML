import numpy as np
import torch

from mlp import MLP, predict
from text_to_matrix import TextMatrixGenerator

input_size = 100
num_classes = 5

model = MLP(input_size=input_size, hidden_sizes=[8], num_classes=num_classes)
model.load_state_dict(torch.load("trained_model.pth"))


generator = TextMatrixGenerator(size=10)
new_letter = "B"
new_matrix = generator.text_to_matrix(new_letter)
new_vector = new_matrix.flatten()
new_X = np.array([new_vector])
new_X_tensor = torch.Tensor(new_X)

# Run inference
prediction = predict(model, new_X_tensor)
predicted_class = prediction.item()
print(f"Predicted class for letter '{new_letter}': {predicted_class}")
