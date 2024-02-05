def predict(model, data):
    model.eval()
    with torch.no_grad():
        output = model(data)
        return output.argmax(dim=1)
