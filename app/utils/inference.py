import torch
from app.models.model import load_model


# Load the model
model = load_model()

def inference(input_tensor):
    # Perform inference using the loaded model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = input_tensor.to(device)  # Move input tensor to the correct device
    with torch.no_grad():
        output_tensor = model(input_tensor)  # Run inference
    return output_tensor
