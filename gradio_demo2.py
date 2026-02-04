import requests
import torch
from PIL import Image
from torchvision import transforms

# Load pretrained ResNet18 model
model = torch.hub.load(
    'pytorch/vision:v0.6.0',
    'resnet18',
    pretrained=True
)
model.eval()

# Download ImageNet labels
response = requests.get("https://git.io/JJkYN")
labels = response.text.split("\n")

# Image preprocessing (VERY important)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

def predict(inp):
    # Apply preprocessing
    inp = preprocess(inp).unsqueeze(0)

    with torch.no_grad():
        outputs = model(inp)
        prediction = torch.nn.functional.softmax(outputs[0], dim=0)

    # Return top-1000 class probabilities
    confidences = {
        labels[i]: float(prediction[i])
        for i in range(len(labels))
    }

    return confidences
