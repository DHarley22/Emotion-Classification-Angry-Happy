# classifier/models.py
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

class EmotionClassifier(nn.Module):
    def __init__(self, model_path):
        super(EmotionClassifier, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def load_model(self, model_path):
        model = models.efficientnet_b0(pretrained=False)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, 1)
        model.classifier = nn.Sequential(
            model.classifier,
            nn.Sigmoid()
        )
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def predict(self, image_path, threshold=0.5):
        image = Image.open(image_path)
        image = self.transform(image).unsqueeze(0)  # Ajoute une dimension pour le batch
        image = image.to(self.device)

        with torch.no_grad():
            output = self.model(image).item()
            prediction = 'Happy' if output > threshold else 'Angry'
            return prediction
