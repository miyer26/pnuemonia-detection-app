import torch
from torchvision import transforms
from PIL import Image

class Inference:
    def __init__(self, model_path, transforms):
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._get_model()
        self.transforms = transforms
        
    def _get_model(self):
        model = torch.load("best_model_elastic.pth", map_location=self.device)
        return model

    def predict(self, image_path):
        self.model.eval()

        # Open the image using PIL
        image = Image.open(image_path)

        # Convert the image to RGB format if it's not already in RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')

        transformed_image = self.transforms.base_transform(image)
        transformed_image = transformed_image.unsqueeze(0)

        # Move the transformed image tensor to the appropriate device
        transformed_image = transformed_image.to(self.device)

        with torch.no_grad():
            outputs = self.model(transformed_image)
            label = torch.argmax(outputs, dim=-1).item()

        return label