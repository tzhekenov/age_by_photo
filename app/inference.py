# app/inference.py

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

class AgeClassifier:
    def __init__(self, model_path, num_classes=4, device=None):
        """
        Initializes the AgeClassifier with the given model path and number of classes.

        Args:
            model_path (str): Path to the trained model weights.
            num_classes (int): Number of classes for classification.
            device (torch.device, optional): Device to run the model on. Defaults to CPU or GPU if available.
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.build_model(num_classes)
        self.model = self.load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((200, 200)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])
        ])
    
    def build_model(self, num_classes):
        """
        Builds the ResNet50 model architecture.

        Args:
            num_classes (int): Number of output classes.

        Returns:
            nn.Module: Modified ResNet50 model.
        """
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        for param in model.parameters():
            param.requires_grad = False  # Freeze all layers
        
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
        return model
    
    def load_model(self, model_path):
        """
        Loads the model weights from the specified path.

        Args:
            model_path (str): Path to the model weights.

        Returns:
            nn.Module: Model with loaded weights.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file '{model_path}' does not exist.")
        model = self.model
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        return model
    
    def predict(self, image_path):
        """
        Predicts the class label for a given image.

        Args:
            image_path (str): Path to the image file.

        Returns:
            int: Predicted class label.
        """
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(image)
            _, preds = torch.max(outputs, 1)
            return preds.item()
    
    def predict_pil_image(self, pil_image):
        """
        Predicts the class label for a PIL Image.

        Args:
            pil_image (PIL.Image.Image): PIL Image object.

        Returns:
            int: Predicted class label.
        """
        image = self.transform(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(image)
            _, preds = torch.max(outputs, 1)
            return preds.item()
