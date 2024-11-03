import os
import sys
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Configure logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler("evaluate_model.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

class AgeDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['filepath']
        label = int(self.data.iloc[idx]['class'])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def create_dataloader(csv_path, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ])
    
    dataset = AgeDataset(csv_file=csv_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    logging.info("DataLoader created successfully.")
    return dataloader

def build_model(num_classes=4, device='cpu'):
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
    
    model = model.to(device)
    logging.info("Model built successfully.")
    return model

def load_model(model, model_path):
    if not os.path.exists(model_path):
        logging.error(f"Model file '{model_path}' does not exist.")
        sys.exit(1)
    
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    logging.info(f"Model loaded from '{model_path}'.")
    return model

def evaluate_model(model, dataloader, device='cpu'):
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds)
    
    # Log Metrics
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"F1-Score: {f1:.4f}")
    logging.info("Classification Report:")
    logging.info(f"\n{report}")
    
    # Plot Confusion Matrix
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(model.fc[-1].out_features), yticklabels=range(model.fc[-1].out_features))
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig('models/confusion_matrix.png')
    plt.close()
    logging.info("Confusion matrix saved to 'models/confusion_matrix.png'.")

def main():
    """
    Main function to evaluate the model.
    """
    # csv_path = 'data/train_augmented.csv'  # Change to your test CSV if available
    csv_path = 'data/test.csv'  # Change to your test CSV if available
    model_path = 'models/best_model.pth'
    os.makedirs('models', exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Step 1: Create DataLoader
    dataloader = create_dataloader(csv_path, batch_size=32)
    
    # Step 2: Build and Load the Model
    num_classes = 4  # Adjust if different
    model = build_model(num_classes=num_classes, device=device)
    model = load_model(model, model_path)
    
    # Step 3: Evaluate the Model
    evaluate_model(model, dataloader, device=device)
    
    logging.info("Model evaluation complete.")

if __name__ == "__main__":
    main()
