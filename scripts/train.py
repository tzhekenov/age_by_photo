# train_model.py

import os
import sys
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Configure logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler("train_model.log", encoding='utf-8'),
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

def create_dataloaders(csv_path, batch_size=32, val_split=0.2):
    transform = transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ])
    
    dataset = AgeDataset(csv_file=csv_path, transform=transform)
    dataset_size = len(dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    logging.info("DataLoaders created successfully.")
    return train_loader, val_loader

def build_model(num_classes=4):
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
    
    logging.info("Model built successfully.")
    return model

def train_model(model, dataloaders, criterion, optimizer, num_epochs=10, device='cuda'):
    best_accuracy = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        logging.info(f"Epoch {epoch+1}/{num_epochs}")
        logging.info("-" * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                loader = dataloaders['train']
            else:
                model.eval()
                loader = dataloaders['val']
            
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in tqdm(loader, desc=phase.capitalize()):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(loader.dataset)
            epoch_acc = running_corrects.double() / len(loader.dataset)
            
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())
            
            logging.info(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            
            # Save the best model
            if phase == 'val' and epoch_acc > best_accuracy:
                best_accuracy = epoch_acc
                torch.save(model.state_dict(), 'models/best_model.pth')
                logging.info("Best model saved.")
    
    logging.info(f"Training complete. Best Accuracy: {best_accuracy:.4f}")
    return history

def plot_history(history, output_path='models/training_history.png'):
    plt.figure(figsize=(12,5))
    
    # Accuracy
    plt.subplot(1,2,1)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    # Loss
    plt.subplot(1,2,2)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Training history saved to '{output_path}'.")

def main():
    """
    Main function to train the model.
    """
    csv_path = 'data/train_augmented.csv'
    model_save_path = 'models/best_model.pth'
    history_save_path = 'models/training_history.png'
    os.makedirs('models', exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Step 1: Create DataLoaders
    train_loader, val_loader = create_dataloaders(csv_path)
    dataloaders = {'train': train_loader, 'val': val_loader}
    
    # Step 2: Build the model
    model = build_model(num_classes=4)
    model = model.to(device)
    
    # Step 3: Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=1e-4)
    
    # Step 4: Train the model
    history = train_model(model, dataloaders, criterion, optimizer, num_epochs=10, device=device)
    
    # Step 5: Plot training history
    plot_history(history, output_path=history_save_path)
    
    # Step 6: Save the final model
    final_model_path = 'models/resnet50_age_classification_final.pth'
    torch.save(model.state_dict(), final_model_path)
    logging.info(f"Final model saved to '{final_model_path}'.")

if __name__ == "__main__":
    main()
