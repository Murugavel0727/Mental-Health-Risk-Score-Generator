import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import MentalHealthDataset
import sys
import os

# Add backend to path to import model
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from backend.model import MentalHealthModel

def train():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    dataset = MentalHealthDataset(num_samples=1000)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = MentalHealthModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Loop
    epochs = 5
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in dataloader:
            text = batch['text'].to(device)
            audio = batch['audio'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(text, audio)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
    
    # Save
    torch.save(model.state_dict(), "../backend/model.pth")
    print("Model saved to ../backend/model.pth")

if __name__ == "__main__":
    train()
