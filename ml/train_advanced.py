import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np

# Define Path to downloaded data
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "mental_health_risk_prediction.csv")

class TabularDataset(Dataset):
    def __init__(self, csv_file):
        # Load Data
        if not os.path.exists(csv_file):
            print(f"Warning: {csv_file} not found. using mock data.")
            self.data = pd.DataFrame({
                'age': np.random.randint(18, 60, 100),
                'stress_level': np.random.random(100),
                'sleep_quality': np.random.random(100),
                'mental_health_risk': np.random.random(100)
            })
        else:
            self.data = pd.read_csv(csv_file)
            
        # Feature Engineering (Normalize)
        self.features = self.data[['age', 'stress level', 'sleep quality']].values.astype(np.float32)
        self.labels = self.data['mental health risk'].values.astype(np.float32)
        
        # Simple Normalization
        self.features[:, 0] = self.features[:, 0] / 100.0 # Age
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.features[idx]), torch.tensor(self.labels[idx])

class TabularModel(nn.Module):
    def __init__(self):
        super(TabularModel, self).__init__()
        self.fc1 = nn.Linear(3, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.sigmoid(x)

def train_advanced():
    dataset = TabularDataset(DATA_PATH)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    model = TabularModel()
    criterion = nn.BCELoss() # Binary Cross Entropy if risk is 0-1 probability
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    print("Training Tabular Model...")
    for epoch in range(20):
        total_loss = 0
        for features, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(features).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Loss {total_loss/len(dataloader):.4f}")
            
    torch.save(model.state_dict(), "../backend/tabular_model.pth")
    print("Tabular model saved.")

if __name__ == "__main__":
    train_advanced()
