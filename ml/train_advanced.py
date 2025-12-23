import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
import numpy as np

# Define Path to downloaded data
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "mental_health_risk_prediction.csv")

class AdvancedTabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.relu = nn.ReLU()
        
    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return self.relu(out)

class HighAccuracyModel(nn.Module):
    def __init__(self, input_dim):
        super(HighAccuracyModel, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU()
        )
        
        # Deep Residual Network for maximum pattern recognition
        self.res_blocks = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64)
        )
        
        self.output_layer = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.input_layer(x)
        x = self.res_blocks(x)
        return self.output_layer(x)

def train_advanced():
    if not os.path.exists(DATA_PATH):
        print("Data not found. Please run download_data.py first.")
        return

    # Load and Preprocess Data
    df = pd.read_csv(DATA_PATH)
    
    # Advanced Feature Engineering
    # Assuming columns: 'Age', 'Sleep', 'Stress', 'Depression_Score', etc.
    # We maintain only relevant numeric columns for this demo script
    numeric_cols = ['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate'] 
    # Note: Adjust columns based on actual Kaggle dataset structure
    
    X = df.iloc[:, :-1].values # Features
    y = df.iloc[:, -1].values  # Target
    
    # Scale Data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    train_dataset = AdvancedTabularDataset(X_train, y_train)
    test_dataset = AdvancedTabularDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize Model
    model = HighAccuracyModel(input_dim=X.shape[1])
    criterion = nn.BCELoss()
    # Using AdamW with Weight Decay for regularization
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    print("Training High Accuracy Model...")
    best_acc = 0.0
    
    for epoch in range(50):
        model.train()
        total_loss = 0
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for features, labels in test_loader:
                outputs = model(features)
                preds = (outputs > 0.5).float()
                all_preds.extend(preds.numpy())
                all_labels.extend(labels.numpy())
                
        acc = accuracy_score(all_labels, all_preds)
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Loss {total_loss/len(train_loader):.4f} | Val Accuracy: {acc*100:.2f}%")
            
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "../backend/high_acc_model.pth")
            
    print(f"Training Complete. Best Accuracy: {best_acc*100:.2f}%")

if __name__ == "__main__":
    train_advanced()
