import torch
import torch.nn as nn

class TextBranch(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=128):
        super(TextBranch, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, 64)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class AudioBranch(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64):
        super(AudioBranch, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, 64)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class MentalHealthModel(nn.Module):
    def __init__(self):
        super(MentalHealthModel, self).__init__()
        self.text_branch = TextBranch()
        self.audio_branch = AudioBranch()
        
        # Fusion Layer
        self.fusion_fc1 = nn.Linear(64 + 64, 64)
        self.fusion_relu = nn.ReLU()
        self.output_layer = nn.Linear(64, 1) # Regression output (Risk Score 0-100) or Probability (0-1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text_input, audio_input):
        t_out = self.text_branch(text_input)
        a_out = self.audio_branch(audio_input)
        
        # Concatenate
        combined = torch.cat((t_out, a_out), dim=1)
        
        # Fusion
        x = self.fusion_fc1(combined)
        x = self.fusion_relu(x)
        x = self.output_layer(x)
        return self.sigmoid(x)
