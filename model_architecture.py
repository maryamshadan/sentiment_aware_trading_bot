import torch.nn as nn

class PolicyNetwork(nn.Module):
    def __init__(self, state_size=5, action_size=3):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 64), # Index 0
            nn.ReLU(),                 # Index 1
            nn.Dropout(0.2),           # Index 2 
            nn.Linear(64, 64),         # Index 3
            nn.ReLU(),                 # Index 4
            nn.Linear(64, action_size) # Index 5
        )

    def forward(self, x):
        return self.fc(x)
