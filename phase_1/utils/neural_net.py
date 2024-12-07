import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchmetrics import Accuracy

class CreditDataSet(Dataset):
    def __init__(self, df):
        super().__init__()
        # Convert data to a NumPy array and assign to self.data
        self.data = df.to_numpy()
        
    # Implement __len__ to return the number of data samples
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        features = self.data[idx, :-1]
        # Assign last data column to label
        label = self.data[idx, -1]
        return features, label


class Net(nn.Module):
    def __init__(self, n_inputs):
        super(Net, self).__init__()
        # Define the three linear layers
        self.fc1 = nn.Linear(n_inputs, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # Pass x through linear layers adding activations
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = nn.functional.relu(self.fc2(x))
        x = self.dropout(x)
        x = nn.functional.sigmoid(self.fc3(x))
        return x
            
def train_model(optimizer, net, criterion, num_epochs, dataloader_train):
    for epoch in tqdm(range(num_epochs)):
        for features, labels in dataloader_train:
            optimizer.zero_grad()
            outputs = net(features.double())
            loss = criterion(outputs, labels.view(-1, 1))
            loss.backward()
            optimizer.step()

def evaluate_accuracy(net, dataloader):
    acc = Accuracy('binary').to(torch.device('cpu'))
    net.eval()  # Set to evaluation mode
    with torch.no_grad():
        for features, labels in dataloader:
            outputs = net(features.double())
            preds = (outputs >= 0.5).float()
            acc(preds, labels.view(-1, 1))
    return acc.compute().item()
    
