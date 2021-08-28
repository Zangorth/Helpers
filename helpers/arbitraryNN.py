from collections import OrderedDict
from torch import nn
import numpy as np
import torch

class Discriminator():
    def __init__(self, drop=0.1, neurons=[64, 32, 16], lr_nn=0.0001, 
                 epochs=20, layers=3, device='cuda:0', batch_size=2**7):
        self.drop, self.layers = drop, layers
        self.neurons = neurons
        self.lr_nn, self.epochs = lr_nn, epochs
        self.device = torch.device(device)
        self.batch_size = batch_size
        
        return None
        
    class Classifier(nn.Module):
        def __init__(self, shape, neurons, drop, output, layers=3):
            super().__init__()
            
            neurons = [shape] + neurons
            sequential = OrderedDict()
            
            i = 0
            while i < layers:
                sequential[f'linear_{i}'] = nn.Linear(neurons[i], neurons[i+1])
                sequential[f'relu_{i}'] = nn.ReLU()
                sequential[f'drop_{i}'] = nn.Dropout(drop)
                i+=1
                
            sequential['linear_final'] = nn.Linear(neurons[i], output)
            sequential['softmax'] = nn.Softmax(dim=1)
            
            self.model = nn.Sequential(sequential)
            
        def forward(self, x):
            out = self.model(x)
            return out
    
    def fit(self, x, y):
        col_count = x.shape[1]
        output = len(set(y))
        
        x, y = torch.from_numpy(x.values).to(self.device), torch.from_numpy(y.values).to(self.device)
        
        train_set = [(x[i].to(self.device), y[i].to(self.device)) for i in range(len(y))]
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
    
        loss_function = nn.CrossEntropyLoss()
        discriminator = self.Classifier(col_count, self.neurons, self.drop, output, self.layers).to(self.device)
        optim = torch.optim.Adam(discriminator.parameters(), lr=self.lr_nn)
    
        for epoch in range(self.epochs):
            for i, (inputs, targets) in enumerate(train_loader):
                discriminator.zero_grad()
                yhat = discriminator(inputs.float())
                loss = loss_function(yhat, targets.long())
                loss.backward()
                optim.step()
                
        self.model = discriminator
        
        return None
    
    def predict(self, x):
        discriminator = self.model
        discriminator.to(self.device).eval()
        
        x = torch.from_numpy(x.values).to(self.device)
        preds = np.argmax(discriminator(x.float()).cpu().detach(), axis=1)
        
        return preds
    
    def predict_proba(self, x):
        discriminator = self.model
        discriminator.to(self.device).eval()
        
        x = torch.from_numpy(x.values).to(self.device)
        preds = discriminator(x.float()).cpu().detach()
        
        return preds
