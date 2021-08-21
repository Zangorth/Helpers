from collections import OrderedDict
from torch import nn
import numpy as np
import torch

device = torch.device('cuda:0')

class Discriminator():
    def __init__(self, drop, neurons, lr_nn, epochs, output, layers=3):
        self.drop, self.output, self.layers = drop, output, layers
        self.neurons = neurons
        self.lr_nn, self.epochs = lr_nn, epochs
        
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
            output = self.model(x)
            return output
    
    def fit(self, x, y):
        col_count = x.shape[1]
        x, y = torch.from_numpy(x.values).to(device), torch.from_numpy(y.values).to(device)
        
        train_set = [(x[i].to(device), y[i].to(device)) for i in range(len(y))]
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=2**10, shuffle=True)
    
        loss_function = nn.CrossEntropyLoss()
        discriminator = self.Classifier(col_count, self.neurons, self.drop, self.output, self.layers).to(device)
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
        discriminator.to(device).eval()
        
        x = torch.from_numpy(x.values).to(device)
        preds = np.argmax(discriminator(x.float()).cpu().detach(), axis=1)
        
        return preds
    
    def predict_proba(self, x):
        discriminator = self.model
        discriminator.to(device).eval()
        
        x = torch.from_numpy(x.values).to(device)
        preds = discriminator(x.float()).cpu().detach()[:, 1]
        
        return preds
        