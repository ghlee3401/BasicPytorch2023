import torch 
import torch.nn as nn 
import numpy as np 
import matplotlib.pyplot as plt 
np.random.seed(42)
from sklearn.datasets import load_breast_cancer
import torch.nn.functional as F


cancer = load_breast_cancer() 
x = cancer.data
y = cancer.target 

print("x : ", np.shape(x))
print("y : ", np.shape(y))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2, random_state=42)


xt = torch.FloatTensor(x_train)
yt = torch.FloatTensor(y_train).unsqueeze(1)


class LogisticRegression(nn.Module):
    def __init__(self, d_input, d_output):
        super().__init__()
        self.model = nn.Linear(d_input, d_output)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        output = self.model(x)
        output = self.sigmoid(output)
        return output


class Trainer:
    def __init__(self, d_input, d_output, lr):
        self._build(d_input, d_output)
        self._set_optimizer(lr)
        self._set_loss()
        
    def _build(self, d_input, d_output):
        self.model = LogisticRegression(d_input, d_output)

    def _set_optimizer(self, lr):
        self.optimizer = torch.optim.SGD(params=self.model.parameters(), lr=lr)

    def _set_loss(self):
        self.loss = nn.BCELoss()
        
    def run(self, xt, yt, epoch):
        cnt = 0
        
        for k in range(epoch):
            idx = np.random.permutation(len(xt)) 
            epoch_loss = 0 
            for i in idx:
                y_hat = self.model(xt[i].unsqueeze(0))
                loss = self.loss(y_hat, yt[i].unsqueeze(0))
                epoch_loss += loss.data.numpy()
                
                self.model.zero_grad()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
            epoch_loss = epoch_loss / len(idx)
            
            print("Epoch : {}, Loss: {}".format(k, epoch_loss))
                
                
trainer = Trainer(30, 1, 1e-5)
trainer.run(xt, yt, 100)