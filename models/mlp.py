import torch
import torch.nn as nn 

class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.layers= nn.Sequential(nn.Linear(784,512),
                    nn.ReLU(),
                    nn.Linear(512,256),
                    nn.ReLU(),
                    nn.Linear(256,128),
                    nn.ReLU(),
                    nn.Linear(128,64),
                    nn.ReLU(),
                    nn.Linear(64,10))

    def forward(self,x):
        x=x.view(x.size(0),-1)
        x=self.layers(x)
        return x
