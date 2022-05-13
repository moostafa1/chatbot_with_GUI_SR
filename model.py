import torch
from torch import nn
#import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        #super(NeuralNet, self).__init__()  # returns temporary object of the superclass that allows access to all of its methods to the clild class
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_size)    #Apply linear transformation to the incoming data  (x(A.T)+b)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()      # max(0, x)    #non-linear activation function

    def forward(self, x):      # forward propagation to compute loss
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # no activations and no softmax
        # as we'll apply cross-entropy loss, and this'll apply it for us

        return out
