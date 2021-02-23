import torch
import torch.nn as nn


class Regression(nn.Module):
    def __init__(self, dIn, dOut, nNeurons):
        super(Regression, self).__init__()
        ann = nn.Sequential()
        out = nn.Sequential()

        ann.add_module('Input', nn.Linear(dIn, nNeurons[0]))
        ann.add_module('Relu', nn.LeakyReLU())
        #ann.add_module('Sigmoid', nn.Sigmoid())
        for i in range(len(nNeurons)):
            if i > 0:
                ann.add_module('FCN{0}'.format(i), nn.Linear(nNeurons[i-1], nNeurons[i]))
                # if (i+1) is not len(nNeurons):
                #     ann.add_module('Relu{0}'.format(i), nn.LeakyReLU())
                ann.add_module('Relu{0}'.format(i), nn.LeakyReLU())

        out.add_module('Out', nn.Linear(nNeurons[-1], dOut))

        self.ann = ann
        self.out = out
        #self.relu = nn.LeakyReLU()

    def forward(self, x):
        #x = self.relu(self.ann(x))
        x = self.ann(x)
        y = self.out(x)
        return y

