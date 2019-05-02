import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch.optim as optim


class LinearUnit(nn.Module):
    def __init__(self, in_features, out_features, nonlinearity=nn.LeakyReLU(0.2)):
        super(LinearUnit, self).__init__()
        self.residual = (in_features == out_features)
        if nonlinearity is None:
            self.model = nn.Sequential(nn.Linear(in_features, out_features))
        else:
            self.model = nn.Sequential(
                    nn.Linear(in_features, out_features), nonlinearity)

        self.cuda()

    def get_param(self):
        return self.model[0].weight.data, self.model[0].bias.data

    def set_param(self, w, b):
        self.model[0].weight.data = w
        self.model[0].bias.data = b

    def forward(self, x):
        out = self.model(x)
        if self.residual:
            out = out + x
        return out

class ConvUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=1, padding=0, nonlinearity=nn.LeakyReLU(0.2)):
        super(ConvUnit, self).__init__()
        self.model = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel, stride, padding), nonlinearity)

    def forward(self, x):
        return self.model(x)



class NN(nn.Module):
    def __init__(self, input_size, h_size=10, output_size=4):
        super(NN, self).__init__()
        
        self.lin_layers = nn.Sequential(
            LinearUnit(input_size, h_size),
            LinearUnit(h_size, output_size, nonlinearity=nn.Tanh())
        )

        #print(self.lin_layers[0].weight())
        
    def forward(self, x):
        return self.lin_layers(x)
    
    def act(self, state, epsilon):
        if random.random() > epsilon:
            state   = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
            q_value = self.forward(state)
            action  = q_value.max(1)[1].data[0]
        else:
            action = random.randrange(env.action_space.n)
        return action

    def get_param(self):

        weight = []
        bias = []

        for i in self.lin_layers:
            w, b = i.get_param()
            #w = w.cpu().detach().numpy()
            #b = b.cpu().detach().numpy()
            weight.append(w)
            bias.append(b)

        return weight, bias

    def set_param(self, w, b):

        for i, l in enumerate(self.lin_layers):
            w0, b0 = w[i], b[i]
            l.set_param(w0, b0)


    def mutate(self, s=0.05):

        w, b = self.get_param()

        for i in range(len(w)):
            w[i] += torch.randn(w[i].shape).cuda() * s
            b[i] += torch.randn(b[i].shape).cuda() * s

        self.set_param(w, b)

if __name__ == '__main__':

    nn = NN(65)

    inp = torch.randn((1, 65)).cuda()

    out = nn(inp)

    print(out[0])

    print("~~~")

    w, b = nn.get_param()

    nn.set_param(w, b)

    out = nn(inp)

    print(out[0])

    for i in range(10):

        print("~~~")

        nn.mutate()

        out = nn(inp)

        print(out[0])