import random
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

def get_random_action():

    mu = 0
    s = 1.

    x = random.gauss(mu, s)
    y = random.gauss(mu, s)
    rot_ch = random.gauss(mu, s)
    #rot_y = random.gauss(mu, s)
    shoot = random.gauss(mu, s)

    return x, y, rot_ch, shoot

class Agent():

    def __init__(self):
        
        self.hp = 1
        self.reload = 1.
        self.x = 0
        self.y = 0
        self.ang = 0

        self.rays = None
        self.prev_rays = None

        self.id = id
        self.col = None


    def receive_input(self, rays, x, y, ang):
        self.prev_rays = self.rays
        self.rays = rays

        self.x = x
        self.y = y
        self.ang = ang

    def get_actions(self):


        return 0, 0, 0, 0


class PlayerAgent(Agent):

    def __init__(self):

        super().__init__()



class RandomAgent(Agent):

    def __init__(self):

        super().__init__()


    def get_actions(self):

        return get_random_action()


class NeuralNetAgent(Agent):

    def __init__(self, net):

        super().__init__()

        self.net = net


    def get_actions(self):

        if self.rays is None or self.prev_rays is None:
            return get_random_action()

        state = np.zeros(len(self.rays) * 2 + 5)

        state[:len(self.rays)] = self.prev_rays
        state[len(self.rays):len(self.rays)*2] = self.rays

        state[-5] = self.ang
        state[-4] = self.hp
        state[-3] = self.reload
        state[-2] = self.x
        state[-1] = self.y

        state = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True).cuda()

        #print(state.shape)

        out = self.net(state).cpu().detach().numpy()[0]

        #print(out)

        return out[0], out[1], out[2], out[3]

