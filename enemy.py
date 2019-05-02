from enum import Enum

class Mode(Enum):
    INACTIVE = 0
    SEEKING = 1
    ATTACKING = 2


class Enemy():

    def __init__(self):
        self.mode = Mode.SEEKING
        self.alert = 0
        self.timer = 0

class Charger(Enemy):

    def __init__(self):
        super(Charger, self).__init__()

    def defaultRad(self):
        return 10000

class Bomber(Enemy):

    def __init__(self):
        super(Bomber, self).__init__()

    def defaultRad(self):
        return 25

class Attractor(Enemy):

    def __init__(self):
        super(Attractor, self).__init__()

class Tower(Enemy):
    def __init__(self):
        super(Tower, self).__init__()
        self.ang = -3.1415/2.
        self.hot = 0
        self.rot = 0.01
