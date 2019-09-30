import numpy as np
from copy import deepcopy
from math import asin, acos, sin, cos, tan, pi, atan2, copysign

from Config import Config
from vcontour import min_dist_to_target
from strategies import s_strategy

r = Config.CAP_RANGE
R = Config.TAG_RANGE
a = Config.VD/Config.VI
####################################################################################
################################### ENVIRONMENT ####################################

def sign(x):
    return copysign(1, x)

class InitialLocationGenerator(object):

    def __init__(self,  r1_range=[R, 2*R],
                        r2_range=[R, 2*R]):

        self.r1_range = r1_range
        self.r2_range = r2_range

    def generate(self):
        r1 = np.random.uniform(self.r1_range[0], self.r1_range[1])
        r2 = np.random.uniform(self.r2_range[0], self.r2_range[1])
        if abs(r1 - r2) < r:
            tht_min = acos((r1**2 + r2**2 - r**2)/(2*r1*r2))
        else:
            tht_min = 0
        tht = np.random.uniform(tht_min, pi/2)

        return [np.array([0, r1]), r2*np.array([sin(tht), cos(tht)])]

class SDSIGame():
    def __init__(self, game_name='SDSI', x_0=Config.x0):
        
        self.game = game_name
        self.dcount = 1
        self.icount = 1
        self.pcount = self.dcount + self.icount
        self.r = Config.CAP_RANGE
        self.x_generator = InitialLocationGenerator()
#        self.Vfn = ValueFunc(read_dir=vfunc_dir)
        
#        self.frame=frame
        self.x_0 = x_0

        self.defenders = [Player(self, 'defender', self.x_0[0])]
        self.intruders = [Player(self, 'intruder', self.x_0[1])]
        self.players = self.defenders + self.intruders
        self.Rmin = min_dist_to_target(self.get_x()) + R
        
        self.done = False
        
    def is_captured(self):
        cap = False
        for d in self.defenders:
            dcap = np.linalg.norm(d.x - self.intruders[0].x) < self.r
            cap = cap or dcap
        return cap

    def get_x(self):
        return np.concatenate([p.x for p in self.players])

    def step(self, action_n):
        
        for player, action in zip(self.players, action_n):
            player.step(action)
        
        if self.is_captured():
            self.done = True

        return self.get_x(), self.done
    
    def opt_strategy(self, x):
        return s_strategy(x)
    
    def advance(self, n_steps, policy=None):
        if policy is None:
            policy = self.opt_strategy
        xs = [self.get_x()]
        
        for t in range(n_steps):
            act = policy(xs[-1])
            x, done = self.step(act)
            xs.append(x)
            if done:
                print('captured')
                break
                
        return xs
    
    def reset(self, xs=None, rand=False):
        
        if xs is None:
            if rand:
                xs = self.x_generator.generate()
            else:
                xs = Config.x0
                
        for p, x in zip(self.players, xs):
            p.reset(x)
        
        self.Rmin = min_dist_to_target(self.get_x()) + R

        self.done = False
        
        return self.get_x()

####################################################################################
#################################### PLAYERS #######################################

class Player(object):

    def __init__(self, env, role, x):

        self.env = env
        self.role = role
        self.dt = Config.TIME_STEP
        self.x = x
        
        if role == 'defender':
            self.v = Config.VD
        elif role == 'intruder':
            self.v = Config.VI
    
    def step(self, action):
        self.x += self.dt * self.v * np.array([cos(action), sin(action)])
    
    def reset(self, x):
        self.x = deepcopy(x)
