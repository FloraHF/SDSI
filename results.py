import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from math import pi, acos

from Config import Config
from SDSIgame import SDSIGame
from vcontour import tangent
from plotter import plot_vcontour, plot_traj

#ss = []
#colors = ['g', 'c', 'b']
#for R in Rs:
#    ss.append(const_v(R))
#plot_vcontour(ss, Rs, colors=colors, drs=[True, False, False])

#r = Config.CAP_RANGE
#R = Config.TAG_RANGE
#a = Config.VD/Config.VI
#Rs = np.linspace(0.8*R, 1.8*R, 3)
#plot_vcontour(2*R, [Rs[0]])

game = SDSIGame()
game.reset()
xs = game.advance(100)

fig, ax = plt.subplots()
plot_vcontour(ax, xs[0][1], game.Rmin)
plot_traj(ax, xs, game.Rmin)

k = 2
ax.axis('equal')
plt.grid()
plt.savefig('test.png')   
