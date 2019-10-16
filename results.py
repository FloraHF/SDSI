import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from math import pi, cos, sin

from Config import Config
from SDSIgame import SDSIGame
from vcontour import tangent
from plotter import plot_vcontour, plot_traj, plot_constd, plot_dr

r = Config.CAP_RANGE
R = Config.TAG_RANGE
a = Config.VD/Config.VI

rs   = [1.6,  1.9]
thts = [pi/2, -pi/8]
ri   = 2
thti = pi/4
xds = []
for r, tht in zip(rs, thts):
	xds.append(R*r*np.array([cos(tht), sin(tht)]))
xi = R*ri*np.array([cos(thti), sin(thti)])

fig, ax = plt.subplots()
plot_vcontour(ax, xds, 0, color='k')
plot_dr(ax, xi, xds)
ax.axis('equal')
plt.grid()
plt.show()

