import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np
from math import cos, sin, pi
from copy import deepcopy

from Config import Config
from vcontour import tangent, target, dominant_region, min_dist_to_target, time_to_cap
from coords import dthtr_to_phy

r = Config.CAP_RANGE
R = Config.TAG_RANGE
a = Config.VD/Config.VI

def plot_defenders(ax, xds):
    for xd in xds:
        x, y = [], []
        for tht in np.linspace(0, 2*pi, 50):
            x.append(xd[0] + r*cos(tht))
            y.append(xd[1] + r*sin(tht))
        ax.plot(x, y, color='r')
        ax.plot(xd[0], xd[1], 'r.')

def plot_target(ax, R=Config.TAG_RANGE, color='b', linestyle='dashed'):
    
    def get_tg():
        k = 1.1
        x = np.linspace(-k*R, k*R)
        y = np.linspace(-k*R, k*R)
        X, Y = np.meshgrid(x, y)
        T = np.zeros(np.shape(X))
        for i, (xx, yy) in enumerate(zip(X, Y)):
            for j, (xxx, yyy) in enumerate(zip(xx, yy)):
        #        print(dominant_region(np.array([xxx, yyy])))
                T[i,j] = target(np.array([xxx, yyy]), R=R)
        
        return {'X': X, 'Y': Y, 'T': T}
    
    tgt = get_tg()
    CT = ax.contour(tgt['X'], tgt['Y'], tgt['T'], [0], linestyles=(linestyle,))
    plt.contour(CT, levels = [0], colors=(color,), linestyles=(linestyle,))

def plot_dr(ax, xi, xds, color='g', linestyle='dashed', ind=True):
    
    def get_dr(xi, xds):
        k = 1.5
        x = np.linspace(xi[0]-k*R, xi[0]+k*R)
        y = np.linspace(xi[1]-k*R, xi[1]+k*R)
        X, Y = np.meshgrid(x, y)
        D = np.zeros(np.shape(X))
        for i, (xx, yy) in enumerate(zip(X, Y)):
            for j, (xxx, yyy) in enumerate(zip(xx, yy)):
        #        print(dominant_region(np.array([xxx, yyy])))
                D[i,j] = dominant_region(np.array([xxx, yyy]), xi, xds)
        return {'X': X, 'Y': Y, 'D': D}

    # locations of players
    ax.plot(xi[0], xi[1], '.', color=color)
    # for xd in xds:
    #     ax.plot(xd[0], xd[1], '.', color=color)

    # dr unioned
    dr = get_dr(xi, xds)
    CD = ax.contour(dr['X'], dr['Y'], dr['D'], [0], linestyles='solid')
    plt.contour(CD, levels = [0], colors=(color,), linestyles=('solid',))

    # dr individual
    if ind:
        for xd in xds:
            dr = get_dr(xi, [xd])
            CD = ax.contour(dr['X'], dr['Y'], dr['D'], [0], linestyles='dashed')
            plt.contour(CD, levels = [0], colors=(color,), linestyles=('dashed',))


def plot_constd(ax, xds, levels=[0], R=Config.TAG_RANGE, color='k', linestyle='solid'):

    def get_constd(xds):
        k = 3.
        x = np.linspace(-k*Config.TAG_RANGE, k*Config.TAG_RANGE)
        y = np.linspace(-k*Config.TAG_RANGE, k*Config.TAG_RANGE)
        X, Y = np.meshgrid(x, y)
        C = np.zeros(np.shape(X))
        for i, (xx, yy) in enumerate(zip(X, Y)):
            for j, (xxx, yyy) in enumerate(zip(xx, yy)):
        #        print(dominant_region(np.array([xxx, yyy])))
                C[i,j] = min_dist_to_target(np.array([xxx, yyy]), xds)
        return {'X': X, 'Y': Y, 'C': C}

    vctr = get_constd(xds)
    CC = ax.contour(vctr['X'], vctr['Y'], vctr['C'], levels, linestyles=(linestyle,))
    ax.clabel(CC, inline=True, fontsize=10)
    ax.contour(vctr['X'], vctr['Y'], vctr['C'], levels=levels, colors=(color,), linestyles=(linestyle,))


def plot_constt(ax, xds, R=Config.TAG_RANGE, color='k', linestyle='solid'):

    def get_constt(xds):
        k = 2
        x = np.linspace(-k*Config.TAG_RANGE, k*Config.TAG_RANGE)
        y = np.linspace(-k*Config.TAG_RANGE, k*Config.TAG_RANGE)
        X, Y = np.meshgrid(x, y)
        T = np.zeros(np.shape(X))
        for i, (xx, yy) in enumerate(zip(X, Y)):
            for j, (xxx, yyy) in enumerate(zip(xx, yy)):
        #        print(dominant_region(np.array([xxx, yyy])))
                T[i,j] = time_to_cap(np.array([xxx, yyy]), xds)
        return {'X': X, 'Y': Y, 'T': T}

    tctr = get_constt(xds)
    CT = ax.contour(tctr['X'], tctr['Y'], tctr['T'], linestyles=(linestyle,))
    plt.contour(CT, colors=(color,), linestyles=(linestyle,))
    ax.clabel(CT, inline=True, fontsize=10)
#    ax.contour(tctr['X'], tctr['Y'], tctr['T'], colors=(color,), linestyles=(linestyle,))
    
    
def plot_vcontour(ax, xds, dmin, R=Config.TAG_RANGE, color='k'):
    
    plot_defenders(ax, xds)
    plot_target(ax, R=R, linestyle='solid')
#    for R in Rs:
#        plot_constd(ax, r1, R=R, linestyle='solid')
    plot_constd(ax, xds, levels=[dmin], color=color)
#    plot_constt(ax, r1) 

def plot_capture_ring(ax, loc_D, n=50, line_style=(0, ()), color='b'):

    def capture_ring(t, loc):
        return np.array([loc[0]+r*cos(t), loc[1]+r*sin(t)])

    def sample_capture_ring(loc_D, n=50):

        ts = np.linspace(0, 2*pi, num=n)
        x = np.array([[0.0, 0.0] for _ in range(n)])
        for i, t in enumerate(ts):
            x[i] = capture_ring(t, loc_D)
        return x

    cp = sample_capture_ring(loc_D)
    ax.plot(cp[:,0], cp[:,1], color=color, linestyle=line_style)
    
def plot_traj(ax, xs, Rmin, skip=15, line_style='-', dr_colors=['y', 'g', 'c']):
    
    xds = [x[0] for x in xs]
    yds = [x[1] for x in xs]
    xis = [x[2] for x in xs]
    yis = [x[3] for x in xs]

    ax.plot(xds[-1], yds[-1], 'b*')
    ax.plot(xis[-1], yis[-1],  'r*')

#    ax.plot(xds, yds, '.', markevery=skip, color='b', linestyle=line_style, label='D')
#    ax.plot(xis, yis, '.', markevery=skip, color='r', linestyle=line_style, label='I')
#
#    plot_capture_ring(ax, np.array([xds[-1], yds[-1]]), color='r', line_style=line_style)
    plot_target(ax, R=Rmin, color='b', linestyle='dashed')
    
    nc = len(dr_colors)
    for i, x in enumerate(xs):
        if i%skip == 0:
            plot_dr(ax, x, color=dr_colors[int(i/skip)%nc])
    
    
