import numpy as np
from math import cos, acos, sqrt, sin, pi
from scipy.optimize import NonlinearConstraint, minimize

from Config import Config
from coords import dthtr_to_phy

r = Config.CAP_RANGE
a = Config.VD/Config.VI

def dominant_region(x, xi, xds):
    for i, xd in enumerate(xds):
        if i == 0:
            inDR = a*np.linalg.norm(x-xi) - (np.linalg.norm(x-xd) - r)
        else:
            inDR = max(inDR, a*np.linalg.norm(x-xi) - (np.linalg.norm(x-xd) - r))
    return inDR

def target(x, R=Config.TAG_RANGE):
    return sqrt(x[0]**2 + x[1]**2) - R

def tangent(xi, xds):
    def dr(x, xi=xi, xds=xds):
        return dominant_region(x, xi, xds)
    def target_square(x):
        return x[0]**2 + x[1]**2
    on_dr = NonlinearConstraint(dr, -np.inf, 0)
    sol = minimize(target_square, xi, constraints=(on_dr,))
    return sol.x, sqrt(target_square(sol.x)), np.linalg.norm(sol.x - xi)/Config.VI

def min_dist_to_target(xi, xds):
    return tangent(xi, xds)[1] - Config.TAG_RANGE

def time_to_cap(xi, xds):
    return tangent(xi, xds)[2]

    