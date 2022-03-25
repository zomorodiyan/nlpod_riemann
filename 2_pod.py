# -*- coding: utf-8 -*-
"""
POD basis generation for the 2D Marsigli flow problem
This code was used to generate the results accompanying the following paper:
    "Nonlinear proper orthogonal decomposition for convection-dominated flows"
    Authors: Shady E Ahmed, Omer San, Adil Rasheed, and Traian Iliescu
    Published in the Physics of Fluids journal

For any questions and/or comments, please email me at: shady.ahmed@okstate.edu
"""

#%% Import libraries
import numpy as np
import os
import sys

from scipy.fftpack import dst, idst
from numpy import linalg as LA
import matplotlib.pyplot as plt

#%% Define functions


def import_fom_data(bc, n):
    filename = "./Results/npz/bc_" + str(bc) + "/P" + str(n) + ".npy"
    z = np.load(filename)
    return z


def POD_svd(zdata, nr):
    ne, nd, ns = zdata.shape
    Az = np.zeros([nd, ns * ne])
    # stack data along first axis
    for p in range(ne):
        Az[:, p * ns : (p + 1) * ns] = zdata[p, :, :]

    # mean subtraction
    zm = np.mean(Az, axis=1)
    Az = Az - zm.reshape([-1, 1])

    # singular value decomposition
    Uz, Sz, Vhz = LA.svd(Az, full_matrices=False)

    Phiz = Uz[:, :nr]
    Lz = Sz ** 2
    # compute RIC (relative importance index)
    RICz = np.cumsum(Lz) / np.sum(Lz) * 100
    return zm, Phiz, Lz, RICz


def PODproj_svd(u, Phi):  # Projection
    a = np.dot(Phi.T, u)  # u = Phi * a if shape of a is [nr,ns]
    return a


def PODrec_svd(a, Phi):  # Reconstruction
    u = np.dot(Phi, a)
    return u


#%% Main program
# Inputs
lx = 1
ly = 1
nx = 100
ny = nx

bcList = [1, 2, 3, 4, 5]
bcTrain = [1, 2, 3, 5]
# Ri = 4
# Pr = 1

Tm = 0.25
nt = 50
ns = nt
dt = 0.25 / nt
freq = int(nt / ns)
# twick 1

#%% grid
dx = lx / nx
dy = ly / ny

x = np.linspace(0.0, lx, nx)
y = np.linspace(0.0, ly, ny)
X, Y = np.meshgrid(x, y, indexing="ij")

#%% Loading data
print("loading ...", end=" ")
zdata = np.zeros([len(bcList), (nx) * (ny), ns + 1])  # wdata is pressure

nstart = 0
nend = nt
nstep = freq
for i, bc in enumerate(bcList):
    j = 0
    for n in range(nstart, nend + 1, nstep):
        z = import_fom_data(bc, n)  # twick_3 we can add nx_ny too
        zdata[i, :, j] = z.reshape(
            [
                -1,
            ]
        )
        j = j + 1

print("done!")
#%% POD basis generation
print("generating POD basis ...", end=" ")
zdata = np.zeros([len(bcList), (nx) * (ny), ns + 1])  # wdata is pressure
nr = 100  # number of basis to store [we might not need to *use* all of them]
# twick 2

# compute mean field and basis functions for potential voriticity
mask = [bcList.index(bc) for bc in bcTrain]
inc = 1
zmean, zbasis, zeigenvalues, zric = POD_svd(zdata[mask, :, ::inc], nr)
print("done!")

#%% compute true modal coefficients
print("computing coeff ...", end=" ")
zcoeff = np.zeros([len(bcList), nr, ns + 1])

for i, bc in enumerate(bcList):
    z = np.copy(zdata[i, :, :])
    tmp = z - zmean.reshape(-1, 1)
    zcoeff[i, :, :] = PODproj_svd(tmp, zbasis)

print("done!")
#%% Save data

print("make folder ./Results/euler_nx_ny")
folder = "euler_" + str(nx) + "_" + str(ny)
if not os.path.exists("./Results/" + folder):
    os.makedirs("./Results/" + folder)

filename = "./Results/" + folder + "/P_POD_data.npz"
print("saving ...", end=" ")
np.savez(
    filename,
    zmean=zmean,
    zbasis=zbasis,
    zcoeff=zcoeff,
    zeigenvalues=zeigenvalues,
    zric=zric,
)
print("done!")
