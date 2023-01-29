import os
import sys
import random
import numpy as np
from datetime import date

from utility_soft_rod_n_node_explicit import *

############ Defining Inputs (in SI units)  ###############################
nv = 3 # Number of vertices/nodes
dt = 1e-5  # TIme step
RodLength = 1  # Rod Length

r = np.random.uniform(low=0.005, high=0.01, size=(nv,))  # Node radii from uniform distribution
# r = [0.0025, 0.005, 0.0025]
# Density variables
rho_metal = 7000
rho_gl = 1000
rho = rho_metal - rho_gl

r0 = 1e-3  # Cross-sectional radius of rod
Y = 1e9  # Young's modulus
g = np.array([0, -9.81])  # gravity
visc = 1000.0  # Viscosity

# Tolerance on force function. This is multiplied by ScaleSolver so that we
# do not have to update it based on edge length and time step size
tol = 1e-3

# Maximum number of iterations in Newton Solver
maximum_iter = 100

# Total simulation time (it exits after t=totalTime)
totalTime = 5

# Indicate whether images should be saved
saveImage = 0

# How often the plot should be saved? (Set plotStep to 1 for each plot to
# be saved)
plotStep = 10000

######################### Utility quantities ####################################

ne = nv - 1
EI = Y * np.pi * r0 ** 4 / 4
EA = Y * np.pi * r0 ** 2

########################## Geometry of the rod ##################################

nodes = np.zeros((nv, 2))

for c in range(0, nv):
    nodes[c, 0] = c * RodLength / ne

###################### Multiplier for Force & Jacobian ##########################

# if (np.linalg.norm(g) == 0.0):
#     ScaleSolver = 1.0
#     print('Inspect ScaleSolver variable. Setting it to 1\n');
# else:
#     ScaleSolver = EI / RodLength ** 2

################################# Compute Mass ##################################

m = np.zeros((2 * nv, 1))

for k in range(0, nv):
    m[2 * k:2 * k + 2, 0] = 4 / 3 * np.pi * pow(r[k], 3) * rho_metal

################################# gravity #######################################

mg = np.zeros((2 * nv, 1))

for k in range(0, nv):
    mg[2 * k:2 * k + 2, 0] = 4 / 3 * np.pi * pow(r[k], 3) * rho_metal * g

############################### Reference length and Voronoi length ############

refLen = np.zeros((ne, 1))

for k in range(0, ne):
    dx = nodes[k + 1, :] - nodes[k, :]
    refLen[k] = np.linalg.norm(dx)

voronoiRefLen = np.zeros((nv, 1))

for k in range(0, nv):
    if k == 0:
        voronoiRefLen[k] = 0.5 * refLen[k]
    elif k == -1:
        voronoiRefLen[k] = 0.5 * refLen[k - 1]
    else:
        voronoiRefLen[k] = 0.5 * (refLen[k - 2] + refLen[k - 1])

##################################### Initial ##################################

q0 = np.zeros((2 * nv, 1))

for c in range(0, nv):
    q0[2 * c] = nodes[c, 0]
    q0[2 * c + 1] = nodes[c, 1]

q = q0

u = (q - q0) / dt

###################### Create directory to save image #############################

today = date.today()
imageDirectory = today.strftime("%b-%d-%Y")

if (saveImage == 1):
    os.makedirs(imageDirectory, exist_ok=True)

############################# Time marching #####################################
Nsteps = round(totalTime / dt)  # number of time steps

ctime = 0

fig = plt.figure(figsize=(5, 5))

all_pos = np.zeros((Nsteps, 1))
all_v = np.zeros((Nsteps, 1))

all_q = []
x_pos, ypos = [], []

for timeStep in range(2, Nsteps):
    print(f't = {ctime}\n')
    q = fun(qOld=q0,
            dt=dt,
            q0=q0,
            m=m,
            u=u,
            nv=nv,
            visc=visc,
            r=r,
            mg=mg,
            tol=tol,
            maximum_iter=maximum_iter,
            EI=EI,
            EA=EA,
            refLen=refLen
            )

    u = (q - q0) / dt  # velocity
    ctime += dt  # current time

    q0 = q  # update x0

    if ((timeStep - 1) % plotStep == 0):
        plt.clf()
        plotrod(q, ctime, fig)
        plt.show(block=False)
        plt.pause(1)

print('Done!')