import os
import sys
import random
import numpy as np
from datetime import date

from utility_soft_rod_three_node import *

### These are not important

global q0, q
global m, EI, EA, dt, u, nv, ne
global mg, ctime, RodLength, r0
global ScaleSolver
global tol, maximum_iter
global Y, visc, DeltaL 
global R1, R2, R3 

############ Defining Inputs (in SI units)  ###############################
nv = 3                          # Number of vertices
dt = 1e-2                       # TIme step
RodLength = 0.10                # Rod Length
deltaL = RodLength / (nv - 1)   # Discrete length
R1 = 0.005                      # Radius of sphere
R2 = 0.025
R3 = 0.005

# Density variables
rho_metal = 7000
rho_gl = 1000
rho = rho_metal - rho_gl

r0 = 1e-3                     # Cross-sectional radius of rod
Y = 1e9                       # Young's modulus
g = np.array([0, -9.81])      # gravity
visc = 1000.0                 # Viscosity

# Tolerance on force function. This is multiplied by ScaleSolver so that we
# do not have to update it based on edge length and time step size
tol = 1e-3

# Maximum number of iterations in Newton Solver
maximum_iter = 100

# Total simulation time (it exits after t=totalTime)
totalTime = 10

# Indicate whether images should be saved
saveImage = 0

# How often the plot should be saved? (Set plotStep to 1 for each plot to
# be saved)
plotStep = 50

######################### Utility quantities ####################################

ne = nv - 1
EI = Y * np.pi * r0 ** 4 / 4
EA = Y * np.pi * r0 ** 2

########################## Geometry of the rod ##################################

nodes = np.zeros((nv, 2))

for c in range(0, nv):
    nodes[c, 0] = c * RodLength / ne

###################### Multiplier for Force & Jacobian ##########################

if (np.linalg.norm(g) == 0.0):
    ScaleSolver = 1.0
    print('Inspect ScaleSolver variable. Setting it to 1\n');
else:
    ScaleSolver = EI / RodLength ** 2

################################# Compute Mass ##################################

m = np.zeros((2*nv, 1))

m[0:2,0] = 4/3 * np.pi * R1 ** 3 * rho_metal
m[2:4,0] = 4/3 * np.pi * R2 ** 3 * rho_metal
m[4:6,0] = 4/3 * np.pi * R3 ** 3 * rho_metal

################################# gravity #######################################

mg = np.zeros((2*nv, 1))
mg[0:2,0] = 4/3 * np.pi * R1 ** 3 * rho * g
mg[2:4,0] = 4/3 * np.pi * R2 ** 3 * rho * g
mg[4:6,0] = 4/3 * np.pi * R3 ** 3 * rho * g 

##################################### Initial ##################################
q0 = np.zeros((2*nv, 1))

for c in range(0, nv):
    q0[2*c] = nodes[c, 0]
    q0[2*c + 1] = nodes[c, 1]

q = q0

u = (q - q0)/dt

###################### Create director to save image #############################

today = date.today()
imageDirectory = today.strftime("%b-%d-%Y")

if (saveImage == 1):
    os.makedirs(imageDirectory, exist_ok=True)

############################# Time marching #####################################
Nsteps = round(totalTime / dt) # number of time steps

ctime = 0

fig = plt.figure(figsize=(5,5))

all_pos = np.zeros((Nsteps, 1))
all_v = np.zeros((Nsteps, 1))
midAngle = np.zeros((Nsteps, 1))

for timeStep in range(1, Nsteps):
    print(f't = {ctime}\n')
    q, error = fun( qOld = q0, 
                    dt = dt, 
                    q0 = q0, 
                    m = m, 
                    u = u, 
                    nv = nv,
                    visc = visc, 
                    R1 = R1, 
                    R2 = R2, 
                    R3 = R3,
                    mg=mg,
                    ScaleSolver = ScaleSolver,
                    tol=tol,
                    maximum_iter=maximum_iter,
                    EI=EI,
                    EA=EA,
                    deltaL=deltaL
                    )

    if error < 0:
        print('Could not converge. Sorry!\n')
        break 

    u = (q - q0) / dt   # velocity
    ctime += dt         # current time

    q0 = q              # update x0

    if ((timeStep - 1) % plotStep == 0):
        plt.clf()
        plotrod(q, ctime, fig)
        plt.show(block=False)
        plt.pause(5)
    
    all_pos[timeStep] = q[3,0]
    all_v[timeStep] = u[3,0]

    # angle at the center
    vec1 = np.array([q[2,0], q[3,0], 0]) - np.array([q[0,0], q[1,0], 0])
    vec2 = np.array([q[4,0], q[5,0], 0]) - np.array([q[2,0], q[3,0], 0])
    midAngle[timeStep] = np.arctan2(np.linalg.norm(np.cross(vec1,vec2)),np.dot(vec1,vec2)) * 180 / np.pi


