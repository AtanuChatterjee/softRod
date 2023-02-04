import os
import math
import numpy as np
import pandas as pd
from utilitySoftRod import *
import matplotlib.pyplot as plt


def main(nv, bnode, totalTime, Np, Ng, l, w, t, Y, gamma, dNp0, nestDir, kc, F_ind, f0, alpha):
    '''
        This function obtains the geometry, state vectors and sets the simulation parameters
    '''
    # ring = Ring(nv, w, t, l, Y, dNp=dNp0)  # create ring object
    # EA, EI, nodes, ne, dL = ring.createRing()  # create ring geometry
    # q0, dNp0 = ring.getStateVectors()  # get initial state vectors

    rod = Rod(nv, w, t, l, Y, dNp=dNp0)  # create rod object
    EA, EI, nodes, ne, dL = rod.createRod()  # create rod geometry
    q0, dNp0 = rod.getStateVectors()  # get initial state vectors

    sim = Simulation(totalTime, nv)  # create simulation object
    dt, plotStep, Nsteps = sim.setSimulation()  # set simulation parameters
    all_q, all_qDot = sim.getDataArrays(Nsteps)  # create data arrays

    ################# Assigning initial vetor to be current vector (t = 0) ###########
    q = q0
    dNp = dNp0

    ################# Rate of change #################################################
    qDot = (q - q0) / dt

    params = {
        'nv': nv,  # number of node vertices
        'ne': ne,  # number of discrete units the rod is divided
        'f0': f0,  # single ant force
        'gamma': gamma,  # damping factor
        'dNp0': dNp0,  # set initial state at each node for (Np+ - Np-)
        'Np': Np,  # number of puller ants
        'Ng': Ng,  # number of informed ants
        'nestDir': nestDir,  # orientation of the nest
        'dt': dt,  # simulation timestep (default dt=1e-3)
        'boundary_node': bnode,  # node to set boundary condition
        'alpha': alpha,  # factor to scale spring force
        'dL': dL,
        # undeformed edge length between a pair of nodes to be taken as reference length for elastic force calculation
        'kc': kc,  # rate constant to switch direction of pulling (default kc=0.7)
        'F_ind': F_ind,  # individuality parameter (default F_ind=0.428)
        'l': l,  # rod length/ring diameter
        'EI': EI,  # flexural rigidity
        'EA': EA  # stiffness
    }
    ############################# Plot rod ##########################################
    fig, ax = plt.subplots(figsize=(5, 5))

    ############################# Time marching #####################################
    ctime = 0  # Current time

    for timeStep in range(1, Nsteps):
        print(f't = {ctime}\n')

        constraint = Constraints(nv, 0, 0, 0)  # create constraint object

        # Create solver object
        solver = Solver(constraint)

        # Solve
        q, dNp = solver.solve(q, q0, qDot, dNp0, params=params)

        qDot = (q - q0) / dt  # update rate of change
        ctime += dt  # update current time

        q0 = q  # update current state
        dNp0 = dNp  # update current state

        all_q[timeStep - 1, :], all_qDot[timeStep - 1, :] = q.T, qDot.T  # store data

        if ((timeStep - 1) % plotStep == 0):
            plt.clf()
            solver.plot(q, ctime, l)  # plot rod
            plt.show(block=False)
            plt.pause(1)

    qData, qDotData = sim.createDataFrame(all_q, dt), sim.createDataFrame(all_qDot, dt)  # create dataframe

    directory = os.path.join('data', 'nv{}'.format(nv))  # create directory to save data
    if not os.path.exists(directory):  # create directory if it does not exist
        os.makedirs(directory)  # create directory

    qData.to_csv(os.path.join(directory, 'qData_l{}_nv{}'.format(l, nv) + '.csv'), index=False)  # save data
    qDotData.to_csv(os.path.join(directory, 'qDotData_l{}_nv{}'.format(l, nv) + '.csv'), index=False)  # save data
    print('Done with execution!')


if __name__ == '__main__':
    nv = 11  # number of node vertices
    bnode = 1  # boundary node
    totalTime = 500  # simulation duration
    Gamma = 1e-4  # damping factor

    # l, w, t, Y = 5, 0.086, 0.094, 8e6  # white rod variables [cgs units]: length, width, thickness, young's modulus
    # l, w, t, Y = 20, 0.09, 0.075, 5e6  # red rod variables [cgs units]: length, width, thickness, young's modulus

    l, w, t, Y = 10, 0.09, 0.01, 0.5e6  # red rod variables [cgs units]: length, width, thickness, young's modulus

    if l == 5:
        N = 50
        G = 0.001 * N

    elif l == 10:
        N = 100
        G = 0.001 * N

    elif l == 20:
        N = 200
        G = 0.1 * N

    Np = N / (nv - 1)
    Ng = G / (nv - 1)
    gamma = Gamma * (nv - 1)
    dNp0 = 4  # initial state of dNp
    if dNp0 >= Np or dNp0 <= -Np:
        raise ValueError('dNp0 is out of range')

    main(nv, bnode, totalTime, Np, Ng, l, w, t, Y, gamma, dNp0, nestDir=0, kc=0.7,
         F_ind=0.428, f0=17.5, alpha=1)
