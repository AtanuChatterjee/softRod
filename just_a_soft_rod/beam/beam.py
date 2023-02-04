import numpy as np
import pandas as pd
from beam_utility import *


def main(nv, w, t, l, Y, mass, g, d, p, totalTime):
    ''' This function obtains the geometry, state vectors and sets the simulation '''
    EA, EI, nodes, dL = createBeam(nv, w, t, l, Y)  # create a rod and divide it into finite elements
    q0, m, mg, Fp = getStateVectors(nv, nodes, mass, g, d, p)  # obtain state vectors q
    dt, plotStep, Nsteps = setSimulation(totalTime)  # set the simulation
    all_q = getDataArrays(nv, Nsteps)  # get data for each node at every time step

    q = q0
    ################# Rate of change #################################################
    u = (q - q0) / dt

    ctime = 0

    fig = plt.figure(figsize=(5, 5))

    for timeStep in range(2, Nsteps):
        print(f't = {ctime}\n')
        q = fun(qOld=q0,
                dt=dt,
                q0=q0,
                m=m,
                u=u,
                nv=nv,
                mg=mg,
                Fp=Fp,
                EI=EI,
                EA=EA,
                refLen=dL
                )

        u = (q - q0) / dt  # velocity
        ctime += dt  # current time

        q0 = q  # update x0

        all_q[timeStep - 1, :] = q.T

        if ((timeStep - 1) % plotStep == 0):
            plt.clf()
            plotrod(q, ctime, l, fig)
            plt.show(block=False)
            plt.pause(1)

    # qData = createDataFrame(all_q)
    # qData.to_csv('qData_N_15.csv', index=False)
    # createGIF(qData, 100)

    print('Done!')


if __name__ == '__main__':
    nv = 5
    main(nv, w=14.5e-3, t=0.54e-3, l=10e-2, Y=0.5e6, mass=1.10e-3, g=np.array([0, -9.81]), d=0.75, p=np.array([0, 0]), totalTime=1)
