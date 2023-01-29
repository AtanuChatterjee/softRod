import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def createBeam(nv, w, t, l, Y):
    ne = nv - 1
    EI = Y * w * t ** 3 / 12
    EA = Y * w * t
    # EI = Y * np.pi * (R ** 4 - r ** 4) / 4
    # EA = Y * np.pi * (R ** 2 - r ** 2)

    nodes = np.zeros((nv, 2))

    for c in range(0, nv):
        nodes[c, 0] = c * l / ne

    refLen = np.zeros((ne, 1))

    for k in range(0, ne):
        dx = nodes[k + 1, :] - nodes[k, :]
        refLen[k] = np.linalg.norm(dx)

    return EA, EI, nodes, refLen


def setSimulation(totalTime):
    dt = 1e-5
    Nsteps = round(totalTime / dt)
    plotStep = 1e3
    return dt, plotStep, Nsteps


def getStateVectors(nv, nodes, mass, g, d, p):
    q0 = np.zeros((2 * nv, 1))

    for c in range(0, nv):
        q0[2 * c] = nodes[c, 0]
        q0[2 * c + 1] = nodes[c, 1]

    m = np.zeros((2 * nv, 1))

    for k in range(0, nv):
        m[2 * k:2 * k + 2, 0] = mass / (nv-1)

    mg = np.zeros((2 * nv, 1))

    for k in range(0, nv):
        mg[2 * k:2 * k + 2, 0] = mass * g / (nv-1)

    ################################# point force #######################################
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    c = find_nearest(nodes[:, 0], d)

    Fp = np.zeros((2 * nv, 1))
    Fp[2 * c:2 * c + 2, 0] = p

    return q0, m, mg, Fp


def getDataArrays(nv, Nsteps):
    '''
    Obtain empty arrays to populate position and velocity of each node as a function of time
        :param nv: number of node vertices
        :param Nsteps: number of simulation steps
        :return: two empty arrays of size (Nsteps, 2 * nv)
    '''

    all_q= np.zeros((Nsteps, 2 * nv))

    return all_q


def fun(qOld, dt, q0, m, u, nv, mg, Fp,
        EI,
        EA,
        refLen
        ):
    Fb = getFb(qOld, EI, refLen, nv)
    Fs = getFs(qOld, EA, refLen, nv)
    Fg = getFg(mg)
    Fp = getFp(Fp)

    #   Equation of motion
    f = m * (qOld - q0) / dt ** 2 - m * u / dt - (Fb + Fs + Fg + Fp)

    #   Update
    qNew = q0 + ((m * u / dt + Fb + Fs + Fg + Fp) * dt ** 2) / m
    qNew[0: 2, 0] = q0[0: 2, 0]
    qNew[2 * nv - 2: 2 * nv, 0] = q0[2 * nv - 2: 2 * nv, 0]
    return qNew


def getFb(q, EI, refLen, nv):
    Fb = q * 0

    for k in range(0, nv - 2):
        gradEnergy = gradEb(q[2 * k, 0], q[2 * k + 1, 0], q[2 * k + 2, 0], q[2 * k + 3, 0], q[2 * k + 4, 0],
                            q[2 * k + 5, 0], 0, refLen[k], EI)
        Fb[2 * k:2 * k + 6] = Fb[2 * k:2 * k + 6] - gradEnergy

    return Fb


def getFp(Fp):
    return Fp

def getFg(mg):
    return mg

def getFs(q, EA, refLen, nv):
    Fs = q * 0

    for k in range(0, nv - 1):
        gradEnergy = gradEs(q[2 * k], q[2 * k + 1], q[2 * k + 2], q[2 * k + 3], refLen[k], EA)
        Fs[2 * k:2 * k + 4] = Fs[2 * k:2 * k + 4] - gradEnergy

    return Fs


def gradEb(xkm1, ykm1, xk, yk, xkp1, ykp1, curvature0, lk, EI):
    node0 = np.array([xkm1, ykm1, 0])
    node1 = np.array([xk, yk, 0])
    node2 = np.array([xkp1, ykp1, 0])

    m2e = np.array([0, 0, 1])
    m2f = np.array([0, 0, 1])
    kappaBar = curvature0

    #   Computation of gradient of the two curvatures
    gradKappa = np.zeros((6, 1))

    ee = node1 - node0
    ef = node2 - node1

    norm_e = np.linalg.norm(ee)
    norm_f = np.linalg.norm(ef)

    te = ee / norm_e
    tf = ef / norm_f

    #   Curvature binormal

    if (1.0 + np.dot(te, tf)) == 0:
        kb = 2.0 * np.cross(te, tf) / (1.0 + np.dot(te, tf) + 1e34)
        chi = 1.0 + np.dot(te, tf) + 1e34

    else:
        kb = 2.0 * np.cross(te, tf) / (1.0 + np.dot(te, tf))
        chi = 1.0 + np.dot(te, tf)

    tilde_t = (te + tf) / chi
    tilde_d2 = (m2e + m2f) / chi

    #   Curvatures
    kappa1 = kb[2]
    # 0.5 * np.dot( kb, m2e + m2f)

    Dkappa1De = 1.0 / norm_e * (-kappa1 * tilde_t + np.cross(tf, tilde_d2))
    Dkappa1Df = 1.0 / norm_f * (-kappa1 * tilde_t - np.cross(te, tilde_d2))

    gradKappa[0:2, 0] = -Dkappa1De[0:2]
    gradKappa[2:4, 0] = Dkappa1De[0:2] - Dkappa1Df[0:2]
    gradKappa[4:6, 0] = Dkappa1Df[0:2]

    #   Gradient of Eb
    dkappa = kappa1 - kappaBar
    dF = gradKappa * EI * dkappa / lk

    return dF


def gradEs(xk, yk, xkp1, ykp1, lk, EA):
    """
        This function returns the derivative of stretching energy E_k^s with
        respect to x_{k-1}, y_{k-1}, x_k, and y_k.
    """
    F = np.zeros((4, 1))

    F[0] = -(1 - np.sqrt((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / lk) / (
            np.sqrt((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) * lk) * 2 * (- xkp1 + xk)
    F[1] = -(1 - np.sqrt((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / lk) / (
            np.sqrt((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) * lk) * 2 * (- ykp1 + yk)
    F[2] = -(1 - np.sqrt((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / lk) / (
            np.sqrt((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) * lk) * 2 * (xkp1 - xk)
    F[3] = -(1 - np.sqrt((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / lk) / (
            np.sqrt((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) * lk) * 2 * (ykp1 - yk)

    F = 0.5 * EA * lk * F

    return F


def plotrod(x, currTime, l, fig):
    x1 = x[0:len(x):2]  # x[0:::2]
    x2 = x[1:len(x):2]  # x[1:::2]

    plt.plot(x1, x2, 'ko-')
    plt.xlabel(r'$x~[m]$')
    plt.ylabel(r'$y~[m]$')
    plt.ylim([-0.1 * l, 0.1 * l])
    plt.xlim([- 0.02 * l, l * 1.02])
    plt.title(f'$t = {currTime:.2f}$')
    plt.tight_layout()

def createDataFrame(x):
    ''' This function creates a data frame '''
    y = pd.DataFrame(x)
    y['time'] = range(0, len(y))
    y.columns = [f'{e}' for i, e in enumerate(y.columns, 1)]
    return y

def createGIF(qData, n):
    ''' This function creates a GIF animation from the position time data '''

    def get_data_x(i):
        return x.loc[i]

    def get_data_y(i):
        return y.loc[i]

    def animate(i):
        x = get_data_x(i * n)
        y = get_data_y(i * n)
        fixedEnd1.set_data(x[0], y[0])
        fixedEnd2.set_data(x[-1], y[-1])
        beam.set_data(x, y)
        frame_text.set_text(r'$Time = %.1f$' % i)
        return beam, frame_text

    qData = qData.loc[:, qData.columns != 'time']
    nFrames = len(qData)

    x = qData.iloc[:, 0::2]
    y = qData.iloc[:, 1::2]

    fig, ax = plt.subplots(figsize=(5, 5))

    beam, = ax.plot([], [], '-ro')
    fixedEnd1, = ax.plot([], [], 'go')
    fixedEnd2, = ax.plot([], [], 'bo')
    frame_text = ax.text(0.02, 0.92, '', transform=ax.transAxes)

    ax.set_xlabel(r'$x~[m]$')
    ax.set_ylabel(r'$y~[m]$')
    ax.set_xlim([-0.25, 1.25])
    ax.set_ylim([-0.25, 0.25])
    ax.grid('both', linestyle='--', linewidth=1)

    anim = FuncAnimation(fig, animate, frames=nFrames // n, interval=1, blit=False)
    anim.save('beam_N_15.gif', dpi=150, writer=PillowWriter(fps=60))
