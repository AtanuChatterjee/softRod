import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def fun(qOld, dt, q0, m, u, nv, visc, r, mg,
        EI,
        EA,
        refLen
        ):
    Fb = getFb(qOld, EI, refLen, nv)
    Fs = getFs(qOld, EA, refLen, nv)
    Fv = getFv(qOld, nv, visc, u, r)
    Fg = getFg(mg)

    #   Equation of motion
    f = m * (qOld - q0) / dt ** 2 - m * u / dt - (Fb + Fs + Fv + Fg)


    #   Update
    qNew = q0 + ((m * u / dt + Fb + Fs + Fv + Fg) * dt ** 2 ) / m

    return qNew


def getFb(q, EI, refLen, nv):
    Fb = q * 0
    # Jb = np.zeros((len(q), len(q)))

    for k in range(0, nv - 2):
        gradEnergy = gradEb(q[2 * k, 0], q[2 * k + 1, 0], q[2 * k + 2, 0], q[2 * k + 3, 0], q[2 * k + 4, 0],
                            q[2 * k + 5, 0], 0, refLen[k], EI)
        Fb[2 * k:2 * k + 6] = Fb[2 * k:2 * k + 6] - gradEnergy

    return Fb


def getFg(mg):
    return mg


def getFs(q, EA, refLen, nv):
    Fs = q * 0

    for k in range(0, nv - 1):
        gradEnergy = gradEs(q[2 * k], q[2 * k + 1], q[2 * k + 2], q[2 * k + 3], refLen[k], EA)
        Fs[2 * k:2 * k + 4] = Fs[2 * k:2 * k + 4] - gradEnergy

    return Fs 


def getFv(q, nv, visc, u, r):
    # Compute viscous force
    Fv = - 6 * np.pi * visc * u

    for k in range(0, nv):
        Fv[2 * k:2 * k + 2] = r[k] * Fv[2 * k:2 * k + 2]

    return Fv


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


def plotrod(x, currTime, fig):
    x1 = x[0:len(x):2]  # x[0:::2]
    x2 = x[1:len(x):2]  # x[1:::2]

    plt.plot(x1, x2, 'ko-')
    plt.xlabel(r'$x~[m]$')
    plt.ylabel(r'$y~[m]$')
    plt.ylim([-0.1, 0.01])
    # plt.xlim([0, 0.1])
    plt.title(f'$t = {currTime:.2f}$')
    plt.tight_layout()

    return x1, x2