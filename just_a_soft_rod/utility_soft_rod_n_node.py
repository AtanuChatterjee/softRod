import os
import sys
import numpy as np
import matplotlib.pyplot as plt


def crossMat(a):
    A = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])

    return A


def fun(qOld, dt, q0, m, u, nv, visc, r, mg,
        ScaleSolver,
        tol,
        maximum_iter,
        EI,
        EA,
        deltaL
        ):
    #   mass matrix
    mMat = np.diag(m)
    qNew = qOld

    #   Newton-Raphson scheme
    iter = 0
    normf = tol * ScaleSolver * 10
    error = 1

    while (normf > tol * ScaleSolver):

        #   forces
        Fb, Jb = getFb(qNew, EI, deltaL, nv)
        Fs, Js = getFs(qNew, EA, deltaL, nv)
        Fv, Jv = getFv(qNew, nv, visc, q0, dt, r)
        Fg = getFg(mg)

        #   Equation of motion
        f = m * (qNew - q0) / dt ** 2 - m * u / dt - (Fb + Fs + Fv + Fg)

        #   Manipulate Jacobians
        J = mMat / dt ** 2 - (Jb + Js + Jv)

        #   Newton's Update
        qNew = qNew - np.matmul(np.linalg.inv(J), f)
        normfNew = np.linalg.norm(f)

        iter = iter + 1
        print(f'Iter = {iter - 1}, error = {normfNew}\n')

        normf = normfNew

        if (iter > maximum_iter):
            error = -1
            return qNew, error
    return qNew, error


def getFb(q, EI, deltaL, nv):
    Fb = q * 0
    Jb = np.zeros((len(q), len(q)))

    for k in range(0, nv - 2):
        gradEnergy = gradEb(q[2 * k, 0], q[2 * k + 1, 0], q[2 * k + 2, 0], q[2 * k + 3, 0], q[2 * k + 4, 0],
                            q[2 * k + 5, 0], 0, deltaL, EI)
        Fb[2 * k:2 * k + 6] = Fb[2 * k:2 * k + 6] - gradEnergy

        hessEnergy = hessEb(q[2 * k, 0], q[2 * k + 1, 0], q[2 * k + 2, 0], q[2 * k + 3, 0], q[2 * k + 4, 0],
                            q[2 * k + 5, 0], 0, deltaL, EI)
        Jb[2 * k:2 * k + 6, 2 * k:2 * k + 6] = Jb[2 * k:2 * k + 6, 2 * k:2 * k + 6] - hessEnergy

    return Fb, Jb


def getFg(mg):
    return mg


def getFs(q, EA, deltaL, nv):
    Fs = q * 0
    Js = np.zeros((len(q), len(q)))

    for k in range(0, nv - 1):
        gradEnergy = gradEs(q[2 * k], q[2 * k + 1], q[2 * k + 2], q[2 * k + 3], deltaL, EA)
        Fs[2 * k:2 * k + 4] = Fs[2 * k:2 * k + 4] - gradEnergy
        hessEnergy = hessEs(q[2 * k], q[2 * k + 1], q[2 * k + 2], q[2 * k + 3], deltaL, EA)
        Js[2 * k:2 * k + 4, 2 * k:2 * k + 4] = Js[2 * k:2 * k + 4, 2 * k:2 * k + 4] - hessEnergy

    return Fs, Js


def getFv(q, nv, visc, q0, dt, r):
    # Compute stretching force
    Fv = - 6 * np.pi * visc * (q - q0) / dt

    for k in range(0, nv):
        Fv[2 * k:2 * k + 2] = r[k] * Fv[2 * k:2 * k + 2]

    # Compute the jacobian of the stretching force
    Jv = - 6 * np.pi * visc / dt * np.eye(len(q))

    for k in range(0, nv):
        Jv[2 * k:2 * k + 2, 2 * k:2 * k + 2] = r[k] * Jv[2 * k:2 * k + 2, 2 * k:2 * k + 2]

    return Fv, Jv


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


def hessEb(xkm1, ykm1, xk, yk, xkp1, ykp1, curvature0, lk, EI):
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

    # Curvature binormal
    if (1.0 + np.dot(te, tf)) == 0:
        kb = 2.0 * np.cross(te, tf) / (1.0 + np.dot(te, tf) + 1e34)
        chi = 1.0 + np.dot(te, tf) + 1e34
    else:
        kb = 2.0 * np.cross(te.T, tf.T) / (1.0 + np.dot(te, tf))
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
    DDkappa1 = np.zeros((6, 6))

    norm2_e = norm_e ** 2
    norm2_f = norm_f ** 2

    tilde_t_transpose = tilde_t[:, np.newaxis]
    tt_o_tt = tilde_t_transpose * tilde_t

    tmp = np.cross(tf, tilde_d2)
    tmp_transpose = tmp[:, np.newaxis]
    tf_c_d2t_o_tt = tmp_transpose * tilde_t

    tt_o_tf_c_d2t = np.transpose(tf_c_d2t_o_tt)
    kb_transpose = kb[:, np.newaxis]
    kb_o_d2e = kb_transpose * m2e
    d2e_o_kb = np.transpose(kb_o_d2e)

    Id3 = np.identity(3)
    te_transpose = te[:, np.newaxis]
    D2kappa1De2 = 1.0 / norm2_e * (2 * kappa1 * tt_o_tt - tf_c_d2t_o_tt - tt_o_tf_c_d2t) - kappa1 / (chi * norm2_e) * (
            Id3 - te_transpose * te) + 1.0 / (4.0 * norm2_e) * (kb_o_d2e + d2e_o_kb)

    tmp = np.cross(te, tilde_d2)
    tmp_transpose = tmp[:, np.newaxis]
    te_c_d2t_o_tt = tmp_transpose * tilde_t

    tt_o_te_c_d2t = np.transpose(te_c_d2t_o_tt)
    kb_o_d2f = kb_transpose * m2f
    d2f_o_kb = np.transpose(kb_o_d2f)

    tf_transpose = tf[:, np.newaxis]

    D2kappa1Df2 = 1.0 / norm2_f * (2 * kappa1 * tt_o_tt + te_c_d2t_o_tt + tt_o_te_c_d2t) - kappa1 / (chi * norm2_f) * (
            Id3 - tf_transpose * tf) + 1.0 / (4.0 * norm2_f) * (kb_o_d2f + d2f_o_kb)

    D2kappa1DeDf = -kappa1 / (chi * norm_e * norm_f) * (Id3 + te_transpose * tf) + 1.0 / (norm_e * norm_f) * (
            2 * kappa1 * tt_o_tt - tf_c_d2t_o_tt + tt_o_te_c_d2t - crossMat(tilde_d2))

    D2kappa1DfDe = np.transpose(D2kappa1DeDf)

    #   Curvature terms
    DDkappa1[0: 2, 0: 2] = D2kappa1De2[0: 2, 0: 2]
    DDkappa1[0: 2, 2: 4] = - D2kappa1De2[0: 2, 0: 2] + D2kappa1DeDf[0: 2, 0: 2]
    DDkappa1[0: 2, 4: 6] = - D2kappa1DeDf[0: 2, 0: 2]
    DDkappa1[2: 4, 0: 2] = - D2kappa1De2[0: 2, 0: 2] + D2kappa1DfDe[0: 2, 0: 2]
    DDkappa1[2: 4, 2: 4] = D2kappa1De2[0: 2, 0: 2] - D2kappa1DeDf[0: 2, 0: 2] - D2kappa1DfDe[0: 2, 0: 2] + D2kappa1Df2[
                                                                                                           0: 2, 0: 2]
    DDkappa1[2: 4, 4: 6] = D2kappa1DeDf[0: 2, 0: 2] - D2kappa1Df2[0: 2, 0: 2]
    DDkappa1[4: 6, 0: 2] = - D2kappa1DfDe[0: 2, 0: 2]
    DDkappa1[4: 6, 2: 4] = D2kappa1DfDe[0: 2, 0: 2] - D2kappa1Df2[0: 2, 0: 2]
    DDkappa1[4: 6, 4: 6] = D2kappa1Df2[0: 2, 0: 2]

    #   Hessian of Eb
    dkappa = kappa1 - kappaBar
    dJ = 1.0 / lk * EI * gradKappa * np.transpose(gradKappa)
    temp = 1.0 / lk * dkappa * EI
    dJ = dJ + temp * DDkappa1
    return dJ


def hessEs(xk, yk, xkp1, ykp1, lk, EA):
    J = np.zeros((4, 4))

    J[0, 0] = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / lk ** 2 * (-2 * xkp1 + 2 * xk) ** 2) / 2 + (
            1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / lk) * (
                      ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-3 / 2)) / lk * (
                      (-2 * xkp1 + 2 * xk) ** 2) / 2 - 2 * (
                      1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / lk) * (
                      ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-1 / 2)) / lk
    J[0, 1] = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / lk ** 2 * (-2 * ykp1 + 2 * yk) * (
            -2 * xkp1 + 2 * xk)) / 2 + (
                      1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / lk) * (
                      ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-3 / 2)) / lk * (-2 * xkp1 + 2 * xk) * (
                      -2 * ykp1 + 2 * yk) / 2
    J[0, 2] = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / lk ** 2 * (2 * xkp1 - 2 * xk) * (-2 * xkp1 + 2 * xk)) / 2 + (
            1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / lk) * (
                      ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-3 / 2)) / lk * (-2 * xkp1 + 2 * xk) * (
                      2 * xkp1 - 2 * xk) / 2 + 2 * (1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / lk) * (
                      ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-1 / 2)) / lk
    J[0, 3] = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / lk ** 2 * (2 * ykp1 - 2 * yk) * (-2 * xkp1 + 2 * xk)) / 2 + (
            1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / lk) * (
                      ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-3 / 2)) / lk * (-2 * xkp1 + 2 * xk) * (
                      2 * ykp1 - 2 * yk) / 2
    J[1, 1] = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / lk ** 2 * (-2 * ykp1 + 2 * yk) ** 2) / 2 + (
            1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / lk) * (
                      ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-3 / 2)) / lk * (
                      (-2 * ykp1 + 2 * yk) ** 2) / 2 - 2 * (
                      1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / lk) * (
                      ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-1 / 2)) / lk
    J[1, 2] = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / lk ** 2 * (2 * xkp1 - 2 * xk) * (-2 * ykp1 + 2 * yk)) / 2 + (
            1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / lk) * (
                      ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-3 / 2)) / lk * (-2 * ykp1 + 2 * yk) * (
                      2 * xkp1 - 2 * xk) / 2
    J[1, 3] = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / lk ** 2 * (2 * ykp1 - 2 * yk) * (-2 * ykp1 + 2 * yk)) / 2 + (
            1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / lk) * (
                      ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-3 / 2)) / lk * (-2 * ykp1 + 2 * yk) * (
                      2 * ykp1 - 2 * yk) / 2 + 2 * (1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / lk) * (
                      ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-1 / 2)) / lk
    J[2, 2] = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / lk ** 2 * (2 * xkp1 - 2 * xk) ** 2) / 2 + (
            1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / lk) * (
                      ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-3 / 2)) / lk * (
                      (2 * xkp1 - 2 * xk) ** 2) / 2 - 2 * (
                      1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / lk) * (
                      ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-1 / 2)) / lk
    J[2, 3] = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / lk ** 2 * (2 * ykp1 - 2 * yk) * (2 * xkp1 - 2 * xk)) / 2 + (
            1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / lk) * (
                      ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-3 / 2)) / lk * (2 * xkp1 - 2 * xk) * (
                      2 * ykp1 - 2 * yk) / 2
    J[3, 3] = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / lk ** 2 * (2 * ykp1 - 2 * yk) ** 2) / 2 + (
            1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / lk) * (
                      ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-3 / 2)) / lk * (
                      (2 * ykp1 - 2 * yk) ** 2) / 2 - 2 * (
                      1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / lk) * (
                      ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-1 / 2)) / lk
    #
    J = np.array([[J[0, 0], J[0, 1], J[0, 2], J[0, 3]],
                  [J[0, 1], J[1, 1], J[1, 2], J[1, 3]],
                  [J[0, 2], J[1, 2], J[2, 2], J[2, 3]],
                  [J[0, 3], J[1, 3], J[2, 3], J[3, 3]]])

    J = 0.5 * EA * lk * J

    return J


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

