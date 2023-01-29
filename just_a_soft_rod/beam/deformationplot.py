import os
import glob
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(5, 5))


def getYmaxTheory(P, c, l, Y, R, r):
    EI = Y * np.pi * (R ** 4 - r ** 4) / 4
    ymaxTheory = []
    for p in P:
        ymaxTheory_ = (1000 * p * c * (l ** 2 - c ** 2) ** 1.5) / (EI * l * 9 * 3 ** 0.5)
        ymaxTheory.append(ymaxTheory_)
    return ymaxTheory

def getYmax(file):
    data = pd.read_csv(file)
    data = data.iloc[:, 1::2]
    data = data.abs()
    ymax_ = data.max(axis=1)
    ymax = np.mean(ymax_)
    return ymax

P = [2, 4, 6, 8, 10]

ymax_theory = getYmaxTheory(P, 0.75, 1, 7e10, 0.013, 0.011)

ymax_N5_2 = getYmax('C:/Users/atanu/Documents/GitHub/soft_rod/soft_rod/just_a_soft_rod/beam/N_5/qData_N_5.csv')
ymax_N5_4 = getYmax('C:/Users/atanu/Documents/GitHub/soft_rod/soft_rod/just_a_soft_rod/beam/N_5/qData_N_5_4.csv')
ymax_N5_6 = getYmax('C:/Users/atanu/Documents/GitHub/soft_rod/soft_rod/just_a_soft_rod/beam/N_5/qData_N_5_6.csv')
ymax_N5_8 = getYmax('C:/Users/atanu/Documents/GitHub/soft_rod/soft_rod/just_a_soft_rod/beam/N_5/qData_N_5_8.csv')
ymax_N5_10 = getYmax('C:/Users/atanu/Documents/GitHub/soft_rod/soft_rod/just_a_soft_rod/beam/N_5/qData_N_5_10.csv')
N5 = [ymax_N5_2, ymax_N5_4, ymax_N5_6, ymax_N5_8, ymax_N5_10]

ymax_N10_2 = getYmax('C:/Users/atanu/Documents/GitHub/soft_rod/soft_rod/just_a_soft_rod/beam/N_10/qData_N_10.csv')
ymax_N10_4 = getYmax('C:/Users/atanu/Documents/GitHub/soft_rod/soft_rod/just_a_soft_rod/beam/N_10/qData_N_10_4.csv')
ymax_N10_6 = getYmax('C:/Users/atanu/Documents/GitHub/soft_rod/soft_rod/just_a_soft_rod/beam/N_10/qData_N_10_6.csv')
ymax_N10_8 = getYmax('C:/Users/atanu/Documents/GitHub/soft_rod/soft_rod/just_a_soft_rod/beam/N_10/qData_N_10_8.csv')
ymax_N10_10 = getYmax('C:/Users/atanu/Documents/GitHub/soft_rod/soft_rod/just_a_soft_rod/beam/N_10/qData_N_10_10.csv')
N10 = [ymax_N10_2, ymax_N10_4, ymax_N10_6, ymax_N10_8, ymax_N10_10]

ymax_N15_2 = getYmax('C:/Users/atanu/Documents/GitHub/soft_rod/soft_rod/just_a_soft_rod/beam/N_15/qData_N_15.csv')
ymax_N15_4 = getYmax('C:/Users/atanu/Documents/GitHub/soft_rod/soft_rod/just_a_soft_rod/beam/N_15/qData_N_15_4.csv')
ymax_N15_6 = getYmax('C:/Users/atanu/Documents/GitHub/soft_rod/soft_rod/just_a_soft_rod/beam/N_15/qData_N_15_6.csv')
ymax_N15_8 = getYmax('C:/Users/atanu/Documents/GitHub/soft_rod/soft_rod/just_a_soft_rod/beam/N_15/qData_N_15_8.csv')
ymax_N15_10 = getYmax('C:/Users/atanu/Documents/GitHub/soft_rod/soft_rod/just_a_soft_rod/beam/N_15/qData_N_15_10.csv')
N15 = [ymax_N15_2, ymax_N15_4, ymax_N15_6, ymax_N15_8, ymax_N15_10]

ymax_N20_2 = getYmax('C:/Users/atanu/Documents/GitHub/soft_rod/soft_rod/just_a_soft_rod/beam/N_20/qData_N_20.csv')
ymax_N20_4 = getYmax('C:/Users/atanu/Documents/GitHub/soft_rod/soft_rod/just_a_soft_rod/beam/N_20/qData_N_20_4.csv')
ymax_N20_6 = getYmax('C:/Users/atanu/Documents/GitHub/soft_rod/soft_rod/just_a_soft_rod/beam/N_20/qData_N_20_6.csv')
ymax_N20_8 = getYmax('C:/Users/atanu/Documents/GitHub/soft_rod/soft_rod/just_a_soft_rod/beam/N_20/qData_N_20_8.csv')
ymax_N20_10 = getYmax('C:/Users/atanu/Documents/GitHub/soft_rod/soft_rod/just_a_soft_rod/beam/N_20/qData_N_20_10.csv')
N20 = [ymax_N20_2, ymax_N20_4, ymax_N20_6, ymax_N20_8, ymax_N20_10]

ax.plot(P, ymax_theory, '--k', label=r'$theory$')
ax.plot(P, N5, '-r', label=r'$N=5$')
ax.plot(P, N10, '-g', label=r'$N=10$')
ax.plot(P, N15, '-b', label=r'$N=15$')
ax.plot(P, N20, '-y', label=r'$N=20$')
ax.set_xlabel(r'$Load,~P~[kN]$')
ax.set_ylabel(r'$Maximum~deflection,~y_{max}~[m]$')
ax.grid('both', linestyle='--', linewidth=1)
plt.legend()
plt.tight_layout()
plt.show()

