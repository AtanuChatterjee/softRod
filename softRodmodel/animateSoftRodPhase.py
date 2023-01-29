import os
import glob
import math
import numpy as np
import pandas as pd
from analysisSoftRod import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def rodGIF(qDataFile, nodek, rodLength):
    dt = 0.001
    n = 1000

    def get_data_x(i):
        return x.loc[i]

    def get_data_y(i):
        return y.loc[i]

    def animateRod(i):
        x = get_data_x(i * n)
        y = get_data_y(i * n)
        fixedEnd.set_data(x[0], y[0])
        phaseNode.set_data(x[nodek], y[nodek])
        rod.set_data(x - x[0], y - y[0])
        frame_text.set_text(r'$Time = %.1f$' % i)
        return fixedEnd, phaseNode, rod, frame_text

    qData = pd.read_csv(qDataFile)
    qData = qData.loc[:, qData.columns != 'time']
    nFrames = len(qData)

    x = qData.iloc[:, 0::2]  # Get x data
    y = qData.iloc[:, 1::2]  # Get y data

    Rod, ax = plt.subplots(figsize=(5, 5))

    rod, = ax.plot([], [], '-ro')  # Plot rod
    phaseNode, = ax.plot([], [], 'bo')  # Plot phase node
    fixedEnd, = ax.plot([], [], 'ko')  # Plot fixed end
    frame_text = ax.text(0.02, 0.92, '', transform=ax.transAxes)  # Set time text

    ax.set(xlabel=r'$x~[cm]$', ylabel=r'$y~[cm]$')
    ax.set(xlim=[-(rodLength + 1), (rodLength + 1)], ylim=[-(rodLength + 1), (rodLength + 1)])
    ax.grid('both', linestyle='--', linewidth=1)

    animRod = FuncAnimation(Rod, animateRod, frames=nFrames // n, interval=1000, blit=True)
    return animRod


def phaseGIF(df):
    dt = 0.001
    n = 1000

    def animatePhase(i):
        phase.set_data(theta[:i * n], vel[:i * n])
        return phase

    theta, vel = df.theta, df.vel
    nFrames = len(df)

    Phase, ax = plt.subplots(figsize=(5, 5))

    phase, = ax.plot([], [], 'bo', markersize=0.1)
    ax.plot(theta, vel, 'grey', alpha=0.2)
    ax.set(xlabel=r'$\theta~[deg]$', ylabel=r'$v~[cm/sec]$')
    ax.grid('both', linestyle='--', linewidth=1)
    animPhase = FuncAnimation(Phase, animatePhase, frames=nFrames // n, interval=1000, blit=False)
    return animPhase


def animateSoftRodPhase(qDataFile, qDotDataFile, nodek, rodLength):
    animRod = rodGIF(qDataFile, nodek, rodLength)
    gifFile = qDataFile.replace('qData_', 'rodNodek_')
    animRod.save(gifFile.replace('.csv', '.gif'), dpi=150, writer=PillowWriter(fps=10))

    df = PosVelNodek(qDataFile, qDotDataFile, nodek)
    animPhase = phaseGIF(df)
    gifFile = qDataFile.replace('qData_', 'phaseNodek_')
    animPhase.save(gifFile.replace('.csv', '.gif'), dpi=150, writer=PillowWriter(fps=10))


path = 'data/nv13'  # Path to data folder
qDataFile = glob.glob(os.path.join(path, '*qData*'))[0]  # Get qData file
qDotDataFile = glob.glob(os.path.join(path, '*qDotData*'))[0]  # Get qDotData file
rodLength = int(qDataFile.split('_')[1][1:])  # Get rod length

animateSoftRodPhase(qDataFile, qDotDataFile, 3, rodLength)
