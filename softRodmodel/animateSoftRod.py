import os
import math
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def rodGIF(qDataFile, rodLength):
    '''
        Animate rod motion and save as GIF
    '''
    dt = 0.001  # Time step
    n = 1000  # Number of frames to skip

    def get_data_x(i):  # Get x data
        return x.loc[i]

    def get_data_y(i):  # Get y data
        return y.loc[i]

    def animateRod(i):  # Animate rod
        x = get_data_x(i * n)  # Get x data
        y = get_data_y(i * n)  # Get y data
        fixedEnd.set_data(x[0], y[0])  # Set fixed end
        rod.set_data(x - x[0], y - y[0])  # Set rod
        frame_text.set_text(r'$Time = %.1f$' % i)  # Set time
        return fixedEnd, rod, frame_text

    qData = pd.read_csv(qDataFile)  # Read qData file
    qData = qData.loc[:, qData.columns != 'time']  # Remove time column
    nFrames = len(qData)  # Number of frames

    x = qData.iloc[:, 0::2]  # Get x data
    y = qData.iloc[:, 1::2]  # Get y data

    fig, ax = plt.subplots(figsize=(5, 5))

    rod, = ax.plot([], [], '-ro')  # Plot rod
    fixedEnd, = ax.plot([], [], 'ko')  # Plot fixed end
    frame_text = ax.text(0.02, 0.92, '', transform=ax.transAxes)  # Set time text

    ax.set(xlabel=r'$x~[cm]$', ylabel=r'$y~[cm]$')
    ax.set(xlim=[-(rodLength + 1), (rodLength + 1)], ylim=[-(rodLength + 1), (rodLength + 1)])
    ax.grid('both', linestyle='--', linewidth=1)

    animRod = FuncAnimation(fig, animateRod, frames=nFrames // n, interval=1000, blit=True)

    gifFile = qDataFile.replace('qData_', 'rod_')
    animRod.save(gifFile.replace('.csv', '.gif'), dpi=150, writer=PillowWriter(fps=10))


path = 'data/nv13'  # Path to data folder
qDataFile = glob.glob(os.path.join(path, '*qData*'))[0]  # Get qData file
rodLength = int(qDataFile.split('_')[1][1:])  # Get rod length

rodGIF(qDataFile, rodLength)  # Animate rod
