import os
import math
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def PosVelNodek(qDataFile, qDotDataFile, nodek):
    '''
    This function obtains the angular position and velocity of a specific selected node as a function of time
        :param qDataFile: qData file
        :param qDotDataFile: qDotData file
        :param nodek: Node number
        :return: Dataframe with time, theta, vel
    '''
    qData = pd.read_csv(qDataFile,
                        usecols=['time', 'node{}x'.format(nodek), 'node{}y'.format(nodek)])  # Read qData file
    qDotData = pd.read_csv(qDotDataFile,
                           usecols=['time', 'node{}x'.format(nodek), 'node{}y'.format(nodek)])  # Read qDotData file
    t = qData['time']  # Get time
    x, xDot = qData['node{}x'.format(nodek)], qDotData['node{}x'.format(nodek)]  # Get x and xDot data
    y, yDot = qData['node{}y'.format(nodek)], qDotData['node{}y'.format(nodek)]  # Get y and yDot data

    df = pd.concat([t, x, y, xDot, yDot], axis=1, keys=['t', 'x', 'y', 'xDot', 'yDot'])  # Create dataframe
    thetak = np.rad2deg(np.arctan(df['y'] / df['x']))  # Calculate theta
    velk = []
    for index, row in df.iterrows():
        q_perpendicular = np.array([-row['y'], row['x']])  # Calculate q_perpendicular
        if np.linalg.norm(q_perpendicular) == 0:
            chi = np.linalg.norm(q_perpendicular) + 1e34
        else:
            chi = np.linalg.norm(q_perpendicular)

        unit = q_perpendicular / chi  # Calculate unit vector

        velk_ = np.dot(unit, [row['xDot'], row['yDot']])  # Calculate velocity

        velk.append(velk_)

    velk = np.array(velk)  # Convert to numpy array

    dfNew = pd.DataFrame({'time': t, 'theta': thetak, 'vel': velk})  # Create new dataframe
    dfNew = dfNew.dropna()
    return dfNew


def plotPosVelNodek(df, nodek):
    '''
    This function plots the angular position and velocity of a specific selected node as a function of time
        :param df: Dataframe with time, theta, vel
        :param nodek: Node number
        :return: A figure of the position and velocity of nodek
    '''
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    fig.suptitle(r'$Node~{}$'.format(nodek))  # Set title
    ax0 = ax[0].twinx()  # Create twin axis

    ax[0].plot(df['time'], df['theta'], 'r')  # Plot theta
    ax[0].yaxis.label.set_color('red')  # Set color of y-axis label

    ax0.plot(df['time'], df['vel'], 'g')  # Plot velocity
    ax0.yaxis.label.set_color('green')  # Set color of y-axis label
    ax[0].set(ylabel=r'$\theta~[deg]$', xlabel=r'$time~[sec]$')
    ax0.set(ylabel=r'$v~[cm/sec]$')
    ax[0].grid('both', linestyle='--', linewidth=1)

    ax[1].plot(df['theta'], df['vel'], 'orange')  # Plot theta vs velocity
    ax[1].set(ylabel=r'$v~[cm/sec]$', xlabel=r'$\theta~[deg]$')
    ax[1].grid('both', linestyle='--', linewidth=1)

    plt.tight_layout()
    pdfFile = qDataFile.replace('qData_', 'node{}_'.format(nodek))  # Create pdf file
    plt.savefig(pdfFile.replace('.csv', '.pdf'), bbox_inches='tight')  # Save figure
    plt.show()


def PosnNodes(qDataFile, nv, d):
    '''
    This function obtains the angular position of two nodes at a distance d from the central node (k-1, k, k+1)
        :param qDataFile: qData file
        :param nv: Number of nodes
        :param d: Distance from the central node
        :return: Dataframe with time, theta
    '''
    qData = pd.read_csv(qDataFile)  # Read qData file
    time = qData['time']  # Get time

    dfn = []  # Create empty list

    cnode = (nv - 1) // 2  # Calculate central node
    nodes = [cnode - d, cnode, cnode + d]  # Calculate nodes at a distance d from the central node
    for n in nodes:
        xn, yn = qData['node{}x'.format(n)], qData['node{}y'.format(n)]  # Get x and y data
        thetan = np.rad2deg(np.arctan(yn / xn))  # Calculate theta
        dfn_ = pd.DataFrame({'Node' + str(n): thetan})  # Create dataframe
        dfn.append(dfn_)

    dfnNodes = pd.concat(dfn, axis=1)
    dfnNodes = dfnNodes.dropna()
    dfnNodes = pd.concat([time, dfnNodes], axis=1)  # Create dataframe
    return dfnNodes


def plotPosnNodes(df):
    '''
    This function plots the angular position of the nodes at a distance d from the central node
        :param df: Dataframe with time, theta
        :return: A figure of the angular position of the nodes at a distance d from the central node
    '''
    fig, ax = plt.subplots(figsize=(8, 3))

    nodes = df.columns[1:]  # Get nodes
    colors = plt.cm.rainbow(np.linspace(0, 1, len(nodes)))  # Set colors
    for i, node in enumerate(nodes):
        ax.plot(df['time'], df[node], label=r'$node$' + str(i + 1), color=colors[i])  # Plot theta
    ax.set(ylabel=r'$\theta~[deg]$', xlabel=r'$time~[sec]$')
    ax.grid('both', linestyle='--', linewidth=1)
    plt.legend(loc='lower left', frameon=False)
    plt.tight_layout()
    pdfFile = qDataFile.replace('qData_', 'nNodes_')  # Create pdf file
    plt.savefig(pdfFile.replace('.csv', '.pdf'), bbox_inches='tight')  # Save figure
    plt.show()


def getPosVelNodek(qDataFile, qDotDataFile, nodek):
    '''
    This function obtains the position and velocity of nodek
        :param qDataFile: qData file
        :param qDotDataFile: qDotData file
        :param nodek: Node number
        :return: A figure of the position and velocity of nodek, a dataframe with time, theta, vel, and a csv file
    '''
    df = PosVelNodek(qDataFile, qDotDataFile, nodek)
    fig = plotPosVelNodek(df, nodek)
    csvFile = qDataFile.replace('qData_', 'node{}_'.format(nodek))  # Create csv file
    df.to_csv(csvFile, index=False)  # Save dataframe to csv file
    return fig, df


def getPosnNodes(qDataFile, nv, d):
    '''
    This function obtains the angular position of the nodes at a distance d from the central node
        :param qDataFile: qData file
        :param nv: Number of nodes
        :param d: Distance from the central node
        :return: A figure of the angular position of the nodes at a distance d from the central node, a dataframe with time, theta, and a csv file
    '''

    df = PosnNodes(qDataFile, nv, d)
    fig = plotPosnNodes(df)
    csvFile = qDataFile.replace('qData_', 'nNodes_')  # Create csv file
    df.to_csv(csvFile, index=False)  # Save dataframe to csv file
    return fig, df


def travelingWave(qDotDataFile):
    '''
    This function obtains the traveling wave of the rod
        :param qDotDataFile: qDotData file
        :return: A figure of the traveling wave of the rod, a dataframe with time, theta, and a csv file
    '''

    # Set the time step size
    dt = 0.001
    qDotData = pd.read_csv(qDotDataFile)  # Read qData file
    qDotCols = [col for col in qDotData.columns if col.startswith('node')]  # Get node columns
    qDot = qDotData[qDotCols].to_numpy()  # Get qDot data

    # Plot the velocities
    plt.imshow(qDot, aspect='auto', origin='lower', extent=[0, qDot.shape[0] * dt, 0, qDot.shape[1] / 2])  # Plot qDot
    plt.colorbar(label=r'$vel~[cm/sec]$')  # Set colorbar label
    plt.xlabel(r'$time~[sec]$')
    plt.ylabel(r'$node$')
    plt.tight_layout()
    pdfFile = qDotDataFile.replace('qDotData_', 'travelingWave')  # Create pdf file
    plt.savefig(pdfFile.replace('.csv', '.pdf'), bbox_inches='tight')  # Save figure
    plt.show()


path = 'data/nv13'  # Path to data folder
qDataFile = glob.glob(os.path.join(path, '*qData*'))[0]  # Get qData file
qDotDataFile = glob.glob(os.path.join(path, '*qDotData*.csv'))[0]  # Get qDotData file

rodLength = int(qDataFile.split('_')[1][1:])  # Get rod length
nv = int(qDataFile.split('_')[2][2:-4])  # Get number of nodes

# getPosVelNodek(qDataFile, qDotDataFile, 3)
# getPosnNodes(qDataFile, nv, 1)
# travelingWave(qDotDataFile)
