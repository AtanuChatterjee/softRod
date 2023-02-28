import pickle
import numpy as np
import pandas as pd
from math import sqrt
import scipy.signal as signal
from scipy import interpolate
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d
from matplotlib.animation import FuncAnimation, PillowWriter

def main(dataContour, dataSkeleton, g, lmax, Npoints):
    print('loading data ...')

    nFrames = len(dataSkeleton)

    def XYclean(x, y):
        xy = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1)), axis=1)

        # make PCA object
        pca = PCA(2)
        # fit on data
        pca.fit(xy)

        # transform into pca space
        xypca = pca.transform(xy)
        newx = xypca[:, 0]
        newy = xypca[:, 1]

        # sort
        indexSort = np.argsort(x)
        newx = newx[indexSort]
        newy = newy[indexSort]

        # add some more points (optional)
        f = interpolate.interp1d(newx, newy, kind='linear')
        newX = np.linspace(np.min(newx), np.max(newx), 100)
        newY = f(newX)

        # smooth with a filter (optional)
        window = 43
        newY = signal.savgol_filter(newY, window, 2)

        # return back to old coordinates
        xyclean = pca.inverse_transform(np.concatenate((newX.reshape(-1, 1), newY.reshape(-1, 1)), axis=1))
        xc = xyclean[:, 0]
        yc = xyclean[:, 1]

        return xc, yc

    def getEqDistPoints(x, y, N):
        # Linear length on the line
        distance = np.cumsum(np.sqrt(np.ediff1d(x, to_begin=0) ** 2 + np.ediff1d(y, to_begin=0) ** 2))
        distance = distance / distance[-1]
        fx, fy = interp1d(distance, x), interp1d(distance, y)

        alpha = np.linspace(0, 1, N)
        px, py = fx(alpha), fy(alpha)
        return px, py

    def pointFilter(data):
        xs, ys = data[0], data[1]
        xp, yp = XYclean(xs, ys)
        return xp, yp

    def getContour(frame, data):
        x = [data[frame][0][i][0] for i in range(len(data[frame][0]))]
        y = [data[frame][0][i][1] for i in range(len(data[frame][0]))]
        return x, y

    def getPoints(frame, data, N, fixed_end):
        # Filter - This tried to order the points too.
        x, y = pointFilter(dataSkeleton[frame])

        # If the order is reversed -> reverse it
        if pow(x[0] - fixed_end[0], 2) + pow(y[0] - fixed_end[1], 2) > pow(x[-1] - fixed_end[0], 2) + pow(
                y[-1] - fixed_end[1], 2):
            x = x[::-1]
            y = y[::-1]

        # Get N equidistanced points
        px, py = getEqDistPoints(x, y, N)
        return px, py

    def init():
        outline.set_data([], [])
        trackpoints.set_data([], [])
        fixed_end.set_data([], [])
        free_end.set_data([], [])
        frame_text.set_text('')
        return outline, trackpoints, fixed_end, free_end, frame_text
    #
    ## animation function of frame list
    #
    tx = []
    ty = []
    #
    def animate(k):
        try:
            x, y = getContour(k, dataContour)
            x_, y_ = x-g[0], y-g[1]
            # It is necessary to give an aprox position of the fixed end in order to reorder it correctly
            px, py = getPoints(k, dataSkeleton, Npoints, (g[0], g[1]))
            px_, py_ = px-g[0], py-g[1]

            tx.append(px_), ty.append(py_)

            fix_x = px_[0]
            fix_y = py_[0]
            np.delete(px_, 0)
            np.delete(py_, 0)

            free_x = px_[-1]
            free_y = py_[-1]
            np.delete(px_, -1)
            np.delete(py_, -1)

            outline.set_data(x_, y_)
            trackpoints.set_data(px_, py_)
            fixed_end.set_data(fix_x, fix_y)
            free_end.set_data(free_x, free_y)

            fixed_end.set_markersize(5)
            free_end.set_markersize(5)
            trackpoints.set_markersize(5)
            frame_text.set_text('Frame = %.1f' % k)
            return outline, trackpoints, fixed_end, free_end, frame_text
        except Exception:
            pass

    print('creating tracked gif ...')

    fig = plt.figure()
    ax = plt.axes(xlim=(-lmax-50, lmax+50), ylim=(-lmax-50, lmax+50))
    ax.grid()

    outline, = ax.plot([], [], 'b')
    trackpoints, = ax.plot([], [], 'ro')
    fixed_end, = ax.plot([], [], 'ko')
    free_end, = ax.plot([], [], 'go')
    #
    frame_text = ax.text(0.02, 0.92, '', transform=ax.transAxes)
    # #
    anim = FuncAnimation(fig, animate, frames=nFrames, init_func=init, interval=1, blit=False)
    anim.save(vidpath + color + type + data + video + '_tracked.gif', dpi=150, writer=PillowWriter(fps=25))

    print('saving data ...')

    # Calculate the x- and y-distances of each tracked point from the pivot point
    rx = [(x - x[0]) for k, x in enumerate(tx)]
    ry = [(y - y[0]) for k, y in enumerate(ty)]

    # Create a pandas DataFrame to store the x and y values for each node
    df_xy = pd.DataFrame({'Node{}x'.format(i): rx[i] for i in range(len(rx))})
    df_xy = df_xy.assign(**{'Node{}y'.format(i): ry[i] for i in range(len(ry))})
    df_xy.index = np.arange(1, len(df_xy) + 1)

    theta = np.degrees(np.arctan2(ry, rx)) % 360  # angle of tracked pt. with respect to pivot pt.
    df_theta = pd.DataFrame(theta)
    df_theta = pd.DataFrame(theta, columns=['Node{}'.format(cols) for cols in df_theta.columns])
    df_theta.index = np.arange(1, len(df_theta) + 1)

    df_xy.to_csv(vidpath + color + type + data + video + '_output_xy.csv', index_label='Frame_No')
    df_theta.to_csv(vidpath + color + type + data + video + '_output_theta.csv', index_label='Frame_No')

    print('done!')


if __name__ == '__main__':

    vidpath = 'Z:/Atanu/exp_2021_fluid_ants/soft_rods'
    color = '/white_rod/'  # 'white'
    type = 'white_5cm_hinged'
    data =  '/gif/'
    video = 'S5120007'

    dataContour = pd.read_pickle(vidpath + color + type + data + video + '_contourDict.pkl')

    pivot = []

    for k in dataContour.keys():
        value = dataContour[k]
        valx, valy = map(list, zip(*value[0]))
        pivotk = np.vstack((valx[0], valy[0]))
        pivot.append(pivotk)

    g = [np.median([x[0] for x in pivot]), np.median([y[1] for y in pivot])]

    dataSkeleton = pd.read_pickle(vidpath + color + type + data + video + '_skeletonDict.pkl')

    tail = []

    for k in dataSkeleton.keys():
        value = dataSkeleton[k]
        tailk = [value[0][-1], value[1][-1]]
        tail.append(tailk)

    l = [np.sqrt((t[0]-g[0])**2 + (t[1]-g[1])**2) for t in tail]
    lmax = int(np.max(l))
    Npoints = 6  # Number of points to track (including both ends)

    main(dataContour, dataSkeleton, g, lmax, Npoints)
