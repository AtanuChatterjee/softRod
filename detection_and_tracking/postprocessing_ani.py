import numpy as np
import pandas as pd
from scipy import interpolate
import scipy.signal as signal
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def slope(a, h):
    vel = np.zeros(a.size)
    vel[1:-1] = (a[2:] - a[:-2]) / (2 * h)
    vel[0] = (a[1] - a[0]) / h
    vel[-1] = (a[-1] - a[-2]) / h
    vel = signal.savgol_filter(vel, 43, 2)
    return vel

def filter_outlier(a):
    a = a[(-5<=a) & (a<=5)]
    a = np.array(a)
    return a

def main(thetadata, rdata, j):

    time = thetadata.Frame_No / 25

    theta = thetadata.j
    r = rdata.j
    # angvel_midpt = slope(np.array(angpos_midpt),25)
    #
    # vel_midpt = angvel_midpt*r_midpt
    # vel_midpt = filter_outlier(vel_midpt)
    plt.plot(time, theta)

    # def init():
    #     theta.set_data([], [])
    #     velocity.set_data([], [])
    #     frame_text.set_text('')
    #     return theta, velocity, frame_text
    #
    # def animate(k):
    #     data = posdata.iloc[:int(k + 1)]
    #     time = data.Frame_No / 25
    #     pos_midpt = data.Node4
    #     vel_midpt = slope(pos_midpt,25)
    #     theta.set_data(time, pos_midpt)
    #     velocity.set_data(time, vel_midpt)
    #
    #     frame_text.set_text('Frame = %.1f' % k)
    #     return theta, velocity, frame_text
    #
    # print('creating gif ...')
    #
    # fig, ax1 = plt.subplots()
    # ax1 = plt.axes(xlim=(0,max(time)), ylim=(-360,360))
    # ax1.grid()
    #
    # ax2 = ax1.twinx()
    #
    # theta, = ax1.plot([], [], 'g')
    # velocity, = ax2.plot([], [], 'r')
    # #
    # frame_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes)
    # #
    # anim = FuncAnimation(fig, animate, frames=len(posdata), init_func=init, interval=1, blit=True)
    # # anim.save(path + folder + '/' + video + '_tracked.gif', dpi=300, writer=PillowWriter(fps=25))
    plt.show()

if __name__ == '__main__':

    path = 'C:/Users/atanu/Desktop/soft_pendulum/final/'
    # path = '/Users/atanu/Desktop/soft_pendulum/videos/'
    video = 'S4800001'
    folder = 'S4800001'

    thetadata = pd.read_csv(path + folder + '/' + video + '_output_theta.csv')
    thetadata = pd.DataFrame(thetadata)

    rdata = pd.read_csv(path + folder + '/' + video + '_output_r.csv')
    rdata = pd.DataFrame(rdata)

    j = Node4

    main(thetadata, rdata, j)