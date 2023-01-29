import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Solver:
    def __init__(self, constraint):
        self.constraint = constraint

    def solve(self, q, q0, qDot, dNp0, params):
        '''
        This function solves the differential equation
            :param q: position vector
            :param q0: initial position vector
            :param qDot: velocity vector
            :param dNp0: initial state of node Np+ - Np-
            :param params: a dictionary containing all the parameters required for the simulation
            :return: position and velocity of each node at every time step
        '''

        forces = Forces(q, f0=params['f0'], Ng=params['Ng'], nestDir=params['nestDir'], nv=params['nv'],
                        bNode=params['boundary_node'], dt=params['dt'], kc=params['kc'],
                        Np=params['Np'], F_ind=params['F_ind'], EA=params['EA'], EI=params['EI'], refLen=params['dL'])

        Fg = forces.getFg()  # force due to informed ants
        Fs = forces.getFs()  # force due to stretching
        Fb = forces.getFb()  # force due to bending

        F = qDot

        Fp, dNp = forces.getFp(F, dNp0)  # force due to pulling

        F = (Fp - (Fs + Fg + Fb)) * params['gamma']  # force due to damping

        qDotNew = F

        qNew = q0 + F * params['dt']  # position of each node at every time step

        self.constraint.pinEnd(qNew, qDotNew)  # pinning the end nodes
        # self.constraint.clampEnd(qNew, qDotNew)   # clamping the end nodes

        return qNew, dNp

    def plot(self, x, currTime, size):
        '''
        This function plots the rod simultaneously while solving the differential equations
            :param x: instantaneous position of all nodes
            :param currTime: current time
            :param RodLength: length of rod
            :return: instantaneous node positions of all the nodes on the rod
        '''
        x1 = x[0:len(x):2]  # x[0:::2]
        x2 = x[1:len(x):2]  # x[1:::2]

        plt.plot(x1, x2, 'ko-')
        plt.xlabel(r'$x~[cm]$')
        plt.ylabel(r'$y~[cm]$')
        plt.xlim([-(size + 1), (size + 1)])
        plt.ylim([-(size + 1), (size + 1)])
        plt.title(f'$t = {currTime:.2f}$')
        plt.tight_layout()


class Simulation:
    def __init__(self, totalTime, nv):
        '''
            Initialize the simulation with total simulation time,
            number of node vertices and initialize variables for simulation step size,
            step intervals when the plots are updated, number of simulation steps,
            empty arrays to store position and velocity of each node as a function of time, new position, and nodal forces
        '''
        self.totalTime = totalTime  # total simulation time
        self.nv = nv  # number of node vertices
        self.dt = None  # simulation step size
        self.plotStep = None  # step intervals when the plots are updated
        self.Nsteps = None  # number of simulation steps
        self.all_q = None  # empty array to store position of each node as a function of time
        self.all_qDot = None  # empty array to store velocity of each node as a function of time
        self.qNew = None  # new position
        self.dNp = None  # nodal occupancy

    def setSimulation(self):
        '''
        This function sets the simulation
            :param totalTime: total simulation length
            :return: simulation step size, step intervals when the plots are updated, number of simulation steps
        '''
        # Set the simulation step size, step intervals when the plots are updated and number of simulation steps
        dt = 1e-6
        plotStep = 1e5
        Nsteps = round(self.totalTime / dt)
        return dt, plotStep, Nsteps

    def getDataArrays(self, Nsteps):
        '''
        Obtain empty arrays to populate position and velocity of each node as a function of time
            :param nv: number of node vertices
            :param Nsteps: number of simulation steps
            :return: two empty arrays of size (Nsteps, 2 * nv)
        '''
        # Create empty arrays for position and velocity of each node as a function of time
        all_q, all_qDot = np.zeros((Nsteps, 2 * self.nv)), np.zeros((Nsteps, 2 * self.nv))

        return all_q, all_qDot

    def createDataFrame(self, x, dt):
        '''
            This function creates a dataframe of the rod's position at every time step
        '''
        df = pd.DataFrame(x)
        columns = []
        for i in range(self.nv):
            columns.append("node{}x".format(i))
            columns.append("node{}y".format(i))

        df.columns = columns
        df.insert(0, 'time', (np.arange(len(df)) * dt), True)

        return df


class Node:
    def __init__(self, x, y=0, dNp=0):
        self.x = x  # x coordinate of the node
        self.y = y  # y coordinate of the node
        self.dNp = dNp  # Initial axial force at the node


class Ring:
    def __init__(self, nv, w, t, r, Y, dNp=0):
        self.nv = nv  # Number of nodes in the ring
        self.w = w  # Width of the ring
        self.t = t  # Thickness of the ring
        self.r = r  # Radius of the ring
        self.Y = Y  # Young's modulus of the ring
        self.EA = Y * w * t  # Axial stiffness of the ring
        self.EI = Y * (w * t ** 3) / 12  # Flexural stiffness of the ring
        self.dNp = dNp  # Initial (Np+ - Np-) at each node
        angles = np.linspace(0, 2 * np.pi, nv, endpoint=False)
        self.nodes = [Node(r * np.cos(angle), r * np.sin(angle), dNp=self.dNp) for angle in angles]
        self.ne = nv  # Number of elements in the ring
        self.dL = np.zeros((self.ne, 1))  # Array to store element lengths

    def createRing(self):
        self.nodes.append(Node(self.nodes[0].x, self.nodes[0].y, dNp=self.dNp))

        for k in range(self.ne):
            dl = np.array([self.nodes[k + 1].x - self.nodes[k].x,
                           self.nodes[k + 1].y - self.nodes[k].y])  # Calculate the length of each element
            self.dL[k] = np.linalg.norm(dl)  # Store the length of the element in the dL array
        return self.EA, self.EI, self.nodes, self.ne, self.dL

    def getStateVectors(self):
        '''
            This function obtains the initial state vectors at each node
        '''
        q0 = np.zeros((2 * self.nv, 1))  # state vector with x and y coordinates
        dNp0 = np.zeros((self.nv, 1))  # state vector for (Np+ - Np-)

        for k in range(self.nv):
            q0[2 * k] = self.nodes[k].x  # Store x coordinates of the nodes in the q0 array
            q0[2 * k + 1] = self.nodes[k].y  # Store y coordinates of the nodes in the q0 array
            dNp0[k] = self.nodes[k].dNp  # Store initial (Np+ - Np-) at each node in the dNp0 array
        return q0, dNp0


class Rod:
    '''
        This class creates the rod with the given parameters
    '''

    def __init__(self, nv, w, t, l, Y, dNp=0):
        self.nv = nv  # Number of nodes in the rod
        self.w = w  # Width of the rod
        self.t = t  # Thickness of the rod
        self.l = l  # Length of the rod
        self.Y = Y  # Young's modulus of the rod
        self.EA = Y * w * t  # Axial stiffness of the rod
        self.EI = Y * (w * t ** 3) / 12  # Flexural stiffness of the rod
        self.dNp = dNp  # Initial (Np+ - Np-) at each node

        self.nodes = [Node(x, dNp=self.dNp) for x in np.linspace(0, l,
                                                                 nv)]  # Create an array of nodes with given x coordinates and initial axial force
        self.ne = nv - 1  # Number of elements in the rod
        self.dL = np.zeros((self.ne, 1))  # Array to store element lengths

    def createRod(self):
        for k in range(self.ne):
            dl = np.array([self.nodes[k + 1].x - self.nodes[k].x])  # Calculate the length of each element
            self.dL[k] = np.linalg.norm(dl)  # Store the length of the element in the dL array
        return self.EA, self.EI, self.nodes, self.ne, self.dL

    def getStateVectors(self):
        '''
            This function obtains the initial state vectors at each node
        '''
        q0 = np.zeros((2 * self.nv, 1))  # state vector with x and y coordinates
        dNp0 = np.zeros((self.nv, 1))  # state vector for (Np+ - Np-)

        for k in range(self.nv):
            q0[2 * k] = self.nodes[k].x  # Store x coordinates of the nodes in the q0 array
            q0[2 * k + 1] = self.nodes[k].y  # Store y coordinates of the nodes in the q0 array
            dNp0[k] = self.nodes[k].dNp  # Store initial (Np+ - Np-) at each node in the dNp0 array
        return q0, dNp0


class Constraints:
    '''
        The class Constraints has two methods: pinEnd and clampEnd.
        pinEnd: takes two arguments q and qDot and sets the x,y coordinates of q to fixed x,y coordinates
        and the x,y velocity of qDot to 0.
        clampEnd: takes two arguments q and qDot and sets the x,y coordinates of q to fixed x,y coordinates
        and the angle of the endpoint represented by cosine and sine of angle to fixed angle and x,y, angular velocity of qDot to 0.
    '''

    def __init__(self, nv, fix_x, fix_y, theta_fixed):
        self.nv = nv  # number of variables
        self.fix_x = fix_x  # x-coordinate of fixed point
        self.fix_y = fix_y  # y-coordinate of fixed point
        self.theta_fixed = theta_fixed  # fixed angle

    def pinEnd(self, q, qDot):
        """
            Pin the endpoint to a fixed location
        """
        q[0] = self.fix_x  # set x-coordinate of endpoint to fixed x-coordinate
        q[1] = self.fix_y  # set y-coordinate of endpoint to fixed y-coordinate
        qDot[0] = 0  # set x-velocity of endpoint to 0
        qDot[1] = 0  # set y-velocity of endpoint to 0

    def clampEnd(self, q, qDot):
        """
            Clamp the endpoint to a fixed location and angle
        """
        q[0] = self.fix_x  # set x-coordinate of endpoint to fixed x-coordinate
        q[1] = self.fix_y  # set y-coordinate of endpoint to fixed y-coordinate
        q[2] = np.cos(self.theta_fixed)  # set cosine of angle of endpoint to fixed angle
        q[3] = np.sin(self.theta_fixed)  # set sine of angle of endpoint to fixed angle
        qDot[0] = 0  # set x-velocity of endpoint to 0
        qDot[1] = 0  # set y-velocity of endpoint to 0
        qDot[2] = 0  # set angular velocity of endpoint to 0
        qDot[3] = 0  # set angular velocity of endpoint to 0


class Forces:
    def __init__(self, q, f0, Ng, nestDir, nv, bNode, dt, kc, Np, F_ind, EA, EI, refLen):  # , ):
        self.q = q  # position vector
        self.f0 = f0  # single ant force
        self.Ng = Ng  # number of informed ants
        self.nestDir = nestDir  # direction of nest
        self.nv = nv  # number of node vertices
        self.bNode = bNode  # starting node or boundary node for force calculation
        self.dt = dt  # time step
        self.kc = kc  # spring constant
        self.Np = Np  # Number of time steps
        self.F_ind = F_ind  # force applied at a specific node
        self.EA = EA  # Axial stiffness of the rod
        self.EI = EI  # Flexural stiffness of the rod
        self.refLen = refLen  # reference length of the rod
        self.Fg = None  # force due to informed ants
        self.Fp = None  # force due to pulling
        self.Fs = None  # force due to stretching
        self.Fb = None  # force due to bending

    def getFg(self):
        '''
        Obtain force from informed ants
            :param q: array of position vector
            :param f0: single ant force
            :param Ng: number of informed ants
            :param nestDir: direction of nest
            :param nv: number of node vertices
            :param boundary_node: starting node or boundary node for force calculation
            :return: force from informed ants
        '''
        Fg = np.zeros((2 * self.nv, 1))  # create an array of zeros to store force from informed ants
        for k in range(self.bNode, self.nv):  # loop through the nodes from the starting node to the last node
            Fg[2 * k: 2 * k + 2, 0] = self.f0 * self.Ng * np.array([np.cos(self.nestDir), np.sin(self.nestDir)])
        self.Fg = -Fg  # store the calculated force in the Fg attribute of the class and make it negative
        return self.Fg

    def getFp(self, FOld, dNp0):
        '''
            Obtain force due to puller ants
            :param FOld: force from previous time step
            :param dNp0: initial (Np+ - Np-)
            :return: force due to pulling
        '''
        Fp = np.zeros((2 * self.nv, 1))  # create an array of zeros to store the force due to pulling
        dNp = dNp0  # initialize the array of initial state of Np+ - Np-

        for k in range(self.bNode, self.nv):
            Fk = FOld[2 * k: 2 * k + 2, :]  # get the force vector at kth node from the force vector array

            q_ = self.q[2 * k: 2 * k + 2, :]  # Get the position vector
            q_perpendicular = np.array(
                [-q_[1, 0], q_[0, 0]])  # get the normal vector with respect to the position vector
            q_perpendicular = q_perpendicular[:, np.newaxis]  # Making it a column vector

            if np.linalg.norm(q_perpendicular) == 0:
                chi = np.linalg.norm(q_perpendicular) + 1e34  # add a small value to avoid divide by zero error
            else:
                chi = np.linalg.norm(q_perpendicular)

            unit = q_perpendicular / chi  # normalize the normal vector

            Fk_ = np.dot(unit.T, Fk)  # dot product of the normal vector and the force vector

            # update the Np+ - Np- state at kth node
            dNp[k] = ((self.kc * self.Np * np.sinh(Fk_ / self.F_ind) - 2 * self.kc * dNp0[
                k] * np.cosh(Fk_ / self.F_ind)) * self.dt) + dNp0[k]

            Fp_ = self.f0 * dNp[k] * unit  # Getting the Fp for kth node in the tangential direction

            Fp[2 * k: 2 * k + 2] = Fp_  # store the force due to pulling at kth node

        return Fp, dNp

    def getFs(self):
        '''
        Obtain stretching force array
            :param q: position state vector
            :param nv: number of node vertices
            :param EA: stiffness
            :param refLen: un-stretched length
            :alpha: factor to scale spring force
            :return: stretching force array
        '''
        Fs = np.zeros((2 * self.nv, 1))  # initialize the stretching force array to store stretching force at each node
        alpha = 1  # factor to scale spring force
        for k in range(self.nv - 1):
            gradEnergy = self.gradEs(self.q[2 * k], self.q[2 * k + 1], self.q[2 * k + 2], self.q[2 * k + 3],
                                     self.refLen[k],
                                     self.EA)  # calculate the gradient of stretching energy for kth element
            Fs[2 * k: 2 * (k + 1) + 2] = Fs[2 * k: 2 * (
                    k + 1) + 2] - gradEnergy  # subtract gradient of stretching energy from the force array
            Fs[2 * k: 2 * (k + 1) + 2] = (
                    Fs[2 * k: 2 * (k + 1) + 2] * alpha)  # scale the force array with the alpha factor
        return Fs

    def gradEs(self, xk, yk, xkp1, ykp1, lk, EA):
        '''
        Gradient of stretching/spring energy
            :param xk: x position of node k
            :param yk: y position of node k
            :param xkp1: x+1 position of node k
            :param ykp1: y+1 position of node k
            :param lk: reference length or un-stretched edge length
            :param EA: edge stiffness
            :return: spring force
        '''
        F = np.zeros((4, 1))

        F[0] = (1 - np.sqrt((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / lk) / (
                np.sqrt((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) * lk) * 2 * (- xkp1 + xk)  # x component of force
        F[1] = (1 - np.sqrt((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / lk) / (
                np.sqrt((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) * lk) * 2 * (- ykp1 + yk)  # y component of force
        F[2] = (1 - np.sqrt((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / lk) / (
                np.sqrt((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) * lk) * 2 * (xkp1 - xk)  # x+1 component of force
        F[3] = (1 - np.sqrt((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / lk) / (
                np.sqrt((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) * lk) * 2 * (ykp1 - yk)  # y+1 component of force

        F = 0.5 * EA * lk * F  # multiply the force with the stiffness and reference length

        return F

    def getFb(self):
        '''
        Obtain bending force array
            :param q: array of position vector
            :param boundary_node: starting node or boundary node for force calculation
            :param nv: number of node vertices
            :param EI: flexural rigidity
            :param refLen: un-stretched length
            :return: bending force array
        '''
        Fb = np.zeros((2 * self.nv, 1))  # initialize the bending force array to store bending force at each node

        for k in range(self.bNode, self.nv - 2):
            gradEnergy = self.gradEb(self.q[2 * k, 0], self.q[2 * k + 1, 0], self.q[2 * k + 2, 0], self.q[2 * k + 3, 0],
                                     self.q[2 * k + 4, 0],
                                     self.q[2 * k + 5, 0], 0, self.refLen[k],
                                     self.EI)  # calculate the gradient of bending energy for kth element

            Fb[2 * k: 2 * (k + 1) + 4] = Fb[2 * k: 2 * (
                    k + 1) + 4] - gradEnergy  # subtract gradient of bending energy from the force array

            Fb[2 * k: 2 * (k + 1) + 4] = Fb[2 * k: 2 * (k + 1) + 4]  # scale the force array with the alpha factor

        return Fb

    def gradEb(self, xkm1, ykm1, xk, yk, xkp1, ykp1, curvature0, lk, EI):
        '''
        Gradient of bending energy
            :param xkm1: x-1 position of node k
            :param ykm1: y-1 position of node k
            :param xk: x position of node k
            :param yk: y position of node k
            :param xkp1: x+1 position of node k
            :param ykp1: y+1 position of node k
            :param curvature0: initial curvature
            :param lk: un-stretched length
            :param EI: flexural rigidity
            :return: bending force
        '''
        node0 = np.array([xkm1, ykm1, 0])  # node 0
        node1 = np.array([xk, yk, 0])  # node 1
        node2 = np.array([xkp1, ykp1, 0])  # node 2

        m2e = np.array([0, 0, 1])  # material to element coordinate transformation matrix
        m2f = np.array([0, 0, 1])  # material to global coordinate transformation matrix
        kappaBar = curvature0  # initial curvature

        #   Computation of gradient of the two curvatures
        gradKappa = np.zeros((6, 1))  # initialize the gradient of curvature array

        ee = node1 - node0  # edge vector
        ef = node2 - node1  # edge vector

        norm_e = np.linalg.norm(ee)  # norm of edge vector
        norm_f = np.linalg.norm(ef)  # norm of edge vector

        te = ee / norm_e  # unit vector
        tf = ef / norm_f  # unit vector

        #   Curvature binormal  (kappaBar)

        if (1.0 + np.dot(te, tf)) == 0:  # check if the dot product is zero
            kb = 2.0 * np.cross(te, tf) / (1.0 + np.dot(te, tf) + 1e34)  # curvature binormal
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
        dkappa = kappa1 - kappaBar  # curvature difference
        F = gradKappa * EI * dkappa / lk  # bending force

        return F
