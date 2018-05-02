import numpy as np
import matplotlib.pyplot as plt
from numpy import pi

from OpticalElement import Optical_element


class Beam(object):

    def __init__(self,N=10000):

        self.x = np.zeros(N)
        self.y = np.zeros(N)
        self.z = np.zeros(N)


        self.vx = np.zeros(N)
        self.vz = np.zeros(N)
        self.vy = np.zeros(N)

        self.flag = np.zeros(N)

        self.N = N
        self.counter=0

    def set_point(self,x,y,z):

        self.x = x + self.x * 0.0
        self.y = y + self.y * 0.0
        self.z = z + self.z * 0.0


    def initialize_from_arrays(self, x,y,z,vx,vy,vz,flag,counter):

        self.x = x
        self.y = y
        self.z = z

        self.vx = vx
        self.vy = vy
        self.vz = vz

        self.flag = flag

        self.N = x.size
        self.counter=counter


    def duplicate(self):
        b = Beam(N=self.N)
        b.initialize_from_arrays(self.x.copy(),
                                 self.y.copy(),
                                 self.z.copy(),
                                 self.vx.copy(),
                                 self.vy.copy(),
                                 self.vz.copy(),
                                 self.flag.copy(),
                                 self.counter,
                                 )
        return b



    def set_gaussian_divergence(self,dx,dz):                                                                      # gaussian velocity distribution
        N=self.N
        self.vx = dx * (np.random.randn(N))
        self.vz = dz * (np.random.randn(N))
        self.vy = np.random.random(N)
        self.vy = np.sqrt(1 - self.vx**2 - self.vz**2)


    def set_flat_divergence(self,dx,dz):                                                                         # uniform velocity distribution
            N=self.N
            self.vx = dx * (np.random.random(N) - 0.5)*2
            self.vz = dz * (np.random.random(N) - 0.5)*2
            self.vy = np.random.random(N)
            self.vy = np.sqrt(1 - self.vx**2 - self.vz**2)

    def set_divergences_collimated(self):
        self.vx = self.x * 0.0 + 0.0
        self.vy = self.y * 0.0 + 1.0
        self.vz = self.z * 0.0 + 0.0

    #
    #
    #  graphics
    #
    def plot_xz(self):
        plt.figure()
        plt.plot(self.x, self.z, 'ro')
        plt.xlabel('x axis')
        plt.ylabel('z axis')

    def plot_xy(self):
        plt.figure()
        plt.plot(self.x, self.y, 'ro')
        plt.xlabel('x axis')
        plt.ylabel('y axis')

    def plot_zy(self):
        plt.figure()
        plt.plot(self.z, self.y, 'ro')
        plt.xlabel('z axis')
        plt.ylabel('y axis')

    def plot_yx(self):
        plt.figure()
        plt.plot(self.y, self.x, 'ro')
        plt.xlabel('y axis')
        plt.ylabel('x axis')

    def plot_xpzp(self):
        plt.figure()
        plt.plot(1e6*self.vx, 1e6*self.vz, 'ro')
        plt.xlabel('xp axis [urad]')
        plt.ylabel('zp axis [urad]')


    def plot_good_xz(self):
        plt.figure()
        indices = np.where(self.flag > 0)
        plt.plot(self.x[indices], self.z[indices], 'ro')
        plt.xlabel('x axis')
        plt.ylabel('z axis')

    def plot_good_xy(self):
        indices = np.where(self.flag > 0)
        plt.figure()
        plt.plot(self.x[indices], self.y[indices], 'ro')
        plt.xlabel('x axis')
        plt.ylabel('y axis')

    def plot_good_zy(self):
        indices = np.where(self.flag > 0)
        plt.figure()
        plt.plot(self.z[indices], self.y[indices], 'ro')
        plt.xlabel('z axis')
        plt.ylabel('y axis')

    def plot_good_yx(self):
        indices = np.where(self.flag > 0)
        plt.figure()
        plt.plot(self.y[indices], self.x[indices], 'ro')
        plt.xlabel('y axis')
        plt.ylabel('x axis')

    def plot_good_xpzp(self):
        indices = np.where(self.flag > 0)
        plt.figure()
        plt.plot(1e6*self.vx[indices], 1e6*self.vz[indices], 'ro')
        plt.xlabel('xp axis [urad]')
        plt.ylabel('zp axis [urad]')



    def histogram(self):
        plt.figure()
        plt.hist(self.x)
        plt.title('x position for gaussian distribution')

        plt.figure()
        plt.hist(self.z,1000)
        plt.title('z position for uniform distribution')



if __name__ == "__main__":

    #### Generation of the Beam

    beam1=Beam(2)
    beam1.set_point(0,0,0)
    #beam1.set_gaussian_divergence(0.001,0.0001)
    beam1.set_flat_divergence(5e-3,5e-2)

    #### Data of the plane mirron

    p=1.
    q=0.
    theta=0
    alpha=0
    R=2*p*q/(p+q)/np.cos(pi/180.0*theta)
    spherical_mirror=Optical_element.initialize_as_plane_mirror(p,q,theta,alpha)

    beam1=spherical_mirror.trace_optical_element(beam1)

    #beam1.plot_xz()


    plt.show()

