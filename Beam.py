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

    def set_point(self,x,y,z):

        self.x = x + self.x
        self.y = y + self.y
        self.z = z + self.z


    def initialize_from_arrays(self, x,y,z,vx,vy,vz,flag):

        self.x = x
        self.y = y
        self.z = z

        self.vx = vx
        self.vy = vy
        self.vz = vz

        self.flag = flag

        self.N = x.size

    @classmethod
    def initialize_mysource(cls, dx,dz,p0,p1,N=10000,x=0,y=0,z=0):

        beam = Beam(N=N)
        if p0==0:                                                                       # point source
            beam.set_point_source(x,y,z)
        if p0==1:                                                                        #square spot with of 1 micrometer
            for i in range (0,N):
                beam.x=1*(np.random.random(N)-0.5)*1e-6
                beam.y=1*(np.random.random(N)-0.5)*1e-6
                beam.z=1*(np.random.random(N)-0.5)*1e-6

        if p1==0:                                                                         # uniform velocity distribution
            beam.vx = dx * (np.random.random(N) - 0.5)*2
            beam.vz = dz * (np.random.random(N) - 0.5)*2
            beam.vy = np.random.random(N)
            for i in range(0, N):
                beam.vy[i] = np.sqrt(1 - beam.vx[i] ** 2 - beam.vz[i] ** 2)
        if p1==1:                                                                        # gaussian velocity distribution
            beam.vx = dx * (np.random.randn(N))
            beam.vz = dz * (np.random.randn(N))
            beam.vy = np.random.random(N)
            for i in range(0, N):
                beam.vy[i] = np.sqrt(1 - beam.vx[i] ** 2 - beam.vz[i] ** 2)

        beam.t=np.zeros(N)
        return beam

    def duplicate(self):
        b = Beam(N=self.N)
        b.initialize_from_arrays(self.x.copy(),
                                 self.y.copy(),
                                 self.z.copy(),
                                 self.vx.copy(),
                                 self.vy.copy(),
                                 self.vz.copy(),
                                 self.flag.copy()  )
        return b


    def set_point_source(self,x0,y0,z0):
        self.x = self.x * 0.0 + x0
        self.y = self.y * 0.0 + y0
        self.z = self.z * 0.0 + z0


    def set_gaussian_divergence(self,dx,dz):                                                                      # gaussian velocity distribution
            N=self.N
            self.vx = dx * (np.random.randn(N))
            self.vz = dz * (np.random.randn(N))
            self.vy = np.random.random(N)
            for i in range(0, N):
                self.vy[i] = np.sqrt(1 - self.vx[i] ** 2 - self.vz[i] ** 2)


    def set_flat_divergence(self,dx,dz):                                                                         # uniform velocity distribution
            N=self.N
            self.vx = dx * (np.random.random(N) - 0.5)*2
            self.vz = dz * (np.random.random(N) - 0.5)*2
            self.vy = np.random.random(N)
            for i in range(0, N):
                self.vy[i] = np.sqrt(1 - self.vx[i] ** 2 - self.vz[i] ** 2)

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

    beam1=spherical_mirror.reflection(beam1)

    #beam1.plot_xz()


    plt.show()

