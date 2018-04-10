import numpy as np
import matplotlib.pyplot as plt
from numpy import pi

def normalization3d(v):
    mod=np.sqrt(v[0]**2+v[1]**2+v[2]**2)
    if mod==0:
        v[0]=0
        v[1]=0
        v[2]=0
    else:
        v=v/mod
    return v

class Beam(object):

    def __init__(self,N=1000):                    #p1: velocity distribution, p0: source geometry

        self.x = np.zeros(N)
        self.y = np.zeros(N)
        self.z = np.zeros(N)


        self.vx = np.zeros(N)
        self.vz = np.zeros(N)
        self.vy = np.zeros(N)

        self.t = np.zeros(N)

        self.N = N

    @classmethod
    def initialize_mysource(cls, dx,dz,p0,p1,N=1000,x=0,y=0,z=0):

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
            beam.vx = dx * (np.random.randn(N)) * 2
            beam.vz = dz * (np.random.randn(N)) * 2
            beam.vy = np.random.random(N)
            for i in range(0, N):
                beam.vy[i] = np.sqrt(1 - beam.vx[i] ** 2 - beam.vz[i] ** 2)

        beam.t=np.zeros(N)
        return beam

    def set_point_source(self,x0,y0,z0):
        self.x = self.x * 0.0 + x0
        self.y = self.y * 0.0 + y0
        self.z = self.z * 0.0 + z0

    def set_divergences_collimated(self):
        self.vx = self.x * 0.0 + 0.0
        self.vy = self.y * 0.0 + 1.0
        self.vz = self.z * 0.0 + 0.0

    def free_propagation(self,p):
        self.free_propagation_inclined(p,0.0,1.0,0.0)

    def free_propagation_inclined(self, p, v1, v2, v3):
        for i in range(0, self.N):
            t=p/(v1*self.vx[i]+v2*self.vy[i]+v3*self.vz[i])*v2
            self.x[i] = self.x[i]+self.vx[i]* t
            self.y[i] = self.y[i]+self.vy[i]* t
            self.z[i] = self.z[i]+self.vz[i]* t
            self.t[i] = self.t[i]+ t

    def rotation(self, n):
        self.y = self.y - self.vy * self.t

        y = np.array([0, 1, 0])
        if n[0] ** 2 + n[1] ** 2 == 0:
            fi = 0
            theta = np.array([pi / 2])
            w = [0, 1, 0]
        else:
            if n[0] != 0:
                wx = np.sqrt(1 / (1 + n[1] ** 2 / n[0] ** 2))
            else:
                wx = 0
            wy = np.sqrt(1 - wx ** 2)
            fi = np.arcsin((wx))
            w = [wx, wy, 0]

        k = np.cross(w, n)
        k1 = normalization3d(k)

        if k1[0] == 0 and k1[1] == 0 and k1[2] == 0:
            k[1] = 1
        else:
            k = k1

        if n[0] ** 2 + n[1] ** 2 != 0:
            mw = np.sqrt(w[0] ** 2 + w[1] ** 2)
            mn = np.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2)
            theta = np.array([np.arccos(mw / mn)])

        vrot = np.random.random(3)
        v = np.random.random(3)

        for i in range(0, self.N):
            v[0] = self.x[i]
            v[1] = 0
            v[2] = self.z[i]
            vrot = v * np.cos(theta[0]) + np.cross(k, v) * np.sin(theta[0]) + k * np.dot(k, v) * (1 - np.cos(theta[0]))
            self.x[i] = vrot[0]
            self.y[i] = vrot[1]
            self.z[i] = vrot[2]

    #
    #  graphics
    #
    def plot_xz(self):
        plt.figure()
        plt.plot(self.x, self.z, 'ro')
        plt.xlabel('x axis')
        plt.ylabel('z axis')

    def histogram(self):
        plt.figure()
        plt.hist(self.x)
        plt.title('x position for gaussian distribution')

        plt.figure()
        plt.hist(self.z)
        plt.title('z position for uniform distribution')




if __name__ == "__main__":

    # beam1=Beam(5000)
    # beam1.set_point(0,0,0)
    # beam1.set_gaussian_divergence(1.0,1.0)
    # beam1.set_flat_divergence(1.0,1.0)

    beam1=Beam.initialize_mysource(3e-3, 5e-6, 0, 0, N=5000,x=0,y=0,z=0)

    v=(0,1,1)


    beam1.free_propagation(2)

    beam1.free_propagation_inclined(10000,v[0],v[1],v[2])

    beam1.plot_xz()

    beam1.rotation(v)

    beam1.plot_xz()

    plt.show()

