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

def moltiplication_beetween_matrix_and_array(a,b):
    (ra,ca)=a.shape
    c=np.zeros((ra))
    for i in range(0,ra):
        c[i]=np.dot(a[i,:],b)
    return c




class Vector(object):

    def __init__(self,x,y,z):

        self.x = x
        self.y = y
        self.z = z

    def rotation_x(self,alpha):
        rotational_matrix = np.array([[np.cos(alpha),0,np.sin(alpha)],[0,1,0],[-np.sin(alpha),0,np.cos(alpha)]])
        [self.x,self.y,self.z]=moltiplication_beetween_matrix_and_array(rotational_matrix,[self.x,self.y,self.z])


    def rotation_y(self,beta):
        rotational_matrix = np.array([[1, 0, 0], [0, np.cos(beta), -np.sin(beta)], [0, np.sin(beta), np.cos(beta)]])
        [self.x,self.y,self.z]=moltiplication_beetween_matrix_and_array(rotational_matrix,[self.x,self.y,self.z])


    def rotation_z(self,gamma):
        rotational_matrix = np.array([[np.cos(gamma), -np.sin(gamma), 0], [np.sin(gamma), np.cos(gamma), 0], [0, 0, 1]])
        [self.x,self.y,self.z]=moltiplication_beetween_matrix_and_array(rotational_matrix,[self.x,self.y,self.z])


    def rotation(self,alpha,axis="x"):

        angle=pi/180.0*alpha
        if axis == "x":
            self.rotation_x(angle)
        elif axis=="y":
            self.rotation_y(angle)
        elif axis=="z":
            self.rotation_z(angle)






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

    def rotation(self, theta, alpha):                         # rotation along x axis of an angle depending on theta, and along y with an angle tetha
        self.alpha=alpha
        self.theta=theta

        for i in range(0,self.N):
            position = Vector(self.x[i],self.y[i],self.z[i])
            velocity = Vector(self.vx[i],self.vy[i],self.vz[i])
            position.rotation(alpha)
            position.rotation(90-theta,"y")
            velocity.rotation(alpha)
            velocity.rotation(90-theta,"y")
            [self.x[i],self.y[i],self.z[i]] = [position.x,position.y,position.z]
            [self.vx[i],self.vy[i],self.vz[i]] = [velocity.x,velocity.y,velocity.z]


    def translation(self,point):
        vector_point=Vector(point[0],point[1],point[2])
        vector_point.rotation(self.alpha)
        vector_point.rotation(90-self.theta,"y")
        self.x=self.x-vector_point.x
        self.y=self.y-vector_point.y
        self.z=self.z-vector_point.z

    def translation_to_the_screen(self,point):
        self.x=self.x-point[0]
        self.y=self.y-point[1]
        self.z=self.z-point[2]


    def intersection_with_plane_mirror(self):
        for i in range(0,self.N):
            t=-self.z[i]/self.vz[i]
            self.x[i]=self.x[i]+self.vx[i]*t
            self.y[i]=self.y[i]+self.vy[i]*t
            self.z[i]=self.z[i]+self.vz[i]*t


    def intersection_with_spherical_mirror(self,R):
        self.R=R
        for i in range(0,self.N):
            a=self.vx[i]**2+self.vy[i]**2+self.vz[i]**2
            b=self.x[i]*self.vx[i]+self.y[i]*self.vy[i]+self.z[i]*self.vz[i]+self.vz[i]*R                             #This is not b but b/2
            c=self.x[i]**2+self.y[i]**2+self.z[i]**2+2*self.z[i]*R
            t=(-b+np.sqrt(b**2-a*c))/a
            if t>0:
                t=t
            else:
                t=(-b-np.sqrt(b**2-a*c))/a
            self.x[i]=self.x[i]+self.vx[i]*t
            self.y[i]=self.y[i]+self.vy[i]*t
            self.z[i]=self.z[i]+self.vz[i]*t


    def intersection_with_the_screen(self):
        for i in range(0,self.N):
            t=-self.y[i]/self.vy[i]
            self.x[i]=self.x[i]+self.vx[i]*t
            self.y[i]=self.y[i]+self.vy[i]*t
            self.z[i]=self.z[i]+self.vz[i]*t



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

    def histogram(self):
        plt.figure()
        plt.hist(self.x)
        plt.title('x position for gaussian distribution')

        plt.figure()
        plt.hist(self.z)
        plt.title('z position for uniform distribution')






class Optical_element(object):

    def __init__(self):
        self.p=0
        self.q=0
        self.theta=0
        self.alpha=0
        self.R=0
        self.type=""

    @classmethod
    def initialize_as_plane_mirror(cls, p, q,theta, alpha):

        plane_mirror=Optical_element()
        plane_mirror.p=p
        plane_mirror.q=q
        plane_mirror.theta=theta
        plane_mirror.alpha=alpha
        plane_mirror.type="Plane mirror"

        return plane_mirror


    @classmethod
    def initialize_as_spherical_mirror(cls, p, q,theta, alpha, R):

        spherical_mirror=Optical_element()
        spherical_mirror.p=p
        spherical_mirror.q=q
        spherical_mirror.theta=theta
        spherical_mirror.alpha=alpha
        spherical_mirror.R=R
        spherical_mirror.type="Spherical mirror"

        return spherical_mirror

    def set_parameters(self,p,q,theta,alpha,R=0):
            self.p = p
            self.q = q
            self.theta = theta
            self.alpha = alpha
            self.R=R


    def reflaction_plane_mirror(self,beam):

        point_before_the_mirror = np.array([[0],[self.p],[0]])
        point_after_the_mirror  = np.array([[0.],[self.q],[0.]])

        beam.rotation(self.theta,self.alpha)
        beam.translation(point_before_the_mirror)
        beam.intersection_with_plane_mirror()


        normal = np.array([0, 0, 1])
        for i in range(0,beam.N):
            vector=[beam.vx[i],beam.vy[i],beam.vz[i]]
            vperp  = -np.dot(vector, normal)*normal
            vparall= vector+vperp
            [beam.vx[i], beam.vy[i], beam.vz[i]] = vperp+vparall

        beam.plot_xy()
        plt.title("Position of the rays at the mirror")



        beam.rotation(self.theta,self.alpha)
        beam.translation_to_the_screen(point_after_the_mirror)
        beam.intersection_with_the_screen()

        beam1.plot_xz()
        plt.title("Position of the rays at the screen")

        return beam

    def reflaction_spherical_mirror(self,beam):

        point_before_the_mirror = np.array([[0],[self.p],[0]])
        point_after_the_mirror  = np.array([[0.],[self.q],[0.]])

        beam.rotation(self.theta,self.alpha)

        beam.translation(point_before_the_mirror)
        beam.intersection_with_spherical_mirror(self.R)

        beam.plot_xz()
        beam.plot_xy()

        #We have to write the effect of the sferical mirror


        for i in range(0,beam.N):
            normal  = np.array([2*beam.x[i], 2*beam.y[i], 2*beam.z[i]+2*self.R])
            normal  = normalization3d(normal)
            vector  = [beam.vx[i],beam.vy[i],beam.vz[i]]
            vperp   = -np.dot(vector, normal)*normal
            vparall = vector+vperp
            [beam.vx[i], beam.vy[i], beam.vz[i]] = vperp+vparall


        beam.rotation(self.theta,self.alpha)
        beam.translation_to_the_screen(point_after_the_mirror)
        beam.intersection_with_the_screen()


        beam.plot_xz()

        return beam1






if __name__ == "__main__":

    #### Generation of the Beam

    beam1=Beam()
    beam1.set_point(0,0,0)
    #beam1.set_gaussian_divergence(0.001,0.0001)
    beam1.set_flat_divergence(5e-7,5e-6)
    beam1.plot_xz()

    #### Data of the plane mirron

    p=1.
    q=1.
    theta=20
    alpha=0
    R=1

    spherical_mirror=Optical_element.initialize_as_spherical_mirror(p,q,theta,alpha,R)

    beam1=spherical_mirror.reflaction_spherical_mirror(beam1)

    print spherical_mirror.type



    plt.show()

