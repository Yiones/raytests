
import numpy as np
import matplotlib.pyplot as plt
from Vector import Vector
from numpy import pi

from Vector import normalization3d

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

    def reflection(self, beam):
        if self.type == "Plane mirror":
            beam_out = self.reflection_plane_mirror(beam)
        elif self.type == "Spherical mirror":
            beam_out = self.reflection_spherical_mirror(beam)
        else:
            raise NotImplemented("Surface not valid")

        return beam_out



    def rotation(self, beam):                               # rotation along x axis of an angle depending on theta, and along y with an angle tetha

        position = Vector(beam.x,beam.y,beam.z)
        velocity = Vector(beam.vx,beam.vy,beam.vz)
        position.rotation(self.alpha,"y")
        position.rotation(-(90-self.theta),"x")
        velocity.rotation(self.alpha,"y")
        velocity.rotation(-(90-self.theta),"x")
        [beam.x,beam.y,beam.z] = [position.x,position.y,position.z]
        [beam.vx,beam.vy,beam.vz] = [velocity.x,velocity.y,velocity.z]


    def rotation2(self,beam):
        position = Vector(beam.x, beam.y, beam.z)
        velocity = Vector(beam.vx, beam.vy, beam.vz)
        position.rotation_x2(-(90-self.theta))
        velocity.rotation_x2(-(90-self.theta))
        [beam.x,beam.y,beam.z] = [position.x,position.y,position.z]
        [beam.vx,beam.vy,beam.vz] = [velocity.x,velocity.y,velocity.z]



    def translation(self,beam):
        vector_point=Vector(0,self.p,0)
        vector_point.rotation(self.alpha,"y")
        vector_point.rotation(-(90-self.theta),"x")

        beam.x=beam.x-vector_point.x
        beam.y=beam.y-vector_point.y
        beam.z=beam.z-vector_point.z


    def rotation_to_the_screen(self,beam):

        position = Vector(beam.x,beam.y,beam.z)
        velocity = Vector(beam.vx,beam.vy,beam.vz)
        #position.rotation(self.alpha,"y")
        position.rotation(-(90-self.theta),"x")
        #velocity.rotation(self.alpha,"y")
        velocity.rotation(-(90-self.theta),"x")
        [beam.x,beam.y,beam.z] = [position.x,position.y,position.z]
        [beam.vx,beam.vy,beam.vz] = [velocity.x,velocity.y,velocity.z]





    def translation_to_the_screen(self,beam):
        beam.y=beam.y-self.q


    def intersection_with_plane_mirror(self,beam):
        for i in range(0,beam.N):
            t=-beam.z[i]/beam.vz[i]
            beam.x[i]=beam.x[i]+beam.vx[i]*t
            beam.y[i]=beam.y[i]+beam.vy[i]*t
            beam.z[i]=beam.z[i]+beam.vz[i]*t






    def intersection_with_spherical_mirror(self,beam):
        for i in range(0,beam.N):
            a=beam.vx[i]**2+beam.vy[i]**2+beam.vz[i]**2
            b=beam.x[i]*beam.vx[i]+beam.y[i]*beam.vy[i]+beam.z[i]*beam.vz[i]-beam.vz[i]*self.R                             #This is not b but b/2
            c=beam.x[i]**2+beam.y[i]**2+beam.z[i]**2-2*beam.z[i]*self.R
            t=(-2*b+np.sqrt(4*b**2-4*a*c))/(2*a)
            if t>0:
                t=t
            else:
                t=(-b-np.sqrt(b**2-a*c))/a
            beam.x[i]=beam.x[i]+beam.vx[i]*t
            beam.y[i]=beam.y[i]+beam.vy[i]*t
            beam.z[i]=beam.z[i]+beam.vz[i]*t


    def intersection_with_the_screen(self,beam):
        for i in range(0,beam.N):
            t=-beam.y[i]/beam.vy[i]
            beam.x[i]=beam.x[i]+beam.vx[i]*t
            beam.y[i]=beam.y[i]+beam.vy[i]*t
            beam.z[i]=beam.z[i]+beam.vz[i]*t







    def reflection_plane_mirror(self, beam):


        self.rotation(beam)
        self.translation(beam)
        self.intersection_with_plane_mirror(beam)


        normal=np.array([0,0,1])

        for i in range(0,beam.N):
            vector=[beam.vx[i],beam.vy[i],beam.vz[i]]
            vperp  = -np.dot(vector, normal)*normal
            vparall= vector+vperp
            [beam.vx[i], beam.vy[i], beam.vz[i]] = vperp+vparall


        beam.plot_yx()


        print (np.mean(beam.vx),np.mean(beam.vy),np.mean(beam.vz))
        self.rotation_to_the_screen(beam)
        print (np.mean(beam.vx),np.mean(beam.vy),np.mean(beam.vz))


        self.translation_to_the_screen(beam)

        self.intersection_with_the_screen(beam)



        return beam

    def reflection_spherical_mirror(self, beam):

        self.rotation(beam)
        self.translation(beam)

        self.intersection_with_spherical_mirror(beam)


        beam.plot_yx()




        for i in range(0,beam.N):
            normal  = np.array([2*beam.x[i], 2*beam.y[i], 2*beam.z[i]-2*self.R])
            normal  = normalization3d(normal)
            vector  = [beam.vx[i],beam.vy[i],beam.vz[i]]
            vperp   = -np.dot(vector, normal)*normal
            vparall = vector+vperp
            [beam.vx[i], beam.vy[i], beam.vz[i]] = vperp+vparall

        self.rotation_to_the_screen(beam)

        self.translation_to_the_screen(beam)
        self.intersection_with_the_screen(beam)



        return beam




