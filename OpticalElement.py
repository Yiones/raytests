import numpy as np
import matplotlib.pyplot as plt
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














    def reflection_plane_mirror(self, beam):

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

        beam.plot_xz()
        plt.title("Position of the rays at the screen")

        return beam

    def reflection_spherical_mirror(self, beam):

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

        return beam




