
import numpy as np
import matplotlib.pyplot as plt
from Vector import Vector


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

    def trace_optical_element(self, beam):
        if self.type == "Plane mirror":
            beam_out = self.trace_plane_mirror(beam)
        elif self.type == "Spherical mirror":
            beam_out = self.trace_spherical_mirror(beam)
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
        position.rotation(-(90-self.theta),"x")
        velocity.rotation(-(90-self.theta),"x")
        [beam.x,beam.y,beam.z] = [position.x,position.y,position.z]
        [beam.vx,beam.vy,beam.vz] = [velocity.x,velocity.y,velocity.z]



    def translation_to_the_screen(self,beam):
        beam.y=beam.y-self.q


    def intersection_with_plane_mirror(self,beam):
        t=-beam.z/beam.vz
        beam.x = beam.x+beam.vx*t
        beam.y = beam.y+beam.vy*t
        beam.z = beam.z+beam.vz*t


    def intersection_with_spherical_mirror(self,beam):
            a=beam.vx**2+beam.vy**2+beam.vz**2
            b=beam.x*beam.vx+beam.y*beam.vy+beam.z*beam.vz-beam.vz*self.R                             #This is not b but b/2
            c=beam.x**2+beam.y**2+beam.z**2-2*beam.z*self.R
            t=(-2*b+np.sqrt(4*b**2-4*a*c))/(2*a)
            if t[0]>=0:
                t=t
            else:
                t=(-b-np.sqrt(b**2-a*c))/a
            beam.x = beam.x+beam.vx*t
            beam.y = beam.y+beam.vy*t
            beam.z = beam.z+beam.vz*t

    def output_direction(self, beam):

        position=Vector(beam.x,beam.y,beam.z)
        if self.type == "Plane mirror":
            normal=position.spherical_normal(self.R)
        elif self.type == "Spherical mirror":
            normal=position.spherical_normal(self.R)
        normal.normalization()
        velocity=Vector(beam.vx,beam.vy,beam.vz)
        vperp=velocity.perpendicular_component(normal)
        v2=velocity.sum(vperp)
        v2=v2.sum(vperp)

        [beam.vx,beam.vy,beam.vz] = [ v2.x, v2.y, v2.z]




    def intersection_with_the_screen(self,beam):
        t=-beam.y/beam.vy
        beam.x = beam.x+beam.vx*t
        beam.y = beam.y+beam.vy*t
        beam.z = beam.z+beam.vz*t



    def trace_plane_mirror(self, beam):

        #
        # change beam to o.e. frame
        #
        self.rotation(beam)
        self.translation(beam)


        #
        # intersection and output direction
        #
        self.intersection_with_plane_mirror(beam)

        beam.plot_yx()
        plt.title("footprint")

        # effect of the plane mirror #############################

#        beam.vz=-beam.vz


        v=Vector(beam.vx,beam.vy,beam.vz)
        normal=v.plane_normal()
        normal.normalization()
        vperp=v.perpendicular_component(normal)
        v2=v.sum(vperp)
        v2=v2.sum(vperp)

        [beam.vx,beam.vy,beam.vz] = [ v2.x, v2.y, v2.z]
        ###########################################################

        #
        # from o.e. frame to image frame
        #

        self.rotation_to_the_screen(beam)
        self.translation_to_the_screen(beam)
        self.intersection_with_the_screen(beam)



        return beam

    def trace_spherical_mirror(self, beam):

        self.rotation(beam)
        self.translation(beam)
        self.intersection_with_spherical_mirror(beam)

        beam.plot_yx()
        plt.title("footprint")

        self.output_direction(beam)


#        # effect of the spherical mirror ##########################
#
#        position=Vector(beam.x,beam.y,beam.z)
#        normal=position.spherical_normal(self.R)
#        normal.normalization()
#        v=Vector(beam.vx,beam.vy,beam.vz)
#        vperp=v.perpendicular_component(normal)
#        v2=v.sum(vperp)
#        v2=v2.sum(vperp)
#
#        [beam.vx,beam.vy,beam.vz] = [ v2.x, v2.y, v2.z]
#        ###########################################################


        self.rotation_to_the_screen(beam)
        self.translation_to_the_screen(beam)
        self.intersection_with_the_screen(beam)

        return beam






