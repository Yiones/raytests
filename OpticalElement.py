
import numpy as np
import matplotlib.pyplot as plt
from Vector import Vector
from SurfaceConic import SurfaceConic


class Optical_element(object):

    def __init__(self,ccc=SurfaceConic()):
        self.p=0
        self.q=0
        self.theta=0
        self.alpha=0
        self.R=0
        self.type=""

        self.ccc_object = ccc

    def set_parameters(self,p,q,theta,alpha,R=0):
            self.p = p
            self.q = q
            self.theta = theta
            self.alpha = alpha
            self.R=R


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


    @classmethod
    def initialize_from_coefficients(cls):
        if np.array(cls.ccc_object).size != 10:
            raise Exception("Invalid coefficients (dimension must be 10)")
        # return Optical_element(ccc=ccc)
        else:
            cls.ccc_object = SurfaceConic.initialize_from_coefficients(cls)


    @classmethod
    def initialize_as_plane(cls):
        return SurfaceConic(np.array([0,0,0,0,0,0,0,0,-1.,0]))

    #
    # initializers from focal distances
    #

    @classmethod
    def initialize_as_sphere_from_focal_distances(cls, p, q, theta, cylindrical=0, cylangle=0.0,
                                                  switch_convexity=0):
        oe=Optical_element()
        oe.p=p
        oe.q=q
        oe.theta=theta
        oe.type="Surface conical mirror"
        oe.ccc_object = SurfaceConic()
        oe.ccc_object.set_sphere_from_focal_distances(p, q, np.pi/2-theta)
        if cylindrical:
            oe.ccc_object.set_cylindrical(cylangle)
        if switch_convexity:
            oe.ccc_object.switch_convexity()
        return oe


    @classmethod
    def initialize_as_ellipsoid_from_focal_distances(cls, p, q, theta, cylindrical=0, cylangle=0.0,
                                                     switch_convexity=0):
        oe=Optical_element()
        oe.p=p
        oe.q=q
        oe.theta=theta
        oe.type="Surface conical mirror"
        oe.ccc_object = SurfaceConic()
        oe.ccc_object.set_ellipsoid_from_focal_distances(p, q, np.pi/2-theta)
        if cylindrical:
            oe.ccc_object.set_cylindrical(cylangle)
        if switch_convexity:
            oe.ccc_object.switch_convexity()
        return oe


    @classmethod
    def initialize_as_paraboloid_from_focal_distances(cls, p, q, theta, cylindrical=0, cylangle=0.0,
                                                      switch_convexity=0):
        oe=Optical_element()
        oe.p=p
        oe.q=q
        oe.theta=theta
        oe.type="Surface conical mirror"
        oe.ccc_object = SurfaceConic()
        oe.ccc_object.set_paraboloid_from_focal_distances(p, q, np.pi/2-theta)
        if cylindrical:
            oe.ccc_object.set_cylindrical(cylangle)
        if switch_convexity:
            oe.ccc_object.switch_convexity()
        return oe


    @classmethod
    def initialize_as_hyperboloid_from_focal_distances(cls, p, q, theta, cylindrical=0, cylangle=0.0,
                                                       switch_convexity=0):
        oe=Optical_element()
        oe.p=p
        oe.q=q
        oe.theta=theta
        oe.type="Surface conical mirror"
        oe.ccc_object = SurfaceConic()
        oe.ccc_object.set_hyperboloid_from_focal_distances(p, q, np.pi/2-theta)
        if cylindrical:
            oe.ccc_object.set_cylindrical(cylangle)
        if switch_convexity:
            oe.ccc_object.switch_convexity()
        return oe
    #
    # initializars from surface parameters
    #



    def rotation_to_the_optical_element(self, beam):                               # rotation along x axis of an angle depending on theta, and along y with an angle tetha

        position = Vector(beam.x,beam.y,beam.z)
        velocity = Vector(beam.vx,beam.vy,beam.vz)
        position.rotation(self.alpha,"y")
        position.rotation(-(np.pi/2-self.theta),"x")
        velocity.rotation(self.alpha,"y")
        velocity.rotation(-(np.pi/2-self.theta),"x")
        [beam.x,beam.y,beam.z] = [position.x,position.y,position.z]
        [beam.vx,beam.vy,beam.vz] = [velocity.x,velocity.y,velocity.z]



    def translation_to_the_optical_element(self, beam):
        vector_point=Vector(0,self.p,0)
        vector_point.rotation(self.alpha,"y")
        vector_point.rotation(-(np.pi/2-self.theta),"x")

        beam.x=beam.x-vector_point.x
        beam.y=beam.y-vector_point.y
        beam.z=beam.z-vector_point.z


    def rotation_to_the_screen(self,beam):

        position = Vector(beam.x,beam.y,beam.z)
        velocity = Vector(beam.vx,beam.vy,beam.vz)
        position.rotation(-(np.pi/2-self.theta),"x")
        velocity.rotation(-(np.pi/2-self.theta),"x")
        [beam.x,beam.y,beam.z] = [position.x,position.y,position.z]
        [beam.vx,beam.vy,beam.vz] = [velocity.x,velocity.y,velocity.z]



    def translation_to_the_screen(self,beam):
        beam.y=beam.y-self.q


    def intersection_with_plane_mirror(self,beam,bound):
        t=-beam.z/beam.vz
        beam.x = beam.x+beam.vx*t
        beam.y = beam.y+beam.vy*t
        beam.z = beam.z+beam.vz*t
        beam.flag=beam.flag+(np.sign(beam.x-bound.xmin*np.ones(beam.N))-1)/2
        beam.flag=beam.flag+(np.sign(beam.y-bound.ymin*np.ones(beam.N))-1)/2
        beam.flag=beam.flag+(np.sign(bound.xmax*np.ones(beam.N)-beam.x)-1)/2
        beam.flag=beam.flag+(np.sign(bound.xmax*np.ones(beam.N)-beam.x)-1)/2
        beam.flag=np.sign(beam.flag)


    def intersection_with_spherical_mirror(self,beam,bound):
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

            beam.flag=beam.flag+(np.sign(bound.R**2*np.ones(beam.N)-beam.x**2-beam.y**2)-1)/2
            beam.flag=np.sign(beam.flag)


    def intersection_with_surface_conic(self,beam):

        [t, flag] = self.ccc_object.calculate_intercept(np.array([beam.x, beam.y, beam.z]),
                                                        np.array([beam.vx, beam.vy, beam.vz]))
        beam.x = beam.x + beam.vx * t
        beam.y = beam.y + beam.vy * t
        beam.z = beam.z + beam.vz * t


    def output_direction_from_optical_element(self, beam):

        position=Vector(beam.x,beam.y,beam.z)

        if self.type == "Plane mirror":
            normal=position.plane_normal()
        elif self.type == "Spherical mirror":
            normal=position.spherical_normal(self.R)
        elif self.type == "Surface conical mirror":
            normal = position.surface_conic_normal(self.ccc_object.get_coefficients())

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



    def intersection_with_optical_element(self, beam,bound):
        if self.type == "Plane mirror":
            self.intersection_with_plane_mirror(beam,bound)
        elif self.type == "Spherical mirror":
            self.intersection_with_spherical_mirror(beam,bound)
        elif self.type == "Surface conical mirror":
            self.intersection_with_surface_conic(beam)


    def trace_optical_element(self, beam1,bound=None):

        beam=beam1.duplicate()
        #
        # change beam to o.e. frame
        #
        self.rotation_to_the_optical_element(beam)
        self.translation_to_the_optical_element(beam)

        self.intersection_with_optical_element(beam,bound)

        beam.plot_yx()
        plt.title("footprint")

        self.output_direction_from_optical_element(beam)


        self.rotation_to_the_screen(beam)

        self.translation_to_the_screen(beam)
        self.intersection_with_the_screen(beam)



        return beam




#     def trace_surface_conic(self,beam):
# 
#         beam=beam.duplicate()
#         self.rotation_to_the_optical_element(beam)
#         self.translation_to_the_optical_element(beam)
# 
#         x2=np.array([[beam.x],[beam.y],[beam.z]])
#         v=self.ccc_object.get_normal(x2)
#         [t,flag]=self.ccc_object.calculate_intercept(np.array([beam.x,beam.y,beam.z]),np.array([beam.vx,beam.vy,beam.vz]))
# 
#         beam.x = beam.x+beam.vx*t
#         beam.y = beam.y+beam.vy*t
#         beam.z = beam.z+beam.vz*t
# 
# 
#         print(np.mean(beam.x))
#         print(np.mean(beam.y))
# 
#         beam.plot_yx()
#         plt.title("footprint")
# 
#         #####  Output direction ############################################################################################
#         position = Vector(beam.x, beam.y, beam.z)
#         normal = position.surface_conic_normal(self.ccc_object.get_coefficients())
#         normal.normalization()
#         velocity = Vector(beam.vx, beam.vy, beam.vz)
#         vperp = velocity.perpendicular_component(normal)
#         v2 = velocity.sum(vperp)
#         v2 = v2.sum(vperp)
#         [beam.vx, beam.vy, beam.vz] = [v2.x, v2.y, v2.z]
#         ####################################################################################################################
# 
#         print(self.ccc_object.get_coefficients())
#         self.rotation_to_the_screen(beam)
#         self.translation_to_the_screen(beam)
#         self.intersection_with_the_screen(beam)
# 
#         return beam