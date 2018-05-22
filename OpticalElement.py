import numpy as np
import matplotlib.pyplot as plt
from Vector import Vector
from SurfaceConic import SurfaceConic



class Optical_element(object):

    def __init__(self,p,q,theta=0.0,alpha=0.0):
        self.p = p
        self.q = q
        self.theta = theta
        self.alpha = alpha
        self.counter=0
        self.type="None"
        self.bound = None

        #
        # IF ELEMENT IS native
        self.R = None  # SPHERICAL
        self.ccc_object = None # conic
        self.fx = None
        self.fz = None



    def set_parameters(self,p,q,theta=0,alpha=0):
            self.p = p
            self.q = q
            self.theta = theta
            self.alpha = alpha




    #
    # native
    #

    @classmethod
    def initialize_as_plane_mirror(cls, p, q,theta=0., alpha=0.):

        plane_mirror=Optical_element(p,q,theta,alpha)
        plane_mirror.type="Plane mirror"

        return plane_mirror


    @classmethod
    def initialize_as_spherical_mirror(cls, p, q,theta=0. , alpha=0. , R=None):

        spherical_mirror=Optical_element(p,q,theta,alpha)

        if R is None:
            spherical_mirror.R=2*p*q/(p+q)/np.cos(theta)
        else:
            spherical_mirror.R = R

        spherical_mirror.type="Spherical mirror"
        print(spherical_mirror.R)

        return spherical_mirror


    @classmethod
    def ideal_lens(clsc, p, q, fx=None, fz=None):

        oe=Optical_element(p,q,0.0,0.0)
        oe.type="Ideal lens"
        if fx is None:\
            oe.fx = p*q/(p+q)
        else:
            oe.fx = fx

        if fz is None:
            oe.fz = p*q/(p+q)
        else:
            oe.fz = fz

        return oe


    #
    # conic
    #

    @classmethod
    def initialize_from_coefficients(cls):
        if np.array(cls.ccc_object).size != 10:
            raise Exception("Invalid coefficients (dimension must be 10)")
        # return Optical_element(ccc=ccc)
        else:
            cls.ccc_object = SurfaceConic.initialize_from_coefficients(cls)
            cls.type = "Surface conical mirror"


    @classmethod
    def initialize_as_surface_conic_plane(cls,p,q,theta=0.0,alpha=0.0):
        oe = Optical_element(p,q,theta,alpha)
        oe.type = "Surface conical mirror"
        oe.ccc_object = SurfaceConic(np.array([0,0,0,0,0,0,0,0,-1.,0]))
        return oe


    @classmethod
    def initialize_my_hyperboloid(cls,p,q,theta=0.0,alpha=0.0, wolter=0.):
        oe=Optical_element(p,q,theta,alpha)
        oe.type = "Surface conical mirror 1"
        #oe.ccc_object = SurfaceConic(np.array([-1.,-1.,1.,0,0,0,0,0,0,-1.]))
        a=1.
        z0= (25.+np.sqrt(2)*a)*wolter
        oe.ccc_object = SurfaceConic(np.array([-1/a**2, -1/a**2, 1/a**2, 0, 0, 0, 0., 0., -2*z0, -z0**2-1.]))
        return oe
    #
    # initializers from focal distances
    #

    @classmethod
    def initialize_as_surface_conic_sphere_from_focal_distances(cls, p, q, theta=0., alpha=0., cylindrical=0, cylangle=0.0,
                                                  switch_convexity=0):
        oe=Optical_element(p,q,theta,alpha)
        oe.type="Surface conical mirror"
        oe.ccc_object = SurfaceConic()
        oe.ccc_object.set_sphere_from_focal_distances(p, q, np.pi/2-theta)
        if cylindrical:
            oe.ccc_object.set_cylindrical(cylangle)
        if switch_convexity:
            oe.ccc_object.switch_convexity()
        return oe


    @classmethod
    def initialize_as_surface_conic_ellipsoid_from_focal_distances(cls, p, q, theta=0., alpha=0., cylindrical=0, cylangle=0.0,
                                                     switch_convexity=0):
        oe=Optical_element(p,q,theta,alpha)
        oe.type="Surface conical mirror"
        oe.ccc_object = SurfaceConic()
        oe.ccc_object.set_ellipsoid_from_focal_distances(p, q, np.pi/2-theta)
        if cylindrical:
            oe.ccc_object.set_cylindrical(cylangle)
        if switch_convexity:
            oe.ccc_object.switch_convexity()
        return oe


    @classmethod
    def initialize_as_surface_conic_paraboloid_from_focal_distances(cls, p, q, theta=0., alpha=0,  infinity_location="q", cylindrical=0, cylangle=0.0,
                                                      switch_convexity=0):
        oe=Optical_element(p,q,theta,alpha)
        oe.type="Surface conical mirror"
        oe.ccc_object = SurfaceConic()
        oe.ccc_object.set_paraboloid_from_focal_distance(p, q, np.pi/2-theta, infinity_location)
        if cylindrical:
            oe.ccc_object.set_cylindrical(cylangle)
        if switch_convexity:
            oe.ccc_object.switch_convexity()
        return oe


    @classmethod
    def initialize_as_surface_conic_hyperboloid_from_focal_distances(cls, p, q, theta=0., alpha=0., cylindrical=0, cylangle=0.0,
                                                       switch_convexity=0):
        oe=Optical_element(p,q,theta,alpha)
        oe.type="Surface conical mirror"
        oe.ccc_object = SurfaceConic()
        oe.ccc_object.set_hyperboloid_from_focal_distances(p, q, np.pi/2-theta)
        if cylindrical:
            oe.ccc_object.set_cylindrical(cylangle)
        if switch_convexity:
            oe.ccc_object.switch_convexity()
        return oe





    #
    # methods to "trace"
    #


    def trace_optical_element(self, beam1):


        beam=beam1.duplicate()
        beam.counter=beam.counter+1

        #
        # change beam to o.e. frame
        #
        self.rotation_to_the_optical_element(beam)
        self.translation_to_the_optical_element(beam)
        self.intersection_with_optical_element(beam)

        #beam.z=(beam.x**2+beam.y**2)/1000

        beam.plot_yx()
        plt.title("footprint")

        self.output_direction_from_optical_element(beam)


        self.rotation_to_the_screen(beam)

        self.translation_to_the_screen(beam)
        self.intersection_with_the_screen(beam)

        return beam


    def intersection_with_optical_element(self, beam):
        if self.type == "Plane mirror":
            self._intersection_with_plane_mirror(beam)
        elif self.type == "Ideal lens":
            self._intersection_with_plane_mirror(beam)
        elif self.type == "Spherical mirror":
            self._intersection_with_spherical_mirror(beam)
        elif self.type == "Surface conical mirror":
            self._intersection_with_surface_conic(beam)                     #self.intersection_with_surface_conic(beam)
        elif self.type =="Surface conical mirror 1":
            self._new_intersection(beam)

    def intersection_with_the_screen(self,beam):
        t= -beam.y/beam.vy
        beam.x = beam.x+beam.vx*t
        beam.y = beam.y+beam.vy*t
        beam.z = beam.z+beam.vz*t




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

        if self.type == "Ideal lens":
            position.rotation((np.pi/2-self.theta),"x")
            velocity.rotation((np.pi/2-self.theta),"x")
        else:
            position.rotation(-(np.pi/2-self.theta),"x")
            velocity.rotation(-(np.pi/2-self.theta),"x")
        [beam.x,beam.y,beam.z] = [position.x,position.y,position.z]
        [beam.vx,beam.vy,beam.vz] = [velocity.x,velocity.y,velocity.z]



    def translation_to_the_screen(self,beam):
        beam.y=beam.y-self.q


    def mirror_output_direction(self,beam):
        position = Vector(beam.x, beam.y, beam.z)
        if self.type == "Plane mirror":
            normal = position.plane_normal()
        elif self.type == "Spherical mirror":
            normal = position.spherical_normal(self.R)
        elif self.type == "Surface conical mirror":
            normal = position.surface_conic_normal(self.ccc_object.get_coefficients())
        elif self.type == "Surface conical mirror 1":
            normal = position.surface_conic_normal(self.ccc_object.get_coefficients())
        normal.normalization()
        velocity = Vector(beam.vx, beam.vy, beam.vz)
        vperp = velocity.perpendicular_component(normal)
        v2 = velocity.sum(vperp)
        v2 = v2.sum(vperp)
        [beam.vx, beam.vy, beam.vz] = [v2.x, v2.y, v2.z]




    def lens_output_direction(self,beam):

        gamma = np.arctan( beam.x/self.fx)
        alpha = np.arctan( -beam.y/self.fz)


        velocity = Vector(beam.vx, beam.vy, beam.vz)
        velocity.rotation(gamma, "y")
        velocity.rotation(alpha, "x")

        [beam.vx, beam.vy, beam.vz] = [velocity.x, velocity.y, velocity.z]




    def output_direction_from_optical_element(self, beam):

        if self.type == "Ideal lens":
            self.lens_output_direction(beam)
        else:
            self.mirror_output_direction(beam)

#    def output_direction_from_optical_element(self, beam):
#
#        position=Vector(beam.x,beam.y,beam.z)
#
#        if self.type == "Plane mirror":
#            normal=position.plane_normal()
#        elif self.type == "Spherical mirror":
#            normal=position.spherical_normal(self.R)
#        elif self.type == "Surface conical mirror":
#            normal = position.surface_conic_normal(self.ccc_object.get_coefficients())
#        elif self.type =="Surface conical mirror 1":
#            normal = position.surface_conic_normal(self.ccc_object.get_coefficients())
#
#
#        normal.normalization()
#
#        velocity=Vector(beam.vx,beam.vy,beam.vz)
#        vperp=velocity.perpendicular_component(normal)
#        v2=velocity.sum(vperp)
#        v2=v2.sum(vperp)
#
#        [beam.vx,beam.vy,beam.vz] = [ v2.x, v2.y, v2.z]
#

    def rectangular_bound(self,bound):
        self.bound=bound


    #
    # methods to intersection (all private)
    #

    def _intersection_with_plane_mirror(self, beam):
        t=-beam.z/beam.vz
        beam.x = beam.x+beam.vx*t
        beam.y = beam.y+beam.vy*t
        beam.z = beam.z+beam.vz*t
        if self.bound != None:
            indices=np.where(beam.flag>=0)
            beam.flag[indices] = np.zeros(beam.flag[indices].size)
            beam.flag[indices] = beam.flag[indices]+(np.sign(beam.x[indices]-self.bound.xmin*np.ones(beam.flag[indices].size))-1)/2
            beam.flag[indices] = beam.flag[indices]+(np.sign(beam.y[indices]-self.bound.ymin*np.ones(beam.flag[indices].size))-1)/2
            beam.flag[indices] = beam.flag[indices]+(np.sign(self.bound.xmax*np.ones(beam.flag[indices].size)-beam.x[indices])-1)/2
            beam.flag[indices] = beam.flag[indices]+(np.sign(self.bound.ymax*np.ones(beam.flag[indices].size)-beam.y[indices])-1)/2
            beam.flag[indices] = np.sign(np.sign(beam.flag[indices])+0.5)*beam.counter


    def _intersection_with_spherical_mirror(self, beam, ):
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

            if self.bound != None:
                indices = np.where(beam.flag >= 0)
                beam.flag[indices] = np.zeros(beam.flag[indices].size)
                beam.flag[indices] = beam.flag[indices] + (np.sign(beam.x[indices] - self.bound.xmin * np.ones(beam.flag[indices].size)) - 1) / 2
                beam.flag[indices] = beam.flag[indices] + (np.sign(beam.y[indices] - self.bound.ymin * np.ones(beam.flag[indices].size)) - 1) / 2
                beam.flag[indices] = beam.flag[indices] + (np.sign(self.bound.xmax * np.ones(beam.flag[indices].size) - beam.x[indices]) - 1) / 2
                beam.flag[indices] = beam.flag[indices] + (np.sign(self.bound.ymax * np.ones(beam.flag[indices].size) - beam.y[indices]) - 1) / 2
                beam.flag[indices] = np.sign(np.sign(beam.flag[indices]) + 0.5) * beam.counter


    def _intersection_with_surface_conic(self, beam):

        t_source = 0.0
        [t1, t2, flag] = self.ccc_object.calculate_intercept(np.array([beam.x, beam.y, beam.z]),
                                                        np.array([beam.vx, beam.vy, beam.vz]))

        #if np.abs(np.mean(t1)) <= np.abs(np.mean(t2)):
        #    t=t2
        #else:
        #    t=t2

        t=np.ones(beam.N)

        for i in range (beam.N):
            if np.abs(t1[i]-t_source >= np.abs(t2[i]-t_source)):
                t[i]=t1[i]
            else:
                t[i]=t2[i]



        beam.x = beam.x + beam.vx * t
        beam.y = beam.y + beam.vy * t
        beam.z = beam.z + beam.vz * t

        if self.bound != None:
            indices=np.where(beam.flag>=0)
            beam.flag[indices] = np.zeros(beam.flag[indices].size)
            beam.flag[indices] = beam.flag[indices]+(np.sign(beam.x[indices]-self.bound.xmin*np.ones(beam.flag[indices].size))-1)/2
            beam.flag[indices] = beam.flag[indices]+(np.sign(beam.y[indices]-self.bound.ymin*np.ones(beam.flag[indices].size))-1)/2
            beam.flag[indices] = beam.flag[indices]+(np.sign(self.bound.xmax*np.ones(beam.flag[indices].size)-beam.x[indices])-1)/2
            beam.flag[indices] = beam.flag[indices]+(np.sign(self.bound.ymax*np.ones(beam.flag[indices].size)-beam.y[indices])-1)/2
            beam.flag[indices] = np.sign(np.sign(beam.flag[indices])+0.5)*beam.counter



    def _new_intersection(self, beam):



        ccc=self.ccc_object.get_coefficients()
        z0= -ccc[8]/2
        a= -beam.vx**2-beam.vy**2+beam.vz**2
        b= -beam.x*beam.vx-beam.y*beam.vy+beam.z*beam.vz-z0*beam.vz
        c= -beam.x**2-beam.y**2+beam.z**2-1+z0**2-2*beam.z*z0



        t=(-b+np.sqrt(b**2-a*c))/a
        beam.x = beam.x + beam.vx * t
        beam.y = beam.y + beam.vy * t
        beam.z = beam.z + beam.vz * t





    def trace_Wolter_1(self, beam1):


        beam=beam1.duplicate()
        beam.counter=beam.counter+1

        #
        # change beam to o.e. frame
        #
        self.rotation_to_the_optical_element(beam)
        self.translation_to_the_optical_element(beam)
        self.intersection_with_optical_element(beam)      #####intersection with the paraboloid


        self.output_direction_from_optical_element(beam)

        hyper = Optical_element.initialize_my_hyperboloid(p=25,q=0,theta=90*np.pi/180,alpha=0,wolter=1)                                                     ##### we have to introduce the hyperboloid
        hyper.rotation_to_the_optical_element(beam)
        hyper.intersection_with_optical_element(beam)                                                     #####intersection with the hyperboloid with neither rotation and translation
        hyper.output_direction_from_optical_element(beam)                                                     ##### output direction w.r.t. the hyperboloid mirror
        beam.plot_ypzp()
        plt.title("ypzp")

        self.q=self.q+2*np.sqrt(2)
        self.rotation_to_the_screen(beam)


        self.translation_to_the_screen(beam)
        self.intersection_with_the_screen(beam)
#
        return beam


    def trace_Wolter_2(self, beam1):
        beam = beam1.duplicate()
        beam.counter = beam.counter + 1

        #
        # change beam to o.e. frame
        #
        self.rotation_to_the_optical_element(beam)
        self.translation_to_the_optical_element(beam)
        self.intersection_with_optical_element(beam)  #####intersection with the paraboloid

        self.output_direction_from_optical_element(beam)
        beam.plot_ypzp()

        hyper = Optical_element.initialize_my_hyperboloid(p=0, q=0, theta=90 * np.pi / 180,
                                                          alpha=0)  ##### we have to introduce the hyperboloid
        hyper.rotation_to_the_optical_element(beam)
        hyper.intersection_with_optical_element(
            beam)  #####intersection with the hyperboloid with neither rotation and translation
        hyper.output_direction_from_optical_element(beam)  ##### output direction w.r.t. the hyperboloid mirror
        beam.plot_ypzp()



        t = - beam.x / beam.vx

        beam.x = beam.x + beam.vx * t
        beam.y = beam.y + beam.vy * t
        beam.z = beam.z + beam.vz * t


        #self.q = 25.
        #self.rotation_to_the_screen(beam)

        #self.translation_to_the_screen(beam)
        #self.intersection_with_the_screen(beam)

        return beam


    def trace_ideal_lens(self,beam1):

        beam=beam1.duplicate()


        t = self.p / beam.vy
        beam.x = beam.x + beam.vx * t
        beam.y = beam.y + beam.vy * t
        beam.z = beam.z + beam.vz * t

        gamma = np.arctan( beam.x/self.fx)
        alpha = np.arctan(-beam.z/self.fz)


        velocity = Vector(beam.vx, beam.vy, beam.vz)
        velocity.rotation(gamma, "z")
        velocity.rotation(alpha, "x")
        [beam.vx, beam.vy, beam.vz] = [velocity.x, velocity.y, velocity.z]


        #self.q=0
        t = self.q / beam.vy
        beam.x = beam.x + beam.vx * t
        beam.y = beam.y + beam.vy * t
        beam.z = beam.z + beam.vz * t

        return beam



class CompoundOpticalElement(Optical_element):

    def __init__(self,k=2):
        self.k=k          #### Number of optical element

        self.oe1 = None
        self.oe2 = None
        self.oe3 = None
        self.oe4 = None
        self.oe5 = None
        self.oe6 = None
        self.oe7 = None
        self.oe8 = None
        self.oe9 = None



    @classmethod
    def initialiaze_as_wolter_1(cls,p1,q1,theta1):
        alpha1=0.

        wolter = CompoundOpticalElement(2)
        wolter.theta=theta1
        wolter.oe1 = Optical_element.initialize_as_surface_conic_paraboloid_from_focal_distances(p1,q1,theta1,0,"p")
        wolter.oe2 = Optical_element.initialize_my_hyperboloid(p=q1,q=0.,theta=90*np.pi/180,alpha=0,wolter=1)

        return wolter


    def trace_wolter_compound_optical_element(self,beam1):

        beam=beam1.duplicate()

        self.oe1.rotation_to_the_optical_element(beam)
        self.oe1.translation_to_the_optical_element(beam)
        self.oe1.intersection_with_optical_element(beam)      #####intersection with the paraboloid
        self.oe1.output_direction_from_optical_element(beam)

        self.oe2.rotation_to_the_optical_element(beam)
        self.oe2.intersection_with_optical_element(beam)
        self.oe2.output_direction_from_optical_element(beam)

        beam.plot_ypzp()
        plt.title("ypzp")

        self.oe1.q=self.oe1.q+2*np.sqrt(2)
        self.oe1.rotation_to_the_screen(beam)
        self.oe1.translation_to_the_screen(beam)
        self.oe1.intersection_with_the_screen(beam)

        return beam