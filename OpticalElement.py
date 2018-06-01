import numpy as np
import matplotlib.pyplot as plt
from Vector import Vector
from SurfaceConic import SurfaceConic



class Optical_element(object):

    def __init__(self,p=0. ,q=0. ,theta=0.0,alpha=0.0):
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
        self.focal = None
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
    def initialize_my_hyperboloid(cls,p,q,theta=0.0,alpha=0.0, wolter=None, z0=0., distance_of_focalization=0.):
        oe=Optical_element(p,q,theta,alpha)
        oe.type = "My hyperbolic mirror"
        a=q/np.sqrt(2)
        if wolter==1:
            a = abs(z0-distance_of_focalization)/np.sqrt(2)
        if wolter == 2:
            a = abs(z0-distance_of_focalization)/np.sqrt(2)
        if wolter == 3:
            a = distance_of_focalization/np.sqrt(2)
        oe.ccc_object = SurfaceConic(np.array([-1, -1, 1, 0, 0, 0, 0., 0., -2*z0, -z0**2-a**2.]))
        return oe

    @classmethod
    def initialize_as_surface_conic_from_coefficients(cls, ccc):
        oe = Optical_element()
        oe.type = "Surface conical mirror"
        oe.ccc_object = SurfaceConic(ccc)
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
    def initialize_as_surface_conic_paraboloid_from_focal_distances(cls, p, q, theta=0., alpha=0,  infinity_location="q", focal=None, cylindrical=0, cylangle=0.0,
                                                      switch_convexity=0):
        oe=Optical_element(p,q,theta,alpha)
        oe.type="Surface conical mirror"
        oe.focal=focal
        oe.ccc_object = SurfaceConic()
        if focal is None:
            oe.ccc_object.set_paraboloid_from_focal_distance(p, q, np.pi/2-theta, infinity_location)
        else:
            oe.ccc_object.set_paraboloid_from_focal_distance(p, focal, np.pi/2-theta, infinity_location)
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

        self.effect_of_optical_element(beam)

        if self.type != "My hyperbolic mirror":
            self.effect_of_the_screen(beam)

        return beam






    def effect_of_optical_element(self,beam):

        self.rotation_to_the_optical_element(beam)
        self.translation_to_the_optical_element(beam)
        self.intersection_with_optical_element(beam)

        beam.plot_yx(0)
        plt.title("footprint %s" %(self.type))

        beam.plot_ypzp()
        plt.title("Before mirror effect")
        self.output_direction_from_optical_element(beam)
        beam.plot_ypzp()
        plt.title("After mirror effect")

    def effect_of_the_screen(self,beam):

        self.rotation_to_the_screen(beam)
        self.translation_to_the_screen(beam)
        self.intersection_with_the_screen(beam)


    def intersection_with_optical_element(self, beam):
        if self.type == "Plane mirror":
            self._intersection_with_plane_mirror(beam)
        elif self.type == "Ideal lens":
            self._intersection_with_plane_mirror(beam)
        elif self.type == "Spherical mirror":
            self._intersection_with_spherical_mirror(beam)
        elif self.type == "Surface conical mirror":
            self._intersection_with_surface_conic(beam)                     #self.intersection_with_surface_conic(beam)
        elif self.type =="My hyperbolic mirror":
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
        elif self.type == "My hyperbolic mirror":
            normal = position.surface_conic_normal(self.ccc_object.get_coefficients())
        elif self.type == "Surface conical mirror 2":
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

        indices = beam.flag >=0
        beam.flag[indices] = beam.flag[indices] + 1

        if self.bound != None:
            position_x=beam.x.copy()
            indices = np.where(position_x < self.bound.xmin)
            position_x[indices] = 0
            indices = np.where(position_x > self.bound.xmax)
            position_x[indices] = 0
            indices = np.where(position_x == 0)
            beam.flag[indices] = -1*beam.flag[indices]

            position_y=beam.y.copy()
            indices = np.where(position_y < self.bound.ymin)
            position_y[indices] = 0
            indices = np.where(position_y > self.bound.ymax)
            position_y[indices] = 0
            indices = np.where(position_y == 0)
            beam.flag[indices] = -1*beam.flag[indices]





        #if self.bound != None:
        #    indices=np.where(beam.flag>=0)
        #    beam.flag[indices] = np.zeros(beam.flag[indices].size)
        #    beam.flag[indices] = beam.flag[indices]+(np.sign(beam.x[indices]-self.bound.xmin*np.ones(beam.flag[indices].size))-1)/2
        #    beam.flag[indices] = beam.flag[indices]+(np.sign(beam.y[indices]-self.bound.ymin*np.ones(beam.flag[indices].size))-1)/2
        #    beam.flag[indices] = beam.flag[indices]+(np.sign(self.bound.xmax*np.ones(beam.flag[indices].size)-beam.x[indices])-1)/2
        #    beam.flag[indices] = beam.flag[indices]+(np.sign(self.bound.ymax*np.ones(beam.flag[indices].size)-beam.y[indices])-1)/2
        #    beam.flag[indices] = np.sign(np.sign(beam.flag[indices])+0.5)*beam.counter


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
                indices0 = indices
                beam.flag[indices] = np.zeros(beam.flag[indices].size)
                beam.flag[indices] =  (np.sign(beam.x[indices] - self.bound.xmin * np.ones(beam.flag[indices].size)) - 1) / 2
                indices = np.where(beam.flag >= 0)
                beam.flag[indices] = np.zeros(beam.flag[indices].size)
                beam.flag[indices] =  (np.sign(beam.y[indices] - self.bound.ymin * np.ones(beam.flag[indices].size)) - 1) / 2
                indices = np.where(beam.flag >= 0)
                beam.flag[indices] = np.zeros(beam.flag[indices].size)
                beam.flag[indices] = (np.sign(self.bound.xmax * np.ones(beam.flag[indices].size) - beam.x[indices]) - 1) / 2
                indices = np.where(beam.flag >= 0)
                beam.flag[indices] = np.zeros(beam.flag[indices].size)
                beam.flag[indices] =  (np.sign(self.bound.ymax * np.ones(beam.flag[indices].size) - beam.y[indices]) - 1) / 2

                beam.flag[indices0] = np.sign(np.sign(beam.flag[indices0]) + 0.5) * beam.counter

#            if self.bound != None:
#                indices = np.where(beam.flag >= 0)
#                print(np.size(indices))
#                beam.flag[indices] = np.zeros(beam.flag[indices].size)
#                beam.flag[indices] = beam.flag[indices] * 0 + (
#                            np.sign(beam.x[indices] - self.bound.xmin * np.ones(beam.flag[indices].size)) - 1) / 2
#                indices = np.where(beam.flag >= 0)
#                print(np.size(indices))
#                beam.flag[indices] = np.zeros(beam.flag[indices].size)
#                beam.flag[indices] = beam.flag[indices] * 0 + (
#                            np.sign(beam.y[indices] - self.bound.ymin * np.ones(beam.flag[indices].size)) - 1) / 2
#                indices = np.where(beam.flag >= 0)
#                print(np.size(indices))
#                beam.flag[indices] = np.zeros(beam.flag[indices].size)
#                beam.flag[indices] = beam.flag[indices] * 0 + (
#                            np.sign(self.bound.xmax * np.ones(beam.flag[indices].size) - beam.x[indices]) - 1) / 2
#                indices = np.where(beam.flag >= 0)
#                print(np.size(indices))
#                beam.flag[indices] = np.zeros(beam.flag[indices].size)
#                beam.flag[indices] = beam.flag[indices] * 0 + (
#                            np.sign(self.bound.ymax * np.ones(beam.flag[indices].size) - beam.y[indices]) - 1) / 2
#                beam.flag[indices] = np.sign(np.sign(beam.flag[indices]) + 0.5) * beam.counter
#


            print(beam.flag)
            print(np.mean(beam.flag))


    def _intersection_with_surface_conic(self, beam):


        indices = beam.flag >=0
        beam.flag[indices] = beam.flag[indices] + 1
        counter = beam.flag[indices][0]



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



        beam.counter = 1

        beam.x = beam.x + beam.vx * t
        beam.y = beam.y + beam.vy * t
        beam.z = beam.z + beam.vz * t


        if self.bound != None:
            position_x=beam.x.copy()
            indices = np.where(position_x < self.bound.xmin)
            position_x[indices] = 0
            indices = np.where(position_x > self.bound.xmax)
            position_x[indices] = 0
            indices = np.where(position_x == 0)
            beam.flag[indices] = -1*counter

            position_y=beam.y.copy()
            indices = np.where(position_y < self.bound.ymin)
            position_y[indices] = 0
            indices = np.where(position_y > self.bound.ymax)
            position_y[indices] = 0
            indices = np.where(position_y == 0)
            beam.flag[indices] = -1*counter




        #if self.bound != None:
        #    indices=np.where(beam.flag>=0)
        #    beam.flag[indices] = np.zeros(beam.flag[indices].size)
        #    beam.flag[indices] = beam.flag[indices]+(np.sign(beam.x[indices]-self.bound.xmin*np.ones(beam.flag[indices].size))-1)/2
        #    beam.flag[indices] = beam.flag[indices]+(np.sign(beam.y[indices]-self.bound.ymin*np.ones(beam.flag[indices].size))-1)/2
        #    beam.flag[indices] = beam.flag[indices]+(np.sign(self.bound.xmax*np.ones(beam.flag[indices].size)-beam.x[indices])-1)/2
        #    beam.flag[indices] = beam.flag[indices]+(np.sign(self.bound.ymax*np.ones(beam.flag[indices].size)-beam.y[indices])-1)/2
        #    beam.flag[indices] = np.sign(np.sign(beam.flag[indices])+0.5)*beam.counter


        #if self.bound != None:
        #    indices=np.where(beam.flag>=0)
        #    beam.flag[indices] = np.zeros(beam.flag[indices].size)
        #    beam.flag[indices] = beam.flag[indices]+(np.sign(beam.x[indices]-self.bound.xmin*np.ones(beam.flag[indices].size))-1)/2
        #    beam.flag[indices] = beam.flag[indices]+(np.sign(beam.y[indices]-self.bound.ymin*np.ones(beam.flag[indices].size))-1)/2
        #    beam.flag[indices] = beam.flag[indices]+(np.sign(self.bound.xmax*np.ones(beam.flag[indices].size)-beam.x[indices])-1)/2
        #    beam.flag[indices] = beam.flag[indices]+(np.sign(self.bound.ymax*np.ones(beam.flag[indices].size)-beam.y[indices])-1)/2
        #    beam.flag[indices] = np.sign(np.sign(beam.flag[indices])+0.5)*beam.counter



    def _new_intersection(self, beam):


        ccc=self.ccc_object.get_coefficients()
        z0= -ccc[8]/2
        aa=np.sqrt(-z0**2-ccc[9])
        a= -beam.vx**2-beam.vy**2+beam.vz**2
        b= -beam.x*beam.vx-beam.y*beam.vy+beam.z*beam.vz-z0*beam.vz
        c= -beam.x**2-beam.y**2+beam.z**2-1*aa**2+z0**2-2*beam.z*z0





        #c = self.ccc_object.get_coefficients()

        #a = c[0] * beam.vx ** 2 + c[1] * beam.vy ** 2 + c[2] * beam.vz ** 2 + c[3] * beam.vx * beam.vy + c[4] * beam.vy * beam.vz + c[5] * beam.vx * beam.vz

        #b = 2 * c[0] * beam.x * beam.vx + 2 * c[1] * beam.y * beam.vy + 2 * c[2] * beam.z * beam.vz + c[3] * (beam.x * beam.vy + beam.y * beam.vx) + c[4] * (beam.y * beam.vz + beam.z * beam.vy) \
        #    + c[5] * (beam.x * beam.vz + beam.z * beam.vx) + c[6] * beam.vx + c[7] * beam.vy + c[8] * beam.vz

        #cc = c[0] * beam.x ** 2 + c[1] * beam.y ** 2 + c[2] * beam.z ** 2 + c[3] * beam.x * beam.y + c[4] * beam.y * beam.z + c[5] * beam.x * beam.z + c[6] * beam.x + c[7] * beam.y + c[8] * beam.z + c[9]

        #if np.mean(a) < 1e-13:
        #    t = -cc / b
        #else:
        #    t = (-b - np.sqrt(b ** 2 - 4 * a * cc)) / (2 * a)


        #print(np.mean(a), np.mean(b), np.mean(cc))


        t1 = (-b - np.sqrt(b ** 2 - a * c)) / a
        t2 = (-b + np.sqrt(b ** 2 - a * c)) / a

        if  np.mean(t1)*np.mean(t2)>1:
            if np.abs(np.mean(t1)) <= np.abs(np.mean(t2)):
                t = t1
            else:
                t = t2
        elif np.mean(t1) > 0:
            t=t1
        else:
            t=t2

        beam.x = beam.x + beam.vx * t
        beam.y = beam.y + beam.vy * t
        beam.z = beam.z + beam.vz * t


    def _new_intersection2(self, beam):



        ccc=self.ccc_object.get_coefficients()
        aa=1/np.sqrt(-ccc[0])
        y0= -ccc[7]/2
        a= -beam.vx**2-beam.vy**2+beam.vz**2
        b= -beam.x*beam.vx-beam.y*beam.vy+beam.z*beam.vz-y0*beam.vy
        c= -beam.x**2-beam.y**2+beam.z**2-1*aa**2+y0**2-2*beam.y*y0

        t1 = (-b - np.sqrt(b ** 2 - a * c)) / a
        t2 = (-b + np.sqrt(b ** 2 - a * c)) / a

        if  np.mean(t1)*np.mean(t2)>1:
            if np.abs(np.mean(t1)) <= np.abs(np.mean(t2)):
                t = t1
            else:
                t = t2
        elif np.mean(t1) > 0:
            t=t1
        else:
            t=t2


        beam.x = beam.x + beam.vx * t
        beam.y = beam.y + beam.vy * t
        beam.z = beam.z + beam.vz * t



    def output_frame_wolter(self,beam):
        test_ray = Vector(np.mean(beam.vx), np.mean(beam.vy), np.mean(beam.vz))
        #test_ray = Vector(beam.vx[0], beam.vy[0], beam.vz[0])
        velocity = Vector (beam.vx, beam.vy, beam.vz)
        test_ray.normalization()

        s = 0.00000000000000000000000000000000000000000000000000000000001
        t = 2.
        if np.abs(test_ray.x) < 1e-2 and np.abs(test_ray.y) < 1e-2:
            print("1")
            ort = Vector(s, 0, 0.)
        elif np.abs(test_ray.z) < 1e-2 and np.abs(test_ray.y) < 1e-2:
            print("2")
            ort = Vector(0., s, 0)
        elif np.abs(test_ray.x) < 1e-2 and np.abs(test_ray.z) < 1e-2:
            print("3")
            ort = Vector(s, 0., 0.)
        elif np.abs(test_ray.x) < 1e-10:
            print("4")
            ort = Vector(s, t, -test_ray.y / test_ray.z * t)
        elif np.abs(test_ray.y) < 1e-10:
            print("5")
            ort = Vector(t, s, -test_ray.x / test_ray.z * t)
        elif np.abs(test_ray.z) < 1e-10:
            print("6")
            ort = Vector(t, -test_ray.x / test_ray.y * t, s)
        else:
            print("last possibility")
            ort = Vector(s, t, -(test_ray.x * s + test_ray.y * t) / test_ray.z)

        ort.normalization()

        if np.abs(test_ray.x) < 1e-2 and np.abs(test_ray.y) < 1e-2:
            print("1")
            perp = Vector(0., 1., 0.)
        elif np.abs(test_ray.z) < 1e-2 and np.abs(test_ray.y) < 1e-2:
            print("2")
            perp = Vector(0., 0., 1.)
        elif np.abs(test_ray.x) < 1e-2 and np.abs(test_ray.z) < 1e-2:
            print("3")
            perp = Vector(0., 0., 1.)
        elif np.abs(test_ray.x) < 1e-10:
            print("4")
            t = s*ort.z/(test_ray.x/test_ray.y*ort.y-ort.x)
            perp = Vector(s, t, -test_ray.y / test_ray.z * t)
        elif np.abs(test_ray.y) < 1e-10:
            print("5")
            t = s*ort.y/(test_ray.x/test_ray.z*ort.z-ort.x)
            perp = Vector(t, s, -test_ray.x / test_ray.z * t)
        elif np.abs(test_ray.z) < 1e-10:
            print("6")
            t = s*ort.x/(test_ray.y/test_ray.z*ort.z-ort.y)
            perp = Vector(t, -test_ray.x / test_ray.y * t, s)
        else:
            print("last possibility")
            t = s*(ort.z*test_ray.x/test_ray.z-ort.x)/(ort.y-ort.z*test_ray.y/test_ray.z)
            perp = Vector(s, t, -(test_ray.x * s + test_ray.y * t) / test_ray.z)

        perp.normalization()

        y = velocity.dot(test_ray)
        z = velocity.dot(ort)
        x = velocity.dot(perp)

        [beam.vx, beam.vy, beam.vz] = [x, y, z]


    def trace_Wolter_1(self, beam1, z0):

        beam=beam1.duplicate()
        beam.counter=beam.counter+1


        self.effect_of_optical_element(beam)
        self.q = 0.
        self.theta = 90.*np.pi/180
        self.effect_of_the_screen(beam)


        hyper = Optical_element.initialize_my_hyperboloid(p=0., q=z0 + (z0 - self.focal), theta=90*np.pi/180, alpha=0, wolter=1, z0=z0, distance_of_focalization=self.focal)
        hyper.effect_of_optical_element(beam)


        hyper.theta = 0.
        hyper.effect_of_the_screen(beam)

        self.output_frame_wolter(beam)


        return beam


    def trace_Wolter_2(self, beam1, z0=0.):
        beam = beam1.duplicate()
        beam.counter = beam.counter + 1


        self.effect_of_optical_element(beam)

        #hyper = Optical_element.initialize_my_hyperboloid(p=0., q=-self.focal, theta=90 * np.pi / 180, alpha=0, wolter=2, z0=z0, distance_of_focalization=self.focal)
        hyper = Optical_element.initialize_my_hyperboloid(p=0., q=-(self.focal-2*z0), theta=90 * np.pi / 180, alpha=0, wolter=2, z0=z0, distance_of_focalization=self.focal)

        hyper.effect_of_optical_element(beam)


        hyper.theta = 0.
        hyper.effect_of_the_screen(beam)




        self.output_frame_wolter(beam)

        return beam


    def trace_ideal_lens(self,beam1):

        beam=beam1.duplicate()


        t = self.p / beam.vy
        beam.x = beam.x + beam.vx * t
        beam.y = beam.y + beam.vy * t
        beam.z = beam.z + beam.vz * t

        gamma = np.arctan( beam.x/self.fx)
        alpha = np.arctan( -beam.z/self.fz)


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


