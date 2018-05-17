
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
        self.counter=0
        self.type=""

        self.ccc_object = ccc
        self.bound = None

    def set_parameters(self,p,q,theta=0,alpha=0,R=0):
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
    def initialize_as_spherical_mirror(cls, p, q,theta, alpha, R=None):

        spherical_mirror=Optical_element()
        spherical_mirror.p=p
        spherical_mirror.q=q
        spherical_mirror.theta=theta
        spherical_mirror.alpha=alpha
        spherical_mirror.R=R
        if R == None:
            spherical_mirror.R=2*p*q/(p+q)/np.cos(theta)
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
    def initialize_as_surface_conic_plane(cls,p,q,theta,alpha=0):
        oe = Optical_element()
        oe.p = p
        oe.q = q
        oe.theta = theta
        oe.alpha = alpha
        oe.type = "Surface conical mirror"
        oe.ccc_object = SurfaceConic(np.array([0,0,0,0,0,0,0,0,-1.,0]))
        return oe


    @classmethod
    def initialize_my_hyperboloid(cls,p,q,theta,alpha=0):
        oe=Optical_element()
        oe.p = p
        oe.q = q
        oe.theta = theta
        oe.alpha = alpha
        oe.type = "Surface conical mirror 1"
        #oe.ccc_object = SurfaceConic(np.array([-1.,-1.,1.,0,0,0,0,0,0,-1.]))
        z0= (25.+np.sqrt(2))*1
        a=1.
        oe.ccc_object = SurfaceConic(np.array([-1/a**2, -1/a**2, 1/a**2, 0, 0, 0, 0., 0., -2*z0, -z0**2-1.]))
        return oe
    #
    # initializers from focal distances
    #

    @classmethod
    def initialize_as_surface_conic_sphere_from_focal_distances(cls, p, q, theta, alpha=0, cylindrical=0, cylangle=0.0,
                                                  switch_convexity=0):
        oe=Optical_element()
        oe.p=p
        oe.q=q
        oe.theta=theta
        oe.alpha=alpha
        oe.type="Surface conical mirror"
        oe.ccc_object = SurfaceConic()
        oe.ccc_object.set_sphere_from_focal_distances(p, q, np.pi/2-theta)
        if cylindrical:
            oe.ccc_object.set_cylindrical(cylangle)
        if switch_convexity:
            oe.ccc_object.switch_convexity()
        return oe


    @classmethod
    def initialize_as_surface_conic_ellipsoid_from_focal_distances(cls, p, q, theta, alpha=0, cylindrical=0, cylangle=0.0,
                                                     switch_convexity=0):
        oe=Optical_element()
        oe.p=p
        oe.q=q
        oe.theta=theta
        oe.alpha=alpha
        oe.type="Surface conical mirror"
        oe.ccc_object = SurfaceConic()
        oe.ccc_object.set_ellipsoid_from_focal_distances(p, q, np.pi/2-theta)
        if cylindrical:
            oe.ccc_object.set_cylindrical(cylangle)
        if switch_convexity:
            oe.ccc_object.switch_convexity()
        return oe


    @classmethod
    def initialize_as_surface_conic_paraboloid_from_focal_distances(cls, p, q, theta, alpha=0,  infinity_location="q", cylindrical=0, cylangle=0.0,
                                                      switch_convexity=0):
        oe=Optical_element()
        oe.p=p
        oe.q=q
        oe.theta=theta
        oe.alpha=alpha
        if infinity_location=="p":
            oe.type="Surface conical mirror p"
        else:
            oe.type = "Surface conical mirror p"
        oe.ccc_object = SurfaceConic()
        oe.ccc_object.set_paraboloid_from_focal_distance(p, q, np.pi/2-theta, infinity_location)
        if cylindrical:
            oe.ccc_object.set_cylindrical(cylangle)
        if switch_convexity:
            oe.ccc_object.switch_convexity()
        return oe


    @classmethod
    def initialize_as_surface_conic_hyperboloid_from_focal_distances(cls, p, q, theta, alpha=0, cylindrical=0, cylangle=0.0,
                                                       switch_convexity=0):
        oe=Optical_element()
        oe.p=p
        oe.q=q
        oe.theta=theta
        oe.alpha=alpha
        oe.type="Surface conical mirror"
        oe.ccc_object = SurfaceConic()
        oe.ccc_object.set_hyperboloid_from_focal_distances(p, q, np.pi/2-theta)
        if cylindrical:
            oe.ccc_object.set_cylindrical(cylangle)
        if switch_convexity:
            oe.ccc_object.switch_convexity()
        return oe





    @classmethod
    def ideal_lens(clsc, p, q):

        oe=Optical_element
        oe.p=p
        oe.q=q
        oe.type="ideal_lense"
        oe.fx=p*q/(p+q)
        oe.fz=p*q/(p+q)
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


    def intersection_with_plane_mirror(self,beam):
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


    def intersection_with_spherical_mirror(self,beam,):
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


    def intersection_with_surface_conic(self,beam):

        [t, flag] = self.ccc_object.calculate_intercept(np.array([beam.x, beam.y, beam.z]),
                                                        np.array([beam.vx, beam.vy, beam.vz]))
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

    def intersection_with_surface_conic_p(self,beam):

        c=self.ccc_object.get_coefficients()

        a=c[0]*beam.vx**2+c[1]*beam.vy**2+c[2]*beam.vz**2+c[3]*beam.vx*beam.vy+c[4]*beam.vy*beam.vz+c[5]*beam.vx*beam.vz

        b=2*c[0]*beam.x*beam.vx+2*c[1]*beam.y*beam.vy+2*c[2]*beam.z*beam.vz+c[3]*(beam.x*beam.vy+beam.y*beam.vx)+c[4]*(beam.y*beam.vz+beam.z*beam.vy) \
          + c[5]*(beam.x*beam.vz+beam.z*beam.vx)+c[6]*beam.vx+c[7]*beam.vy+c[8]*beam.vz

        cc=c[0]*beam.x**2+c[1]*beam.y**2+c[2]*beam.z**2+c[3]*beam.x*beam.y+c[4]*beam.y*beam.z+c[5]*beam.x*beam.z+c[6]*beam.x+c[7]*beam.y+c[8]*beam.z+c[9]

        if np.mean(a)<1e-13:
            t = -cc / b
        else:
             t=(-b-np.sqrt(b**2-4*a*cc))/(2*a)


        beam.x = beam.x + beam.vx * t
        beam.y = beam.y + beam.vy * t
        beam.z = beam.z + beam.vz * t




    def new_intersection(self,beam):

        z0= (25.+np.sqrt(2))*1
        a= -beam.vx**2-beam.vy**2+beam.vz**2
        b= -beam.x*beam.vx-beam.y*beam.vy+beam.z*beam.vz-z0*beam.vz
        c= -beam.x**2-beam.y**2+beam.z**2-1+z0**2-2*beam.z*z0

        t=(-b+np.sqrt(b**2-a*c))/a
        beam.x = beam.x + beam.vx * t
        beam.y = beam.y + beam.vy * t
        beam.z = beam.z + beam.vz * t

        print(beam.z)



    def output_direction_from_optical_element(self, beam):

        position=Vector(beam.x,beam.y,beam.z)

        if self.type == "Plane mirror":
            normal=position.plane_normal()
        elif self.type == "Spherical mirror":
            normal=position.spherical_normal(self.R)
        elif self.type == "Surface conical mirror":
            normal = position.surface_conic_normal(self.ccc_object.get_coefficients())
        elif self.type == "Surface conical mirror p":
            normal = position.surface_conic_normal(self.ccc_object.get_coefficients())
        elif self.type =="Surface conical mirror 1":
            normal = position.surface_conic_normal(self.ccc_object.get_coefficients())


        normal.normalization()

        if self.type == "Surface conical mirror 1":
            print("Boh!")
            normal.x=-normal.x
            normal.y=-normal.y
            normal.z=-normal.z

        print("Whoa")
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



    def intersection_with_optical_element(self, beam):
        if self.type == "Plane mirror":
            self.intersection_with_plane_mirror(beam)
        elif self.type == "Spherical mirror":
            self.intersection_with_spherical_mirror(beam)
        elif self.type == "Surface conical mirror":
            self.intersection_with_surface_conic(beam)
        elif self.type == "Surface conical mirror p":
            self.intersection_with_surface_conic_p(beam)
        elif self.type =="Surface conical mirror 1":
            self.new_intersection(beam)


    def rectangular_bound(self,bound):
        self.bound=bound



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

        hyper = Optical_element.initialize_my_hyperboloid(p=25,q=0,theta=90*np.pi/180,alpha=0)                                                     ##### we have to introduce the hyperboloid
        hyper.rotation_to_the_optical_element(beam)
        hyper.intersection_with_optical_element(beam)                                                     #####intersection with the hyperboloid with neither rotation and translation
        hyper.output_direction_from_optical_element(beam)                                                     ##### output direction w.r.t. the hyperboloid mirror
        beam.plot_ypzp()

        self.q=self.q+2*np.sqrt(2)+100
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
#
        #self.translation_to_the_screen(beam)
        #self.intersection_with_the_screen(beam)

        return beam




    def trace_ideal_lens(self,beam1):

        beam=beam1.duplicate()

        fx = self.p*self.q/(self.p+self.q)
        fz = self.p*self.q/(self.p+self.q)

        #fx=self.q
        #fz=self.q


        t = self.p / beam.vy
        beam.x = beam.x + beam.vx * t
        beam.y = beam.y + beam.vy * t
        beam.z = beam.z + beam.vz * t

        gamma = np.arctan( beam.x/fx)
        alpha = np.arctan(-beam.z/fz)

        velocity = Vector(beam.vx, beam.vy, beam.vz)
        velocity.rotation(gamma, "z")
        velocity.rotation(alpha, "x")
        [beam.vx, beam.vy, beam.vz] = [velocity.x, velocity.y, velocity.z]

        self.q=0
        t = self.q / beam.vy
        beam.x = beam.x + beam.vx * t
        beam.y = beam.y + beam.vy * t
        beam.z = beam.z + beam.vz * t

        return beam






#   position = Vector(beam.x,beam.y,beam.z)
#   velocity = Vector(beam.vx,beam.vy,beam.vz)
#   position.rotation(self.alpha,"y")
#   position.rotation(-(np.pi/2-self.theta),"x")
#   velocity.rotation(self.alpha,"y")
#   velocity.rotation(-(np.pi/2-self.theta),"x")
#   [beam.x,beam.y,beam.z] = [position.x,position.y,position.z]
#   [beam.vx,beam.vy,beam.vz] = [velocity.x,velocity.y,velocity.z]











