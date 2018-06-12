from Beam import Beam
#from Empty import Optical_element
from OpticalElement import Optical_element
from SurfaceConic import SurfaceConic
from Shape import BoundaryRectangle
import numpy as np
from SurfaceConic import SurfaceConic
import matplotlib.pyplot as plt



class CompoundOpticalElement(object):

    def __init__(self,oe_list=[],oe_name=""):
        self.oe = oe_list
        self.type = oe_name

    def append_oe(self,oe):
        self.oe.append(oe)

    def oe_number(self):
        return len(self.oe)

    def reset_oe_list(self):
        self.oe = []

    def set_type(self,name):
        self.type = name


    @classmethod
    def initialiaze_as_wolter_1(cls,p1,q1,z0):
        theta1 = 0.
        alpha1 = 0.
        print(q1)
        print(2*z0)
        print("dof=%f" %(2*z0-q1))

        #oe1 = Optical_element.initialize_as_surface_conic_paraboloid_from_focal_distances(p1,0.,theta1,alpha1,"p",2*z0-q1)
        oe1 = Optical_element.initialize_as_surface_conic_paraboloid_from_focal_distances(p1, 0., theta1, alpha1, "p", 2*z0-q1)
        #oe2 = Optical_element.initialize_my_hyperboloid(p=0.,q=q1,theta=90*np.pi/180,alpha=0,wolter=1, z0=z0, distance_of_focalization=2*z0-q1)
        oe2 = Optical_element.initialize_my_hyperboloid(p=0., q=q1, theta=90 * np.pi / 180, alpha=0, wolter=1, z0=z0,distance_of_focalization=2*z0-q1)

        return CompoundOpticalElement(oe_list=[oe1,oe2],oe_name="Wolter 1")


    @classmethod
    def initialiaze_as_wolter_1_with_two_parameters(cls,p1, R, theta):

        cp1 = -2 * R / np.tan(theta)
        cp2 = 2 * R * np.tan(theta)
        cp = max(cp1, cp2)
        f = cp / 4
        print("focal=%f" % (f))

        oe1 = Optical_element.initialize_as_surface_conic_paraboloid_from_focal_distances(p=p1, q=f, theta=0., alpha=0.,infinity_location="p")

        s1 = R / np.tan(2 * theta)
        s2 = R / np.tan(4 * theta)
        c = (s1 - s2) / 2
        z0 = f + c


        b1 = np.sqrt(
            0.5 * c ** 2 + 0.5 * R ** 2 + 0.5 * R ** 4 / cp ** 2 - R ** 2 * z0 / cp + 0.5 * z0 ** 2 - 0.5 / cp ** 2 * np.sqrt(
                (
                            -c ** 2 * cp ** 2 - cp ** 2 * R ** 2 - R ** 4 + 2 * cp * R ** 2 * z0 - cp ** 2 * z0 ** 2) ** 2 - 4 * cp ** 2 * (
                            c ** 2 * R ** 4 - 2 * c ** 2 * cp * R ** 2 * z0 + c ** 2 * cp ** 2 * z0 ** 2)))
        b2 = np.sqrt(
            0.5 * c ** 2 + 0.5 * R ** 2 + 0.5 * R ** 4 / cp ** 2 - R ** 2 * z0 / cp + 0.5 * z0 ** 2 + 0.5 / cp ** 2 * np.sqrt(
                (
                            -c ** 2 * cp ** 2 - cp ** 2 * R ** 2 - R ** 4 + 2 * cp * R ** 2 * z0 - cp ** 2 * z0 ** 2) ** 2 - 4 * cp ** 2 * (
                            c ** 2 * R ** 4 - 2 * c ** 2 * cp * R ** 2 * z0 + c ** 2 * cp ** 2 * z0 ** 2)))
        b = min(b1, b2)
        a = np.sqrt(c ** 2 - b ** 2)

        ccc = np.array(
            [-1 / a ** 2, -1 / a ** 2, 1 / b ** 2, 0., 0., 0., 0., 0., -2 * z0 / b ** 2, z0 ** 2 / b ** 2 - 1])
        oe2 = Optical_element.initialize_as_surface_conic_from_coefficients(ccc)
        oe2.set_parameters(p=0., q=z0+c, theta=90*np.pi/180, alpha=0., type="My hyperbolic mirror")
        #oe2.type = "My hyperbolic mirror"
        #oe2.p = 0.
        #oe2.q = z0 + c
        #oe2.theta = 90 * np.pi / 180
        #oe2.alpha = 0.


        return CompoundOpticalElement(oe_list=[oe1,oe2],oe_name="Wolter 1")


    @classmethod
    def initialiaze_as_wolter_2(cls,p1,q1,z0):
        #q1 = - q1
        focal = q1+2*z0
        print("focal=%f" %(focal))
        theta1 = 0.
        alpha1 = 0.

        oe1 = Optical_element.initialize_as_surface_conic_paraboloid_from_focal_distances(p1,0.,theta1,alpha1,"p", focal)
        oe2 = Optical_element.initialize_my_hyperboloid(p=0. ,q=-(focal-2*z0), theta=90*np.pi/180, alpha=0, wolter=2, z0=z0, distance_of_focalization=focal)


        return CompoundOpticalElement(oe_list=[oe1,oe2],oe_name="Wolter 2")



    @classmethod
    def initialiaze_as_wolter_12(cls,p1,q1,focal_parabola,Rmin):


        focal = focal_parabola
        d = q1 - focal_parabola
        z0 = focal_parabola + d/2
        print("focal=%f" %(focal))
        theta1 = 0.
        alpha1 = 0.


        oe1 = Optical_element.initialize_as_surface_conic_paraboloid_from_focal_distances(p1,0.,theta1,alpha1,"p", focal)
        ccc = oe1.ccc_object.get_coefficients()
        cp = -ccc[8]
        print("R=%f, d=%f, cp=%f, z0=%f" %(Rmin,d,cp,z0))
        #b1 = np.sqrt(0.125*d**2+Rmin+2*Rmin**4/cp**2-2*Rmin**2*z0/cp+0.5*z0**2-0.125/cp**2*np.sqrt((-cp**2*d**2-8*cp**2*Rmin-16*Rmin**4+16*cp*Rmin**2*z0-4*cp*z0**2)**2-16*cp**2*(4*d**2*Rmin**4-4*cp*d**2*Rmin**2*z0+cp**2*d**2*z0**2)))
        #b2 = np.sqrt(0.125*d**2+Rmin+2*Rmin**4/cp**2-2*Rmin**2*z0/cp+0.5*z0**2+0.125/cp**2*np.sqrt((-cp**2*d**2-8*cp**2*Rmin-16*Rmin**4+16*cp*Rmin**2*z0-4*cp*z0**2)**2-16*cp**2*(4*d**2*Rmin**4-4*cp*d**2*Rmin**2*z0+cp**2*d**2*z0**2)))
        p1 = -cp ** 2 * d ** 2 - 8 * cp ** 2 * Rmin - 16 * Rmin ** 4 + 16 * cp * Rmin ** 2 * z0 - 4 * cp ** 2 * z0 ** 2
        p1 = p1**2
        p2 = 16 * cp ** 2 * (4 * d ** 2 * Rmin ** 4 - 4 * cp * d ** 2 * Rmin ** 2 * z0 + cp ** 2 * d ** 2 * z0 ** 2)
        sp = 0.125/cp**2*np.sqrt(p1-p2)
        sp0 = 0.125*d**2+Rmin+2*Rmin**4/cp**2-2*Rmin**2*z0/cp+0.5*z0**2
        b = np.sqrt(sp0-sp)
        a = np.sqrt(d**2/4-b**2)

        print("a=%f, b=%f" %(a,b))


        #oe2 = Optical_element.initialize_my_hyperboloid(p=0. ,q=-(focal-2*z0), theta=90*np.pi/180, alpha=0, wolter=1.1, z0=z0, distance_of_focalization=focal)

        cc = np.array([-1/a**2, -1/a**2, 1/b**2, 0., 0., 0., 0., 0., -2*z0/b**2, (z0/b)**2-1])
        oe2 = Optical_element.initialize_as_surface_conic_from_coefficients(cc)
        oe2.type = "My hyperbolic mirror"
        oe2.set_parameters(p=0., q=q1, theta=90.*np.pi/180, alpha=0.)

        return CompoundOpticalElement(oe_list=[oe1,oe2],oe_name="Wolter 1.2")


    @classmethod
    def initialize_as_wolter_3(cls, p, q, distance_between_the_foci):
        f=-q

        oe1 = Optical_element.initialize_as_surface_conic_paraboloid_from_focal_distances(p=p, q=f, theta=0., alpha=0., infinity_location="p")

        #c = z0+np.abs(f)
        c = distance_between_the_foci/2
        z0 = np.abs(c)-np.abs(f)
        b = c+100
        a = np.sqrt((b**2-c**2))
        ccc = np.array([1/a**2, 1/a**2, 1/b**2, 0., 0., 0., 0., 0., -2*z0/b**2, z0**2/b**2-1])

        oe2 = Optical_element.initialize_as_surface_conic_from_coefficients(ccc)
        oe2.set_parameters(p=0., q=z0+z0+np.abs(q), theta=90*np.pi/180)


        return CompoundOpticalElement(oe_list=[oe1,oe2],oe_name="Wolter 3")

    @classmethod
    def initialize_as_kirkpatrick_baez(cls, p, q, separation, theta, bound1, bound2):


        p1 = p - 0.5 * separation
        q1 = p - p1
        q2 = q - 0.5 * separation
        p2 = q - q2
        f1p = p1
        f1q = p+ q - p1
        f2q = q2
        f2p = p + q - q2


        oe1 = Optical_element.initialize_as_surface_conic_ellipsoid_from_focal_distances(p= f1p, q= f1q, theta= theta, alpha=0., cylindrical=1)
        #oe1.bound = bound1
        oe1.set_bound(bound1)
        oe1.p = p1
        oe1.q = q1

        oe2 = Optical_element.initialize_as_surface_conic_ellipsoid_from_focal_distances(p= f2p, q= f2q, theta= theta, alpha=90.*np.pi/180, cylindrical=1)
        #oe2.bound = bound2
        oe2.set_bound(bound2)
        oe2.p = p2
        oe2.q = q2

        return CompoundOpticalElement(oe_list=[oe1,oe2],oe_name="Kirkpatrick Baez")

    def compound_specification_after_oe(self, oe):
        if self.type == "Wolter 1":
            if oe.type == "Surface conical mirror":
                #oe.q = 0.
                #oe.theta = 90.*np.pi/180
                oe.set_parameters(p=None, q=0., theta=90.*np.pi/180)
            elif oe.type == "My hyperbolic mirror":
                #oe.theta = 0.*np.pi/180
                oe.set_parameters(p=None, q=None, theta=0.)


        if self.type == "Wolter 1.2":
            if oe.type == "Surface conical mirror":
                #oe.q = 0.
                #oe.theta = 90.*np.pi/180
                oe.set_parameters(p=None, q=0., theta=90.*np.pi/180)
            elif oe.type == "My hyperbolic mirror":
                #oe.theta = 0.*np.pi/180
                oe.set_parameters(p=None, q=None, theta=0.)



        if self.type == "Wolter 2":
            if oe.type == "Surface conical mirror":
                #oe.q = 0.
                #oe.theta = 90.*np.pi/180
                oe.set_parameters(p=None, q=0., theta=90.*np.pi/180)
            elif oe.type == "My hyperbolic mirror":
                #oe.theta = 0.*np.pi/180
                oe.set_parameters(p=None, q=None, theta=0.)

        if self.type == "Wolter 3":
            if np.abs(oe.theta) < 1e-10:
                #oe.q = 0.
                #oe.theta = 90.*np.pi/180
                oe.set_parameters(p=None, q=0., theta=90.*np.pi/180)
            else:
                #oe.theta = 0.*np.pi/180
                oe.set_parameters(p=None, q=None, theta=0.)



    def compound_specification_after_screen(self, oe, beam):
        if self.type == "Wolter 1":
            if oe.type == "My hyperbolic mirror":
                oe.output_frame_wolter(beam)


        if self.type == "Wolter 2":
            if oe.type == "My hyperbolic mirror":
                oe.output_frame_wolter(beam)

        if self.type == "Wolter 3":
            if oe.theta < 1e-10:
                oe.output_frame_wolter(beam)
                #todo control well this part
                x = beam.x
                z = beam.z
                vx = beam.vx
                vz = beam.vz

                beam.x = z
                beam.z = x
                beam.vx = vz
                beam.vz = vx


    def trace_compound(self,beam1):

        beam=beam1.duplicate()

        for i in range (self.oe_number()):

            print("Iteration number %d"  %(i+1))
            self.oe[i].effect_of_optical_element(beam)
            self.compound_specification_after_oe(oe = self.oe[i])
            self.oe[i].effect_of_the_screen(beam)
            self.compound_specification_after_screen(oe = self.oe[i], beam = beam)

        return beam


    def info(self):

        txt = ("\nThe optical element of the %s system are:\n" %(self.type))

        for i in range (self.oe_number()):
            txt += ("\nThe %d optical element:\n\n" %(i+1))
            txt += self.oe[i].info()
        return txt



    def trace_with_hole(self,beam1):

        beam11=beam1.duplicate()
        beam = beam1.duplicate()

        self.oe[0].rotation_to_the_optical_element(beam11)
        self.oe[0].translation_to_the_optical_element(beam11)

        b1=beam11.duplicate()
        b2=beam11.duplicate()
        [b1, t1] = self.oe[0].intersection_with_optical_element(b1)
        [b2, t2] = self.oe[1].intersection_with_optical_element(b2)

        beam.flag = beam.flag + 1
        if self.type == "Wolter 1":
            indices = np.where (t1>=t2)
        elif self.type == "Wolter 2":
            indices = np.where (t1<=t2)

        beam.flag[indices] = -1*beam.flag[indices]
        print(beam.flag)

        #beam.plot_good_xz(0)

        beam = beam.good_beam()

        beam.plot_good_xz()
        plt.title("Good initial rays")


        l = beam.number_of_good_rays()
        print(l)

        if l >0:
            beam = self.trace_compound(beam)
        else:
            print(">>>>>>NO GOOD RAYS")


        #print("t1 is (that one of the parabolic mirror)")
        ##print(t1)
        #print(np.mean(t1))
        #print("t2 is (that one of the hyperbolic mirror)")
        ##print(t2)
        #print(np.mean(t2))


        print("Number of good rays=%f" %(beam.number_of_good_rays()))

        return beam
