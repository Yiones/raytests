from Beam import Beam
from OpticalElement import Optical_element
from SurfaceConic import SurfaceConic
from Shape import BoundaryRectangle
import numpy as np
from SurfaceConic import SurfaceConic
import matplotlib.pyplot as plt



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

        self.oe = (self.oe1, self.oe2, self.oe3, self.oe4, self.oe5,
                       self.oe6, self.oe7, self.oe8, self.oe9)

        self.type = None



    @classmethod
    def initialiaze_as_wolter_1(cls,p1,q1,z0):
        theta1 = 0.
        alpha1 = 0.

        wolter1 = CompoundOpticalElement(2)
        wolter1.oe1 = Optical_element.initialize_as_surface_conic_paraboloid_from_focal_distances(p1,0.,theta1,alpha1,"p",2*z0-q1)
        print("The focal is %f    "    %(wolter1.oe1.focal))
        wolter1.oe2 = Optical_element.initialize_my_hyperboloid(p=0.,q=q1,theta=90*np.pi/180,alpha=0,wolter=1, z0=z0, distance_of_focalization=2*z0-q1)

        wolter1.type = "Wolter 1"

        return wolter1


    @classmethod
    def initialiaze_as_wolter_2(cls,p1,q1,z0):
        q1 = - q1
        focal = q1
        theta1 = 0.
        alpha1 = 0.

        wolter2 = CompoundOpticalElement(2)
        wolter2.oe1 = Optical_element.initialize_as_surface_conic_paraboloid_from_focal_distances(p1,0.,theta1,alpha1,"p", focal)
        wolter2.oe2 = Optical_element.initialize_my_hyperboloid(p=0. ,q=-(focal-2*z0), theta=90*np.pi/180, alpha=0, wolter=2, z0=z0, distance_of_focalization=focal)

        wolter2.type = "Wolter 2"

        return wolter2


    @classmethod
    def initialize_as_wolter_3(cls, p, q, distance_between_the_foci):
        f=-q

        wolter3 = CompoundOpticalElement(2)
        wolter3.oe1 = Optical_element.initialize_as_surface_conic_paraboloid_from_focal_distances(p=p, q=f, theta=0., alpha=0., infinity_location="p")

        #c = z0+np.abs(f)
        c = distance_between_the_foci/2
        z0 = np.abs(c)-np.abs(f)
        b = c+100
        a = np.sqrt((b**2-c**2))
        ccc = np.array([1/a**2, 1/a**2, 1/b**2, 0., 0., 0., 0., 0., -2*z0/b**2, z0**2/b**2-1])

        wolter3.oe2 = Optical_element.initialize_as_surface_conic_from_coefficients(ccc)
        wolter3.oe2.set_parameters(p=0., q=z0+z0+np.abs(q), theta=90*np.pi/180)

        wolter3.type = "Wolter 3"

        return wolter3

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

        print("\np1=%f, q1=%f, p2=%f, q2=%f, f1p=%f, f1q=%f, f2p=%f, f2q=%f" %(p1, q1, p2, q2, f1p, f1q, f2p, f2q))

        kirk_baez = CompoundOpticalElement(2)
        kirk_baez.oe1 = Optical_element.initialize_as_surface_conic_ellipsoid_from_focal_distances(p= f1p, q= f1q, theta= theta, alpha=0., cylindrical=1)
        kirk_baez.oe1.bound = bound1
        kirk_baez.oe1.p = p1
        kirk_baez.oe1.q = q1

        kirk_baez.oe2 = Optical_element.initialize_as_surface_conic_ellipsoid_from_focal_distances(p= f2p, q= f2q, theta= theta, alpha=90.*np.pi/180, cylindrical=1)
        kirk_baez.oe2.bound = bound2
        kirk_baez.oe2.p = p2
        kirk_baez.oe2.q = q2

        return kirk_baez

    def compound_specification_after_oe(self, oe):
        if self.type == "Wolter 1":
            if oe.type == "Surface conical mirror":
                oe.q = 0.
                oe.theta = 90.*np.pi/180
            elif oe.type == "My hyperbolic mirror":
                oe.theta = 0.*np.pi/180


        if self.type == "Wolter 2":
            if oe.type == "Surface conical mirror":
                oe.q = 0.
                oe.theta = 90.*np.pi/180
            elif oe.type == "My hyperbolic mirror":
                oe.theta = 0.*np.pi/180

        if self.type == "Wolter 3":
            if np.abs(oe.theta) < 1e-10:
                oe.q = 0
                oe.theta = 90.*np.pi/180
            else:
                oe.theta = 0.



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


    def trace_compound(self,beam1):

        oe = (self.oe1, self.oe2, self.oe3, self.oe4, self.oe5,
                       self.oe6, self.oe7, self.oe8, self.oe9)

        beam=beam1.duplicate()

        for i in range (self.k):

            oe[i].effect_of_optical_element(beam)
            self.compound_specification_after_oe(oe = oe[i])
            oe[i].effect_of_the_screen(beam)
            print("Last value of theta = %f" %(oe[i].theta))
            self.compound_specification_after_screen(oe = oe[i], beam = beam)

            print("Good rays after %d optical element = %f" % (i+1, np.size(np.where(beam.flag >= 0))))
        return beam


    #def trace_kirk_patrick_baez(self,beam1):

    #    oe = (self.oe1, self.oe2, self.oe3, self.oe4, self.oe5,
    #                   self.oe6, self.oe7, self.oe8, self.oe9)

    #    beam=beam1.duplicate()



    #    self.oe1.effect_of_optical_element(beam)
    #    self.oe1.effect_of_the_screen(beam)


    #    print("Good rays after first mirror = %f" % (np.size(np.where(beam.flag >= 0))))

    #    self.oe2.effect_of_optical_element(beam)
    #    self.oe2.effect_of_the_screen(beam)

    #    print("Good rays after second mirror = %f" % (np.size(np.where(beam.flag >= 0))))


    #    return beam


#    def trace_wolter3(self, beam1, z0):
#
#        beam = beam1.duplicate()
#
#        self.oe1.effect_of_optical_element(beam)
#
#        #self.oe2.p = 0
#        #self.oe2.theta = 90*np.pi/180
#
#        self.oe2.effect_of_optical_element(beam)
#
#        #self.oe2.q = z0+z0+np.abs(self.oe1.q)
#        self.oe2.theta = 0.
#
#        self.oe2.effect_of_the_screen(beam)
#        return beam
#


