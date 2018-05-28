from Beam import Beam
from OpticalElement import Optical_element
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
    def initialiaze_as_wolter_1(cls,p1,q1,theta1):
        alpha1=0.

        wolter1 = CompoundOpticalElement(2)
        wolter1.oe1 = Optical_element.initialize_as_surface_conic_paraboloid_from_focal_distances(p1,q1,theta1,0,"p")
        wolter1.oe2 = Optical_element.initialize_my_hyperboloid(p=q1,q=np.sqrt(2),theta=90*np.pi/180,alpha=0,wolter=0.)

        wolter1.type="Wolter 1"

        return wolter1


    @classmethod
    def initialiaze_as_wolter_2(cls,p1,q1,theta1):
        alpha1=0.

        wolter2 = CompoundOpticalElement(2)
        wolter2.oe1 = Optical_element.initialize_as_surface_conic_paraboloid_from_focal_distances(p1,q1,theta1,alpha1,"p")
        wolter2.oe2 = Optical_element.initialize_my_hyperboloid(p=q1-np.sqrt(2),q=np.sqrt(2),theta=90*np.pi/180,alpha=0,wolter=2)

        wolter2.type="Wolter 2"

        return wolter2



    def trace_wolter1(self,beam1):

        beam=beam1.duplicate()

        self.oe1.rotation_to_the_optical_element(beam)
        self.oe1.translation_to_the_optical_element(beam)
        self.oe1.intersection_with_optical_element(beam)      #####intersection with the paraboloid
        self.oe1.output_direction_from_optical_element(beam)



        self.oe2.rotation_to_the_optical_element(beam)
        self.oe2.intersection_with_optical_element(beam)
        self.oe2.output_direction_from_optical_element(beam)



        self.oe1.q=self.oe1.q+2*np.sqrt(2)
        self.oe1.rotation_to_the_screen(beam)
        self.oe1.translation_to_the_screen(beam)
        self.oe1.intersection_with_the_screen(beam)

        return beam


    def trace_wolter2(self,beam1):

        beam=beam1.duplicate()

        self.oe1.rotation_to_the_optical_element(beam)
        self.oe1.translation_to_the_optical_element(beam)
        self.oe1.intersection_with_optical_element(beam)      #####intersection with the paraboloid
        self.oe1.output_direction_from_optical_element(beam)



        self.oe2.rotation_to_the_optical_element(beam)
        self.oe2.intersection_with_optical_element(beam)
        self.oe2.output_direction_from_optical_element(beam)


        self.oe1.q=25-2*np.sqrt(2)
        self.oe1.rotation_to_the_screen(beam)
        self.oe1.translation_to_the_screen(beam)
        self.oe1.intersection_with_the_screen(beam)

        return beam




    def trace_compound_optical_element(self,beam):
        if self.type == "Wolter 1":
            beam=self.trace_wolter1(beam)
        if self.type == "Wolter 2":
            beam=self.trace_wolter2(beam)

        return beam

    def trace(self,beam1):

        oe = (self.oe1, self.oe2, self.oe3, self.oe4, self.oe5,
                       self.oe6, self.oe7, self.oe8, self.oe9)

        beam=beam1.duplicate()

        for i in range (self.k):
            print(oe[i].type)

            oe[i].effect_of_optical_element(beam)
            self.compound_specification(wolter_flag=1, oe = oe[i])
            oe[i].effect_of_the_screen(beam)




        #oe[0].effect_of_optical_element(beam)
        #beam.plot_yx()
        #oe[1].effect_of_optical_element(beam)
        ##oe[0].q = oe[1].z0 + (oe[1].z0 - oe[0].focal)
        #oe[0].q = 15. + (15. - 54.)
        #oe[0].effect_of_the_screen(beam)
        #print(oe[0].q)

        return beam

    def compound_specification(self, wolter_flag, oe):
        if wolter_flag == 1:
            if oe.type == "Surface conical mirror":
                oe.q = 0.
                oe.theta = 90.*np.pi/180
            elif oe.type == "Surface conical mirror 1":
                oe.theta = 0.*np.pi/180
