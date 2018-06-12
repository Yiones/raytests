from Beam import Beam
from OpticalElement import Optical_element
from Shape import BoundaryRectangle
import numpy as np
from SurfaceConic import SurfaceConic
import matplotlib.pyplot as plt
from CompoundOpticalElement import CompoundOpticalElement


#def test_compound_wolter2():
#
#    p=26.
#    beam1 = Beam.initialize_as_person()
#    beam1.set_point(p, 0., p)
#    #beam1.set_rectangular_spot(5 / 2 * 1e-5, -5 / 2 * 1e-5, 5 / 2 * 1e-5, -5 / 2 * 1e-5)
#
#    op_ax = Beam (1)
#    op_ax.set_point(p, 0., p)
#
#    beam = op_ax.merge(beam1)
#    beam.set_divergences_collimated()
#    beam.plot_xz()
#
#
#    p = 20.
#    q = 30.
#    z0 = 10.
#
#    wolter2 = CompoundOpticalElement.initialiaze_as_wolter_2(p, q, z0)
#
#    beam = wolter2.trace_compound(beam)
#    beam.plot_xz()
#    beam.retrace(10.)
#
#    beam.plot_xz()
#    plt.show()
#

#def test_compound_wolter2_with_hole():
#
#    p=50.
#    #beam1 = Beam.initialize_as_person(10000)
#    beam1 = Beam(100000)
#    beam1.set_circular_spot(1.)
#    #beam1.set_rectangular_spot(5 / 2 * 1e-5, -5 / 2 * 1e-5, 5 / 2 * 1e-5, -5 / 2 * 1e-5)
#    beam1.x *= 100.
#    beam1.z *= 100.
#    beam1.set_point(p, 0., p)
#
#
#    op_ax = Beam (1)
#    op_ax.set_point(p, 0., p)
#
#    beam = op_ax.merge(beam1)
#    beam.set_divergences_collimated()
#    beam.plot_xz(0)
#
#
#    p = 20000.
#    q = 30.
#    z0 = 10.
#
#    wolter2 = CompoundOpticalElement.initialiaze_as_wolter_2(p, q, z0)
#
#    beam = wolter2.trace_with_hole(beam)
#    #beam.plot_xz(0)
#    #beam.plot_xpzp()
#    #beam = wolter2.trace_compound(beam)
#
#    beam.plot_xz()
#    beam.retrace(10.)
#    beam.plot_good_xz()
#    plt.title("Good final rays")
#
#    print(wolter2.info())
#
#    beam.plot_xz()
#
#    plt.show()



def test_compound_wolter2_with_hole():

    p=0.
    #beam1 = Beam.initialize_as_person(10000)
    beam1 = Beam(100000)
    beam1.set_circular_spot(1.)
    #beam1.set_rectangular_spot(5 / 2 * 1e-5, -5 / 2 * 1e-5, 5 / 2 * 1e-5, -5 / 2 * 1e-5)
    beam1.x *= 10.
    beam1.z *= 10.
    beam1.set_point(p, 0., p)


    op_ax = Beam (1)
    op_ax.set_point(p, 0., p)

    beam = op_ax.merge(beam1)
    beam.set_divergences_collimated()
    beam.plot_xz(0)


    p = 20000.
    q = 30.
    z0 = 5.
    focal = 2*z0+q

    wolter2 = CompoundOpticalElement.initialiaze_as_wolter_2(p1=p, q1=q, z0=z0)

   #oe1 = Optical_element.initialize_as_surface_conic_paraboloid_from_focal_distances(p=p, q=0., theta=0., alpha=0., infinity_location="p", focal=focal)
   #oe2 = Optical_element.initialize_my_hyperboloid(p=0., q=-q, theta=90*np.pi/180, alpha=0., wolter=2, z0=z0, distance_of_focalization=focal)

    #oe1.rotation_to_the_optical_element(beam)
    #oe1.translation_to_the_optical_element(beam)
    #[beam, t] = oe1.intersection_with_optical_element(beam)
    #oe1.output_direction_from_optical_element(beam)

    #[beam, t] = oe2.intersection_with_optical_element(beam)
    #oe2. output_direction_from_optical_element(beam)

    #oe2.theta = 0.
    #oe2.rotation_to_the_screen(beam)
    #oe2.translation_to_the_screen(beam)
    #oe2.intersection_with_the_screen(beam)

    beam = wolter2.trace_compound(beam)

    beam.plot_xz()
    print("mean(beam.x)=%f, mean(beam.y)=%f, mean(beam.z)=%f" %(np.mean(beam.x),np.mean(beam.y),np.mean(beam.z)))

    beam.retrace(10.)
    beam.plot_xz()

    plt.show()
