from Beam import Beam
from OpticalElement import Optical_element
from Shape import BoundaryRectangle
import numpy as np
from SurfaceConic import SurfaceConic
import matplotlib.pyplot as plt
from CompoundOpticalElement import CompoundOpticalElement
from Vector import Vector



#def test_wolter3():
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
#    #beam.plot_xz()
#
#
#
#    #paraboloid = Optical_element.initialize_as_surface_conic_paraboloid_from_focal_distances(p=10., q=-5., theta=0., alpha=0., infinity_location="p")
#    #print("Paraboloid coefficients")
#    #print(paraboloid.ccc_object.get_coefficients())
#
#    #f=5.
#    #z0=5.
#    #c=z0+f
#    #b=c+100
#    #a=np.sqrt((b**2-c**2))
#
#
#    #ccc = np.array([1/a**2, 1/a**2, 1/b**2, 0., 0., 0., 0., 0., -2*z0/b**2, z0**2/b**2-1])
#    #ellips = Optical_element.initialize_as_surface_conic_from_coefficients(ccc)
#    #print("ellips coefficients")
#    #print(ellips.ccc_object.get_coefficients())
#
#    #paraboloid.effect_of_optical_element(beam)
#    #ellips.intersection_with_optical_element(beam)
#    #ellips.output_direction_from_optical_element(beam)
#
#
#    wolter3 = CompoundOpticalElement.initialize_as_wolter_3(20., 5., 5.)
#
#    print(wolter3.oe1.ccc_object.get_coefficients())
#    print(wolter3.oe2.ccc_object.get_coefficients())
#
#    beam = wolter3.trace_wolter3(beam)
#
#
#    #t=-beam.x/beam.vx
#    #beam.x = beam.x + beam.vx * t
#    #beam.y = beam.y + beam.vy * t
#    #beam.z = beam.z + beam.vz * t
#
#    beam.plot_xz()
#    beam.plot_yx()
#
#
#    print(np.mean(beam.z))
#
#    plt.show()







def test_clean_wolter3():

    p=50.
    beam1 = Beam.initialize_as_person()
    beam1.set_point(p, 0., p)
    #beam1.set_rectangular_spot(5 / 2 * 1e-5, -5 / 2 * 1e-5, 5 / 2 * 1e-5, -5 / 2 * 1e-5)

    op_ax = Beam (1)
    op_ax.set_point(p, 0., p)

    beam = op_ax.merge(beam1)
    beam.set_divergences_collimated()

    beam.plot_xz()

    distance_between_the_foci = 10.

    wolter3 = CompoundOpticalElement.initialize_as_wolter_3(20., 5., distance_between_the_foci)

    print(wolter3.oe1.ccc_object.get_coefficients())
    print(wolter3.oe2.ccc_object.get_coefficients())

    #beam = wolter3.trace_wolter3(beam, z0)
    beam = wolter3.trace_compound(beam)


    beam.plot_xz()

    beam.retrace(0.1)
    beam.plot_xz()

    plt.show()

