from Beam import Beam
from OpticalElement import Optical_element
from Shape import BoundaryRectangle
import numpy as np
import matplotlib.pyplot as plt
from numpy.testing import assert_almost_equal
from CompoundOpticalElement import CompoundOpticalElement
from Vector import Vector
from SurfaceConic import SurfaceConic

#def test_wolter1():
#
#    beam1 = Beam()
#    beam1.set_point(15., 0., 15.)
#    op_ax = Beam (1)
#    op_ax.set_point(15., 0., 15.)
#    # beam1.set_rectangular_spot(55/2*1,-55/2*1,55/2*1,-55/2*1)
#    beam1.set_circular_spot(25*1e-3)
#
#    beam=op_ax.merge(beam1)
#    beam.set_divergences_collimated()
#
#    #beam.plot_xz()
#
#
#    p = 36.
#    q = 25.
#    focal = 25.
#    z0 = 15
#    theta = 0 * np.pi / 180
#    alpha = 0 * np.pi / 180
#
#    paraboloid = Optical_element.initialize_as_surface_conic_paraboloid_from_focal_distances(p, 0, theta, alpha, "p", focal)
#
#    beam = paraboloid.trace_Wolter_1(beam, z0)
#
#    beam.retrace(10.)
#
#
#    beam.plot_xz()
#    plt.show()



#def test_compound_wolter1():
#
#    p=26.
#    beam1 = Beam()
#    beam1.set_point(p, 0., p)
#
#    op_ax = Beam (1)
#    op_ax.set_point(p, 0., p)
#    beam1.set_circular_spot(25*1e-3)
#
#    beam=op_ax.merge(beam1)
#    beam.set_divergences_collimated()
#
#    beam.plot_xz()
#
#
#    p = 36.
#    q = 25.
#    z0 = 0.
#
#    wolter1 = CompoundOpticalElement.initialiaze_as_wolter_1(p, q, z0)
#
#    beam = wolter1.trace_compound(beam)
#    beam.plot_xz()
#    beam.retrace(10.)
#    beam.plot_xz()
#    plt.show()


def test_compound_wolter1():

    p=26.
    beam1 = Beam.initialize_as_person()
    beam1.set_point(p, 0., p)

    op_ax = Beam (1)
    op_ax.set_point(p, 0., p)

    beam=op_ax.merge(beam1)
    beam.set_divergences_collimated()

    beam.plot_xz()


    p = 36.
    q = 25.
    z0 = 0.

    wolter1 = CompoundOpticalElement.initialiaze_as_wolter_1(p, q, z0)

    beam = wolter1.trace_compound(beam)
    beam.plot_xz()
    beam.retrace(10.)
    beam.plot_xz()
    plt.show()

# ah = np.sqrt(-1/ccc[0])
# bh = np.sqrt(1/ccc[2])

# z0 = -ccc[8]*bh**2/2

# print("ah = %f, bh = %f, z0 = %f"  %(ah,bh,z0))

# a = -ah*beam.vx**2 - ah*beam.vy**2 + bh*beam.vz**2
# b = -ah*beam.x*beam.vx - ah*beam.y*beam.vy + bh*beam.z*beam.vz - bh*z0*beam.vz
# c = -ah*beam.x**2 - ah*beam.y**2 + bh*beam.z**2 -2*bh*z0*beam.z + b*z0**2 -1



