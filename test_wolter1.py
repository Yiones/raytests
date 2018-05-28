from Beam import Beam
from OpticalElement import Optical_element
from Shape import BoundaryRectangle
import numpy as np
import matplotlib.pyplot as plt
from numpy.testing import assert_almost_equal
from CompoundOpticalElement import CompoundOpticalElement
from Vector import Vector
from SurfaceConic import SurfaceConic

def test_wolter1():

    beam1 = Beam()
    beam1.set_point(15., 0., 15.)
    op_ax = Beam (1)
    op_ax.set_point(15., 0., 15.)
    # beam1.set_rectangular_spot(55/2*1,-55/2*1,55/2*1,-55/2*1)
    beam1.set_circular_spot(25*1e-3)

    beam=op_ax.merge(beam1)
    beam.set_divergences_collimated()

    #beam.plot_xz()


    p = 36.
    q = 25.
    focal = 54.
    z0 = 15
    theta = 0 * np.pi / 180
    alpha = 0 * np.pi / 180

    paraboloid = Optical_element.initialize_as_surface_conic_paraboloid_from_focal_distances(p, 0, theta, alpha, "p", focal)

    beam = paraboloid.trace_Wolter_1(beam, z0)

    beam.retrace(0)


    beam.plot_xz()
    plt.show()


def test_wolter1_new():

    beam1 = Beam()
    beam1.set_point(15., 0., 15.)
    op_ax = Beam (1)
    op_ax.set_point(15., 0., 15.)
    # beam1.set_rectangular_spot(55/2*1,-55/2*1,55/2*1,-55/2*1)
    beam1.set_circular_spot(25*1e-3)

    beam=op_ax.merge(beam1)
    beam.set_divergences_collimated()

    beam.plot_xz()


    p = 36.
    q = 25.
    focal = 54.
    z0 = 15
    theta = 0 * np.pi / 180
    alpha = 0 * np.pi / 180

    paraboloid = Optical_element.initialize_as_surface_conic_paraboloid_from_focal_distances(p, 0, theta, alpha, "p", focal)

    hyp = Optical_element.initialize_my_hyperboloid(p=0., q=15. + (15. - 54.), theta=90*np.pi/180, alpha=0, wolter=1, z0=z0, distance_of_focalization=paraboloid.focal)

    compound_wolter1 = CompoundOpticalElement(2)
    compound_wolter1.oe1 = paraboloid
    compound_wolter1.oe2 = hyp

    beam = compound_wolter1.trace(beam)



    beam.retrace(0)


    beam.plot_xz()
    plt.show()


