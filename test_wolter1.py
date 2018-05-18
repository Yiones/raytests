from Beam import Beam
from OpticalElement import Optical_element
from Shape import BoundaryRectangle
import numpy as np
import matplotlib.pyplot as plt
from numpy.testing import assert_almost_equal


def test_wolter1():
    beam = Beam()
    beam.set_divergences_collimated()
    beam.set_point(35., 0., 35.)
    # beam.set_rectangular_spot(55/2*1,-55/2*1,55/2*1,-55/2*1)
    beam.set_circular_spot(25)

    beam.plot_xz()

    p = 25.
    q = 25.
    theta = 0 * np.pi / 180
    alpha = 0 * np.pi / 180

    prova = Optical_element.initialize_as_surface_conic_paraboloid_from_focal_distances(p, q, theta, alpha, "p")

    beam = prova.trace_Wolter_1(beam)

    beam.plot_xy()
    beam.plot_xz()
    plt.show()





















