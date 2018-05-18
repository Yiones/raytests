from Beam import Beam
from OpticalElement import Optical_element
from Shape import BoundaryRectangle
import numpy as np
from SurfaceConic import SurfaceConic
import matplotlib.pyplot as plt



def test_wolter2():
    beam = Beam()
    beam.set_divergences_collimated()
    beam.set_point(15., 0., 15.)
    beam.set_rectangular_spot(5 / 2 * 1e-1, -5 / 2 * 1e-6, 5 / 2 * 1e-1, -5 / 2 * 1e-1)

    p = 2000.
    q = 25.
    theta = 0 * np.pi / 180
    alpha = 0 * np.pi / 180

    prova = Optical_element.initialize_as_surface_conic_paraboloid_from_focal_distances(p, q, theta, alpha, "p")
    print(prova.ccc_object.get_coefficients())

    beam = prova.trace_Wolter_2(beam)

    print(np.mean(beam.z))
    beam.plot_xy()
    beam.plot_xz()
    plt.show()
