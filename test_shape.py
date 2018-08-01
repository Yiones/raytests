from Beam import Beam
import unittest
from OpticalElement import Optical_element
from Shape import BoundaryRectangle
from Shape import BoundaryCircle
import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_almost_equal

do_plot = False


class ShapeTest(unittest.TestCase):

    def test_rectangular_shape(self):


        beam = Beam(round(1e5))
        plane_mirror = Optical_element.initialize_as_surface_conic_plane(p=10., q=0., theta=0.)

        beam.set_flat_divergence(0.02, 0.1)

        xmax = 0.01
        xmin = -0.0008
        ymax = 1.
        ymin = -0.29



        bound = BoundaryRectangle(xmax=xmax,xmin=xmin,ymax=ymax,ymin=ymin)
        plane_mirror.set_bound(bound)

        beam = plane_mirror.trace_optical_element(beam)

        beam.plot_xz()
        beam.plot_good_xz()

        indices = np.where(beam.flag>0)

        assert_almost_equal(max(beam.x[indices])-xmax , 0., 2)
        assert_almost_equal(-min(beam.x[indices])+xmin, 0., 2)
        assert_almost_equal(max(beam.z[indices])+ymin , 0., 2)
        assert_almost_equal(-min(beam.z[indices])-ymax, 0., 2)


        print(max(beam.x[indices]),min(beam.x[indices]),max(beam.y[indices]),min(beam.y[indices]))

        if do_plot is True:
            plt.show()



#########  BoundaryCircle has to be implemented in the code of intersection_with_optical_element    ####################