from Beam import Beam
from OpticalElement import Optical_element
from Shape import BoundaryRectangle
import numpy as np
import matplotlib.pyplot as plt
from numpy.testing import assert_almost_equal

do_plot = False

def test_my_hyperbolic_mirror():

    beam=Beam()
    beam.set_flat_divergence(0.005,0.0005)
    p1=130.
    q1=0.
    spherical_mirror=Optical_element.initialize_as_spherical_mirror(p1,q1,theta=0,alpha=0,R=130.)
    beam=spherical_mirror.trace_optical_element(beam)

    p=15
    q=p1-p
    theta=0*np.pi/180

    hyp_mirror=Optical_element.initialize_my_hyperboloid(p,q,theta)
    beam=hyp_mirror.trace_optical_element(beam)
    beam.plot_xz()

    assert_almost_equal(beam.x, 0., 10)
    assert_almost_equal(beam.y, 0., 10)
    assert_almost_equal(beam.z, 0., 10)

    if do_plot:
        plt.show()
