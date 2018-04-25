from Beam import Beam
from OpticalElement import Optical_element
import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_almost_equal

def test_plane_mirror():

    beam1=Beam(5000)
    beam1.set_point(0,0,0)
    beam1.set_flat_divergence(5e-3,5e-2)


    p=1.
    q=1.
    theta=45
    alpha=0
    plane_mirror=Optical_element.initialize_as_plane_mirror(p,q,theta,alpha)

    beam1=plane_mirror.trace_optical_element(beam1)
    beam1.plot_xz()

    beam1.plot_xpzp()
    plt.show()



def test_spherical_mirror():

    beam1=Beam(5000)
    beam1.set_point(0,0,0)
    beam1.set_flat_divergence(5e-3,5e-2)

    p=2.
    q=1.
    theta=30
    alpha=0

    spherical_mirror=Optical_element.initialize_as_spherical_mirror(p,q,theta,alpha,0.0)
    spherical_mirror.set_spherical_mirror_radius_from_focal_distances()



    beam1=spherical_mirror.trace_optical_element(beam1)
    beam1.plot_xz()

    beam1.plot_xpzp()
    plt.show()

    assert_almost_equal(np.abs(beam1.x).mean(),0.0,2)
    assert_almost_equal(np.abs(beam1.y).mean(),0.0,2)
    assert_almost_equal(np.abs(beam1.z).mean(),0.0,2)
