from Beam import Beam
from OpticalElement import Optical_element
import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_almost_equal



def test_spherical_mirror():

    beam1=Beam(5000)
    beam1.set_point(0,0,0)
    beam1.set_flat_divergence(5e-3,5e-2)

    p=2.
    q=1.
    theta=80*np.pi/180

    spherical_mirror=Optical_element.initialize_as_sphere_from_focal_distances(p,q,theta)

    beam1=spherical_mirror.trace_surface_conic(beam1)

    beam1.plot_xz()

    beam1.plot_xpzp()
    plt.show()



def test_ellipsoidal_mirror():

    beam1=Beam(5000)
    beam1.set_point(0,0,0)
    beam1.set_flat_divergence(5e-3,5e-2)

    p=20.
    q=10.
    theta=50*np.pi/180

    spherical_mirror=Optical_element.initialize_as_ellipsoid_from_focal_distances(p,q,theta)

    beam1=spherical_mirror.trace_surface_conic(beam1)

    beam1.plot_xz()

    beam1.plot_xpzp()
    plt.show()


def test_paraboloid_mirror():
    beam1=Beam(5000)
    beam1.set_point(0,0,0)
    beam1.set_flat_divergence(5e-3,5e-2)
    p=10.
    q=20.
    theta=88*np.pi/180
    spherical_mirror=Optical_element.initialize_as_paraboloid_from_focal_distances(p,q,theta)
    beam1=spherical_mirror.trace_surface_conic(beam1)
    beam1.plot_xz()
    beam1.plot_xpzp()
    plt.show()




def test_hyperboloid_mirror():
    beam1=Beam(5000)
    beam1.set_point(0,0,0)
    beam1.set_flat_divergence(5e-3,5e-2)
    p=10.
    q=20.
    theta=28*np.pi/180
    spherical_mirror=Optical_element.initialize_as_hyperboloid_from_focal_distances(p,q,theta)
    beam1=spherical_mirror.trace_surface_conic(beam1)
    beam1.plot_xz()
    beam1.plot_xpzp()
    plt.show()