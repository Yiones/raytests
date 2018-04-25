from Beam import Beam
from OpticalElement import Optical_element
import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_almost_equal

def test_plane_mirror():

    beam1=Beam()
    beam1.set_point(0,0,0)
    #beam1.set_gaussian_divergence(0.001,0.0001)
    beam1.set_flat_divergence(5e-3,5e-2)



    p=1.
    q=1.
    theta=45
    alpha=0
    plane_mirror=Optical_element.initialize_as_plane_mirror(p,q,theta,alpha)

    beam1=plane_mirror.reflection(beam1)
    beam1.plot_xz()

    beam1.plot_xpzp()
    plt.show()



def test_spherical_mirror():

    beam1=Beam()
    beam1.set_point(0,0,0)
    #beam1.set_gaussian_divergence(0.001,0.0001)
    beam1.set_flat_divergence(5e-3,5e-2)



    p=1.
    q=1.
    theta=0
    alpha=0
    R=1
    spherical_mirror=Optical_element.initialize_as_spherical_mirror(p,q,theta,alpha,R)

    beam1=spherical_mirror.reflection(beam1)
    beam1.plot_xz()

    beam1.plot_xpzp()
    plt.show()


    assert_almost_equal(beam1.x[(np.random.randint(beam1.N))],0.0,15)
    assert_almost_equal(beam1.y[(np.random.randint(beam1.N))],0.0,15)
    assert_almost_equal(beam1.z[(np.random.randint(beam1.N))],0.0,15)


