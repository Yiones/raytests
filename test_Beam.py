from Beam import Beam
import numpy as np
from numpy.testing import assert_almost_equal
import matplotlib.pyplot as plt

def test_gaussian_beam():
    beam=Beam(5000)
    beam.set_point_source(1.,1.,1.)
    beam.set_gaussian_divergence(0.05,0.0005)

    print(np.mean(beam.vx))
    print(np.mean(beam.vz))

    assert_almost_equal(np.mean(beam.vx),0.0,1)
    assert_almost_equal(np.mean(beam.vz),0.0,1)

def test_duplicate():
    b1=Beam()
    b2=b1.duplicate()

    assert_almost_equal(b1.x, b2.x ,9)
    assert_almost_equal(b1.y, b2.y ,9)
    assert_almost_equal(b1.z, b2.z ,9)
    assert_almost_equal(b1.vx,b2.vx,9)
    assert_almost_equal(b1.vy,b2.vy,9)
    assert_almost_equal(b1.vz,b2.vz,9)


def test_my_beam():

    dx=1e-3
    dz=1e-2
    p0=1         #square spot size
    p1=1         #gaussian velocity distribution

    beam=Beam.initialize_mysource(dx,dz,p0,p1)
    beam.plot_xz()

    print(np.mean(beam.vx))
    print(np.mean(beam.vz))

    assert_almost_equal(np.mean(beam.vx),0.0,3)
    assert_almost_equal(np.mean(beam.vz),0.0,3)

    plt.show()