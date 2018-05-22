from Beam import Beam
from OpticalElement import Optical_element
import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_almost_equal
from Shape import *

do_plot = True

#def test_plane_mirror():
#    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  test_plane_mirror")
#
#    beam1=Beam(5000)
#    beam1.set_point(0,0,0)
#    beam1.set_flat_divergence(5e-3,5e-2)
#
#
#    p=1.
#    q=1.
#    theta=np.pi/4
#    alpha=0
#    plane_mirror=Optical_element.initialize_as_plane_mirror(p,q,theta,alpha)
#
#    xmin =-10**5
#    xmax = 10**5
#    ymin = 10**5
#    ymax =-10**5
#    bound=BoundaryRectangle(xmax,xmin,ymax,ymin)
#    plane_mirror.rectangular_bound(bound)
#    beam1=plane_mirror.trace_optical_element(beam1)
#    beam1.plot_xz()
#
#    beam1.plot_xpzp()
#
#    if do_plot:
#        plt.show()
#
#
#
#def test_spherical_mirror():
#    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  test_spherical_mirror")
#
#    beam1=Beam(5000)
#    beam1.set_point(0,0,0)
#    beam1.set_flat_divergence(5e-3,5e-2)
#
#    p=2.
#    q=1.
#    theta=30
#    theta=theta*np.pi/180
#    alpha=0*np.pi/180
#
#
#    spherical_mirror=Optical_element.initialize_as_spherical_mirror(p,q,theta,alpha)
#    #spherical_mirror.set_spherical_mirror_radius_from_focal_distances()
#    print(spherical_mirror.R)
#
#
#    beam1=spherical_mirror.trace_optical_element(beam1)
#    beam1.plot_xz()
#
#    beam1.plot_xpzp()
#
#    print(np.mean(beam1.flag))
#
#    if do_plot:
#        plt.show()
#
#
#    assert_almost_equal(np.abs(beam1.x).mean(),0.0,2)
#    assert_almost_equal(np.abs(beam1.y).mean(),0.0,2)
#    assert_almost_equal(np.abs(beam1.z).mean(),0.0,2)
#
#
#def test_ideal_lens_with_trace_ideal_lens():
#    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  test_ideal_lens_with_trace_ideal_lens")
#
#    beam=Beam()
#    beam.set_flat_divergence(0.05,0.005)
#
#    p=1.
#    q=5.
#
#    lens = Optical_element.ideal_lens(p,q)
#    beam = lens.trace_ideal_lens(beam)
#
#    beam.plot_xz()
#
#    if do_plot:
#        plt.show()
#
#    assert_almost_equal(np.abs(beam.x).mean(), 0.0, 4)
#    assert_almost_equal(np.abs(beam.z).mean(), 0.0, 4)
#
#
#def test_ideal_lens_with_trace_optical_element():
#    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  test_ideal_lens_with_trace_optical_element")
#
#    beam=Beam()
#    beam.set_flat_divergence(0.05,0.005)
#
#    p=1.
#    q=5.
#
#    lens = Optical_element.ideal_lens(p,q)
#    beam = lens.trace_optical_element(beam)
#
#    beam.plot_xz()
#    if do_plot:
#        plt.show()
#
#    assert_almost_equal(np.abs(beam.x).mean(), 0.0, 4)
#    assert_almost_equal(np.abs(beam.z).mean(), 0.0, 4)
#

#
###################### Doesn't work
#
#def test_ideal_lens_collimated_beam():
#    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  test_ideal_lens_collimated_beam")
#
#    beam=Beam()
#    beam.set_circular_spot(20)
#    beam.set_divergences_collimated()
#    beam.plot_xz()
#
#    p=1.
#    q=5.
#
#    lens = Optical_element.ideal_lens(p,q,q,q)
#    beam = lens.trace_optical_element(beam)
#
#    beam.plot_xz()
#    if do_plot:
#        plt.show()
#
#    assert_almost_equal(np.abs(beam.x).mean(), 0.0, 4)
#    assert_almost_equal(np.abs(beam.z).mean(), 0.0, 4)
#

