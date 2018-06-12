from Beam import Beam
from OpticalElement import Optical_element
from Shape import BoundaryRectangle
import numpy as np
import matplotlib.pyplot as plt
from numpy.testing import assert_almost_equal
from CompoundOpticalElement import CompoundOpticalElement
from Vector import Vector
from SurfaceConic import SurfaceConic

do_plot = False


def test_wolter1():
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  test_wolter_1")

    p=100.
    beam1 = Beam.initialize_as_person()
    beam1.set_point(p, 0., p)

    op_ax = Beam (1)
    op_ax.set_point(p, 0., p)

    beam=op_ax.merge(beam1)
    beam.set_divergences_collimated()

    beam.plot_xz()


    p = 36.
    q = 25.
    z0 = 0.

    wolter1 = CompoundOpticalElement.initialiaze_as_wolter_1(p, q, z0)

    beam = wolter1.trace_compound(beam)
    beam.plot_xz()

    assert_almost_equal(beam.x, 0., 8)
    assert_almost_equal(beam.z, 0., 8)

    beam.retrace(10.)
    beam.plot_xz()
    plt.title("Wolter 1")

    if do_plot:
        plt.show()


def test_wolter2():
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  test_wolter_2")

    p=26.
    beam1 = Beam.initialize_as_person()
    beam1.set_point(p, 0., p)

    op_ax = Beam (1)
    op_ax.set_point(p, 0., p)

    beam = op_ax.merge(beam1)
    beam.set_divergences_collimated()
    beam.plot_xz()


    p = 200.
    q = 30.
    z0 = 5.

    wolter2 = CompoundOpticalElement.initialiaze_as_wolter_2(p, q, z0)

    beam = wolter2.trace_compound(beam)
    beam.plot_xz()

    assert_almost_equal(beam.x, 0., 8)
    assert_almost_equal(beam.z, 0., 8)

    beam.retrace(10.)

    beam.plot_xz()
    plt.title("Wolter 2")

    if do_plot:
        plt.show()


def test_wolter_3():
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  test_wolter_3")

    p=50.
    beam1 = Beam.initialize_as_person()
    beam1.set_point(p, 0., p)

    op_ax = Beam (1)
    op_ax.set_point(p, 0., p)

    beam = op_ax.merge(beam1)
    beam.set_divergences_collimated()

    beam.plot_xz()

    distance_between_the_foci = 10.

    wolter3 = CompoundOpticalElement.initialize_as_wolter_3(20., 5., distance_between_the_foci)

    beam = wolter3.trace_compound(beam)


    beam.plot_xz()

    assert_almost_equal(beam.x, 0., 8)
    assert_almost_equal(beam.z, 0., 8)

    beam.retrace(10.)
    beam.plot_xz()
    plt.title("Wolter 3")

    if do_plot:
        plt.show()


def test_optimezed_wolter1_good_rays():
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>   test_optimezed_wolter1_good_rays")


    p=100.
    beam1 = Beam.initialize_as_person()
    beam1.x *= 50.
    beam1.z *= 50.
    beam1.set_point(p, 0., p)
    op_ax = Beam (1)
    op_ax.set_point(p, 0., p)
    beam=op_ax.merge(beam1)
    beam.set_divergences_collimated()
    beam.plot_xz()


    p = 1e12
    R= 100.
    theta= 1e-3*np.pi/180

    wolter1 = CompoundOpticalElement.initialiaze_as_wolter_1_with_two_parameters(p1=p, R=R, theta=theta)

    beam = wolter1.trace_with_hole(beam)
    beam.plot_good_xz()

    indices = np.where(beam.flag>=0)

    assert_almost_equal(beam.x[indices], 0., 8)
    assert_almost_equal(beam.z[indices], 0., 8)

    beam.retrace(100.)
    beam.plot_good_xz()
    plt.title("optimezed_wolter1_good_rays")

    if do_plot:
        plt.show()


def test_wolter2_good_rays():
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>   test_wolter2_good_rays")

    p = 0.
    # beam1 = Beam.initialize_as_person(10000)
    beam1 = Beam(100000)
    beam1.set_circular_spot(1.)
    # beam1.set_rectangular_spot(5 / 2 * 1e-5, -5 / 2 * 1e-5, 5 / 2 * 1e-5, -5 / 2 * 1e-5)
    beam1.x *= 1000.
    beam1.z *= 1000.
    beam1.set_point(p, 0., p)

    op_ax = Beam(1)
    op_ax.set_point(p, 0., p)

    beam = op_ax.merge(beam1)
    beam.set_divergences_collimated()
    beam.plot_xz(0)

    p = 20000.
    q = 30.
    z0 = 5.
    focal = 2 * z0 + q

    wolter2 = CompoundOpticalElement.initialiaze_as_wolter_2(p1=p, q1=q, z0=z0)

    beam = wolter2.trace_with_hole(beam)

    beam.plot_good_xz()

    beam.retrace(10.)
    beam.plot_good_xz()
    plt.title("test_wolter2_good_rays")

    if do_plot:
        plt.show()