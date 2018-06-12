from Beam import Beam
from OpticalElement import Optical_element
from Shape import BoundaryRectangle
import numpy as np
from SurfaceConic import SurfaceConic
import matplotlib.pyplot as plt

do_plot = True


def test_Object_for_images():
    beam1 = Beam()
    beam1.set_divergences_collimated()
    beam1.set_point(0. + 100, 0., 20. + 100)
    beam1.set_circular_spot(5.)

    beam2 = Beam()
    beam2.set_divergences_collimated()
    beam2.set_point(0. + 100, 0., 0. + 100)
    beam2.set_rectangular_spot(20., -20., 15., 10.)

    beam = beam1.merge(beam2)

    beam3 = Beam(20000)
    beam3.set_divergences_collimated()
    beam3.set_point(0. + 100, 0., 0. + 100)
    beam3.set_rectangular_spot(5., -5., 10., -40.)

    beam = beam.merge(beam3)

    op_ax=Beam(1)
    op_ax.set_divergences_collimated()
    op_ax.set_point(np.mean(beam.x),np.mean(beam.y),np.mean(beam.z))

    print(op_ax.x, op_ax.y, op_ax.z)

    beam=op_ax.merge(beam)
    beam.plot_xz()
    beamd = beam.duplicate()

    test_ray=Beam(1)
    test_ray.set_point(np.mean(beam.x), np.mean(beam.y), np.mean(beam.z))

    p = 10.
    q = 25.
    theta = 0 * np.pi / 180
    alpha = 0 * np.pi / 180
    prova = Optical_element.initialize_as_surface_conic_paraboloid_from_focal_distances(p, q, theta, alpha, "p")
    beamd = prova.trace_optical_element(beamd)
    print(prova.ccc_object.get_coefficients())

    t = (100 - beam.y) / beam.vy

    beamd.x = beamd.x + beamd.vx * t
    beamd.y = beamd.y + beamd.vy * t
    beamd.z = beamd.z + beamd.vz * t

    beamd.plot_xz()

    p = 1000.
    q = 25.
    theta = 0 * np.pi / 180
    alpha = 0 * np.pi / 180
    prova = Optical_element.initialize_as_surface_conic_paraboloid_from_focal_distances(p, q, theta, alpha, "p")
    beam = prova.trace_Wolter_1(beam)

    beam.plot_xz()

    t = (100.) / beam.vy
    beam.x = beam.x + beam.vx * t
    beam.y = beam.y + beam.vy * t
    beam.z = beam.z + beam.vz * t

    beam.plot_xz()

    if do_plot:
        plt.show()

#def test_images_for_lens():
#    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  images_for_lens")
#
#    beam1 = Beam()
#    beam1.set_divergences_collimated()
#    beam1.set_point(0. + 100, 0., 20. + 100)
#    beam1.set_circular_spot(5.)
#    beam2 = Beam()
#    beam2.set_divergences_collimated()
#    beam2.set_point(0. + 100, 0., 0. + 100)
#    beam2.set_rectangular_spot(20., -20., 15., 10.)
#    beam = beam1.merge(beam2)
#    beam3 = Beam()
#    beam3.set_divergences_collimated()
#    beam3.set_point(0. + 100, 0., 0. + 100)
#    beam3.set_rectangular_spot(5., -5., 10., -40.)
#    beam = beam.merge(beam3)
#
#    beam.x=(beam.x-np.mean(beam.x))*1e-3
#    beam.z=(beam.z-np.mean(beam.z))*1e-3
#    beam.plot_xz()
#
#    p=1.
#    q=5
#
#    lens = Optical_element.ideal_lens(p,0,fx=q,fz=q)
#
#    beam = lens.trace_optical_element(beam)
#    beam.retrace(q)
#
#    hyp=Optical_element.initialize_my_hyperboloid(5.-np.sqrt(2),np.sqrt(2),0.0)
#    beam=hyp.trace_optical_element(beam)
#
#    beam.plot_xz(equal_axis=0)
#    beam.retrace(3)
#    beam.plot_xz()
#
#    if do_plot:
#        plt.show()
#