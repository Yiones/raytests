from OpticalElement import Optical_element
from SurfaceConic import SurfaceConic
from Beam import Beam
from Vector import Vector
import matplotlib.pyplot as plt
import numpy as np
import Shadow
from Shape import  BoundaryRectangle
from CompoundOpticalElement import CompoundOpticalElement


#def test_montel_paraboloid():
#    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  test_montel_paraboloid")
#
#    beam = Beam(25000)
#    #beam = Beam.initialize_as_person(25000)
#    beam.set_flat_divergence(25*1e-4, 25*1e-4)
#    beam.set_rectangular_spot(xmax=25*1e-3, xmin=-25*1e-3, zmax=5*1e-3, zmin=-5*1e-3)
#    #beam.set_gaussian_divergence(25*1e-6, 25*1e-6)
#    #beam.set_divergences_collimated()
#
#
#    beam.flag *= 0
#
#    p = 5.
#    q = 15.
#    theta = 80.*np.pi/180
#
#    xmax = 0.
#    xmin = -0.3
#    ymax =  0.1
#    ymin = -0.1
#    zmax =  0.3
#    zmin = 0.
#
#    bound1 = BoundaryRectangle(xmax, xmin, ymax, ymin, zmax, zmin)
#    bound2 = BoundaryRectangle(xmax, xmin, ymax, ymin, zmax, zmin)
#
#
#    montel = CompoundOpticalElement.initialize_as_montel_parabolic(p=p, q=q, theta=theta, bound1=bound1, bound2=bound2, distance_of_the_screen=q-6)
#    beam03 = montel.trace_montel(beam)
#
#    print(beam03[2].N/25000)
#
#    plt.figure()
#    plt.plot(beam03[0].x, beam03[0].z, 'ro')
#    plt.plot(beam03[1].x, beam03[1].z, 'bo')
#    plt.plot(beam03[2].x, beam03[2].z, 'go')
#    plt.xlabel('x axis')
#    plt.ylabel('z axis')
#    plt.axis('equal')
#
#    beam03[2].plot_xz(0)
#
#    print("No reflection = %d\nOne reflection = %d\nTwo reflection = %d" %(beam03[0].N, beam03[1].N, beam03[2].N))
#
#
#    plt.show()


def test_montel_elliptical():
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  test_montel_elliptical")

    beam = Beam(25000)
    beam.set_flat_divergence(25*1e-6, 25*1e-6)
    beam.set_rectangular_spot(xmax=25*1e-6, xmin=-25*1e-6, zmax=5*1e-6, zmin=-5*1e-6)
    beam.set_gaussian_divergence(25*1e-4, 25*1e-4)



    beam.flag *= 0

    p = 5.
    q = 15.
    #theta = np.pi/2 - 0.15
    theta = 85. * np.pi / 180

    xmax = 0.
    xmin = -0.3
    ymax =  0.1
    ymin = -0.1
    zmax =  0.3
    zmin = 0.

    bound1 = BoundaryRectangle(xmax, xmin, ymax, ymin, zmax, zmin)
    bound2 = BoundaryRectangle(xmax, xmin, ymax, ymin, zmax, zmin)


    montel = CompoundOpticalElement.initialize_as_montel_ellipsoid(p=p, q=q, theta=theta, bound1=bound1, bound2=bound2)
    beam03 = montel.trace_montel(beam)

    print(beam03[2].N/25000)

    plt.figure()
    plt.plot(beam03[0].x, beam03[0].z, 'ro')
    plt.plot(beam03[1].x, beam03[1].z, 'bo')
    plt.plot(beam03[2].x, beam03[2].z, 'go')
    plt.xlabel('x axis')
    plt.ylabel('z axis')
    plt.axis('equal')

    beam03[2].plot_xz(0)

    print("No reflection = %d\nOne reflection = %d\nTwo reflection = %d" %(beam03[0].N, beam03[1].N, beam03[2].N))

    plt.show()
