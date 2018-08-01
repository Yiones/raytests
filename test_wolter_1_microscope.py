from Beam import Beam
from OpticalElement import Optical_element
from Shape import BoundaryRectangle
import numpy as np
from SurfaceConic import SurfaceConic
import matplotlib.pyplot as plt
from CompoundOpticalElement import CompoundOpticalElement
from Vector import Vector


def test_wolter_1_microscope():

    p = 0.
    beam1 = Beam.initialize_as_person()
    beam1.set_flat_divergence_with_different_optical_axis(0.005, 0.005)
    beam1.set_point(p, 0., p)

    op_ax = Beam (1)
    op_ax.set_point(p, 0., p)

    beam = op_ax.merge(beam1)
    beam.x = beam.x
    beam.z = beam.z
    #beam.set_divergences_collimated()

    beam.plot_xz()

    distance_of_focalization = 5.

    hyp = Optical_element.initialize_my_hyperboloid(p=distance_of_focalization, q=0., theta=0., alpha=0., wolter=1.1, z0=0., distance_of_focalization=distance_of_focalization)

    ah = distance_of_focalization/np.sqrt(2)
    q = 20.
    z0 = 0.5*(q - np.sqrt(2)*ah)
    c = z0 + np.sqrt(2)*ah
    #b = c + 0.1
    #a=np.sqrt(b**2-c**2)
    b = c*1.5
    a = np.sqrt(b**2-c**2)
    ccc = np.array ([1/a**2, 1/a**2, 1/b**2, 0., 0., 0., 0., 0., -2*z0/b**2, z0**2/b**2-1])
    ellips = Optical_element.initialize_as_surface_conic_from_coefficients(ccc)


    hyp.effect_of_optical_element(beam)

    beam.plot_yx(0)
    plt.title("footprint %s" % (hyp.type))

    ellips.p = 0.
    ellips.theta = 90*np.pi/180
    ellips.effect_of_optical_element(beam)

    beam.plot_yx(0)
    plt.title("footprint %s" % (ellips.type))

    t = -beam.y/beam.vy
    beam.x = beam.x + beam.vx * t
    beam.y = beam.y + beam.vy * t
    beam.z = beam.z + beam.vz * t


    beam.plot_yx(0)
    beam.plot_xz()

    print("a of ellips is: %f"  %(a))
    print("b of ellips is: %f"  %(b))
    print("c of ellips is: %f"  %(c))
    print(np.mean(beam.z))

    beam.histogram()

    plt.show()