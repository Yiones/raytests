from OpticalElement import Optical_element
from SurfaceConic import SurfaceConic
from Beam import Beam
from Vector import Vector
import matplotlib.pyplot as plt
import numpy as np


case = "Rotated2"            ####  "Rotated"    or      "Normal"

main = "__main__"

#class Montel(object):
#
#    def __init__(self):
#
#        self.p = 0.
#        self.q = 0.
#        self.theta = 0.
#
#        self.ccc1 = None
#        self.ccc2 = None
#




if main == "__main__":

    beam = Beam()
    beam.set_flat_divergence(dx=0.001, dz=0.01)
    beam.set_divergences_collimated()

    #beam.plot_xz()

    beam2 = beam.duplicate()


    p = 10.
    q = 5.
    theta = 88.*np.pi/180
    beta = 90*np.pi/180
    alpha = 90*np.pi/180 - theta

    oe1 = Optical_element.initialize_as_surface_conic_ellipsoid_from_focal_distances(p=p, q=q, theta= theta, alpha=0., cylindrical=1)
    #oe2 = Optical_element.initialize_as_surface_conic_ellipsoid_from_focal_distances(p=p, q=q, theta= theta, alpha=alpha, cylindrical=1)

    print("oe1 coefficients")
    print(oe1.ccc_object.get_coefficients())
    #print("oe2 coefficients")
    #print(oe2.ccc_object.get_coefficients())

    #beam = oe2.trace_optical_element()




    ccc1 = oe1.ccc_object.get_coefficients()
    c1 = ccc1[1]
    c2 = ccc1[2]
    c4 = ccc1[4]
    c8 = ccc1[8]
#######  around y   ####################################################################################################


    a = np.cos(beta)
    b = np.sin(beta)

    ccc2 = np.array([c2*b**2, c1, c2*a**2, -c4*b, c4*a, -2*c1*a*b, -c8*b, 0., c8*a, 0.])

    oe2 = Optical_element.initialize_as_surface_conic_from_coefficients(ccc2)
    oe2.p = p
    oe2.q = q
    oe2.theta = theta
    oe2.alpha = 0.
    oe2.type = "Surface conical mirror"


    print(oe2.ccc_object.get_coefficients())

    if case == "Normal":
        oe2 = Optical_element.initialize_as_surface_conic_ellipsoid_from_focal_distances(p=p, q=q, theta= theta, alpha=0., cylindrical=1)

########  around x   ####################################################################################################
#
#    a = np.cos(beta)
#    b = np.sin(beta)
#
#    ccc2 = np.array([0., c1*a**2+c2*b**2+c4*a*b, c1*b**2+c2*a**2-c4*a*b, 0., -2*a*b*c1+2*a*b*c2+c4*a**2-c4*b**2, 0., 0., c8*b, c8*a, 0.])
#
#    oe2 = Optical_element.initialize_as_surface_conic_from_coefficients(ccc2)
#    oe2.p = 3.
#    oe2.q = 5.
#    oe2.theta = theta
#    oe2.alpha = 0.
#    oe2.type = "Surface conical mirror"
#
#    print(oe2.ccc_object.get_coefficients())
#    #beam = oe2.trace_optical_element(beam)
#
##########   rotation     ################################################################################################

    position = Vector(beam.x, beam.y, beam.z)
    velocity = Vector(beam.vx, beam.vy, beam.vz)
    if case == "Rotated":
        position.rotation(-(np.pi / 2 - oe2.theta), "z")
        velocity.rotation(-(np.pi / 2 - oe2.theta), "z")
    if case == "Normal":
        position.rotation(-(np.pi / 2 - oe2.theta), "x")
        velocity.rotation(-(np.pi / 2 - oe2.theta), "x")
    elif case == "combined":
        position.rotation(-(np.pi / 2 - oe2.theta), "z")
        velocity.rotation(-(np.pi / 2 - oe2.theta), "z")
        position.rotation(-(np.pi / 2 - oe2.theta), "x")
        velocity.rotation(-(np.pi / 2 - oe2.theta), "x")

    [beam.x, beam.y, beam.z] = [position.x, position.y, position.z]
    [beam.vx, beam.vy, beam.vz] = [velocity.x, velocity.y, velocity.z]


########  translation    ################################################################################################

    vector_point = Vector(0, oe2.p, 0)
    if case == "Rotated":
        vector_point.rotation(-(np.pi / 2 - oe2.theta), "z")
    elif case == "Normal":
        vector_point.rotation(-(np.pi / 2 - oe2.theta), "x")
    elif case == "Combined":
        vector_point.rotation(-(np.pi / 2 - oe2.theta), "z")
        vector_point.rotation(-(np.pi / 2 - oe2.theta), "x")


    beam.x = beam.x - vector_point.x
    beam.y = beam.y - vector_point.y
    beam.z = beam.z - vector_point.z

########  others    ####################################################################################################

    [beam, t] = oe2.intersection_with_optical_element(beam)
    oe2.output_direction_from_optical_element(beam)

########   rotation to the screen   #####################################################################################
#
#    position = Vector(beam.x, beam.y, beam.z)
#    velocity = Vector(beam.vx, beam.vy, beam.vz)
#
#    if case == "Rotated":
#        position.rotation(-(np.pi / 2 - oe2.theta), "z")
#        velocity.rotation(-(np.pi / 2 - oe2.theta), "z")
#    elif case == "Normal":
#        position.rotation(-(np.pi / 2 - oe2.theta), "x")
#        velocity.rotation(-(np.pi / 2 - oe2.theta), "x")
#
#    [beam.x, beam.y, beam.z] = [position.x, position.y, position.z]
#    [beam.vx, beam.vy, beam.vz] = [velocity.x, velocity.y, velocity.z]
#
######  translation to the screen   ####################################################################################
#
#    oe2.translation_to_the_screen(beam)
#
#######  intersection to the screen   ###################################################################################
#
#    oe2.intersection_with_the_screen(beam)
#
##########################################################################################################################



    #oe1 = Optical_element.initialize_as_surface_conic_ellipsoid_from_focal_distances(p=p, q=q, theta= theta, alpha=0., cylindrical=1)
    #oe1.theta = 0.
    #beam2 = oe1.trace_optical_element(beam2)
    #beam2.plot_xz(0)
    #plt.title("Normal one")


    if case == "Rotated":
        #t = -beam.z/beam.vz
        #beam.x = beam.x + beam.vx * t
        #beam.y = beam.y + beam.vy * t
        #beam.z = beam.z + beam.vz * t

        #beam.plot_yx(0)
        beam.plot_xz()
        plt.title(case)

    elif case == "Normal":
        #t = -beam.x/beam.vx
        #beam.x = beam.x + beam.vx * t
        #beam.y = beam.y + beam.vy * t
        #beam.z = beam.z + beam.vz * t

        #beam.plot_zy(0)
        beam.plot_xz()
        plt.title(case)

    elif case == "combined":

        t = -beam.x/beam.vx
        beam.x = beam.x + beam.vx * t
        beam.y = beam.y + beam.vy * t
        beam.z = beam.z + beam.vz * t

        beam.plot_zy(0)
        plt.title(case)

    #oe3 = Optical_element.initialize_as_surface_conic_ellipsoid_from_focal_distances(p=p, q=q, theta= theta, alpha=90*np.pi/180, cylindrical=1)
    #beam2 = oe3.trace_optical_element(beam2)


    #beam2.plot_xz(0)

    print(oe2.info())

    t = -beam.x / beam.vx
    beam.x = beam.x + beam.vx * t
    beam.y = beam.y + beam.vy * t
    beam.z = beam.z + beam.vz * t

    beam.plot_zy(0)
    beam.plot_xz(0)
    plt.title(case)

    plt.show()


