from OpticalElement import Optical_element
from SurfaceConic import SurfaceConic
from Beam import Beam
from Vector import Vector
import matplotlib.pyplot as plt
import numpy as np
import Shadow

main = "__main__"



def shadow_source():


    iwrite = 0

    beam = Shadow.Beam()
    oe0 = Shadow.Source()

    oe0.FDISTR = 1
    oe0.FSOUR = 1
    oe0.F_PHOT = 0
    oe0.HDIV1 = 0.0
    oe0.HDIV2 = 0.0
    oe0.IDO_VX = 0
    oe0.IDO_VZ = 0
    oe0.IDO_X_S = 0
    oe0.IDO_Y_S = 0
    oe0.IDO_Z_S = 0
    oe0.PH1 = 1000.0
    oe0.VDIV1 = 0.0
    oe0.VDIV2 = 0.0

    if iwrite:
        oe0.write("start.00")

    beam.genSource(oe0)

    if iwrite:
        oe0.write("end.00")
        beam.write("begin.dat")


    return beam


if main == "__main__":


    varx = np.zeros(100)
    varz = np.zeros(100)
    qqq  = np.zeros(100)

    #for i in range (0, 1):

    beam = Beam(25000)
    beam.set_circular_spot(1e-6)
    beam.set_divergences_collimated()


    #shadow_beam = shadow_source()
    #beam = Beam()
    #beam.initialize_from_arrays(
    #    shadow_beam.getshonecol(1),
    #    shadow_beam.getshonecol(2),
    #    shadow_beam.getshonecol(3),
    #    shadow_beam.getshonecol(4),
    #    shadow_beam.getshonecol(5),
    #    shadow_beam.getshonecol(6),
    #    shadow_beam.getshonecol(10),
    #    0
    #)



    beam_prova = beam.duplicate()

    p = 5.
    q = 15.
    theta = 89. * np.pi / 180
    beta = -45.* np.pi / 180
    alpha = 87. * np.pi / 180

    xmax = 0.
    xmin = -0.4
    ymax =  0.4
    ymin = -0.4
    zmax =  0.4
    zmin = 0.

    oe1 = Optical_element.initialize_as_surface_conic_paraboloid_from_focal_distances(p=p, q=q, theta=theta, alpha=0., infinity_location="p", focal=q, cylindrical=1)


    ccc1 = oe1.ccc_object.get_coefficients()
    c1 = ccc1[1]
    c2 = ccc1[2]
    c4 = ccc1[4]
    c8 = ccc1[8]
    #######  rotation of the oe around y   #############################################################################

    a = np.cos(beta)
    b = np.sin(beta)


    ccc2 = np.array([c2 * b ** 2, c1, c2 * a ** 2, -c4 * b, c4 * a, -2 * c2 * a * b, -c8 * b, 0., c8 * a, 0.])

    ######## rotation of 90 around z ####################################################################################

    #bp = -1.

    #ccc1p = np.array([c1*bp, 0., c2, 0, 0, c4*bp, 0., 0., c8, 0.])
    #print(ccc1p)

    #c0p = ccc1p[0]
    #c2p = ccc1p[2]
    #c5p = ccc1p[5]
    #c8p = ccc1p[8]

    ########  rotation of the oe around y   #############################################################################

    #a = np.cos(beta)
    #b = np.sin(beta)

    #ccc2 = np.array([ c0p*a**2+c2p*b**2-c5p*a*b, 0., c0p*b**2+c2p*a**2+c5p*a*b, 0., 0., 2*c0p*a*b-2*c2p*a*b+c5p*a**2-c5p*b**2, -c8p*b, 0., c8p*a, 0.])



    oe2 = Optical_element.initialize_as_surface_conic_from_coefficients(ccc2)
    oe2.p = p
    oe2.q = q
    oe2.theta = theta
    oe2.alpha = 0.
    oe2.type = "Surface conical mirror"



    ccc = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0., -q])

    screen = Optical_element.initialize_as_surface_conic_from_coefficients(ccc)
    screen.set_parameters(p, q, 0., 0., "Surface conical mirror")


    print("\n")
    print(oe1.info())
    print("\n")
    print(oe2.info())
    print("\n")



    ##########   rotation     ##############################################################################################

    position = Vector(beam.x, beam.y, beam.z)
    velocity = Vector(beam.vx, beam.vy, beam.vz)

    position.rotation(-(np.pi / 2 - oe2.theta - 0*np.pi/4), "z")
    velocity.rotation(-(np.pi / 2 - oe2.theta - 0*np.pi/4), "z")


    [beam.x, beam.y, beam.z] = [position.x, position.y, position.z]
    [beam.vx, beam.vy, beam.vz] = [velocity.x, velocity.y, velocity.z]


    ########  translation    ###############################################################################################

    vector_point = Vector(0, oe2.p, 0)

    vector_point.rotation(-(np.pi / 2 - oe2.theta - 0*np.pi/4), "z")

    beam.x = beam.x - vector_point.x
    beam.y = beam.y - vector_point.y
    beam.z = beam.z - vector_point.z

    print(vector_point.info())


    print("position before intersection, x = %f, y = %f, z = %f"  %(np.mean(beam.x), np.mean(beam.y), np.mean(beam.z)))
    print("theta(x) = %f, theta(z) = %f" %(np.arctan(vector_point.x/vector_point.y)*180./np.pi, np.arctan(vector_point.z/vector_point.y)*180./np.pi))
    [beam, t] = oe2.intersection_with_optical_element(beam)

    print("\nposition after intersection, x = %f, y = %f, z = %f"  %(np.mean(beam.x), np.mean(beam.y), np.mean(beam.z)))

    oe2.output_direction_from_optical_element(beam)


    dz = np.ones(100)
    qqq = np.ones(100)

    for iii in range (0,100):

        qqq[iii] = q - 0.5 + 1.*iii/100

        t = (qqq[iii]-beam.y)/beam.vy
        beam.x += beam.vx * t
        beam.y += beam.vy * t
        beam.z += beam.vz * t

        dz[iii] = max(beam.z) - min(beam.z)
        beam.x -= beam.vx * t
        beam.y -= beam.vy * t
        beam.z -= beam.vz * t


    plt.figure()
    plt.plot(qqq,dz, 'r.')

    print(min(dz)*1e6)

    plt.show()