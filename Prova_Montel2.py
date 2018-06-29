from OpticalElement import Optical_element
from SurfaceConic import SurfaceConic
from Beam import Beam
from Vector import Vector
import matplotlib.pyplot as plt
import numpy as np
import Shadow

main = "__main__"
axis = 'z'
both = None
plot_dim = 0

if both == True:
    axis = 'z'
    axis1 = 'x'



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
    beam.set_circular_spot(1e-3)
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
    theta = 88. * np.pi / 180
    beta = 90.* np.pi / 180
    alpha = 87. * np.pi / 180

    xmax = 0.
    xmin = -0.4
    ymax =  0.4
    ymin = -0.4
    zmax =  0.4
    zmin = 0.

    oe1 = Optical_element.initialize_as_surface_conic_paraboloid_from_focal_distances(p=p, q=q, theta=theta, alpha=0., infinity_location="p", focal=q, cylindrical=1)


    ccc1 = oe1.ccc_object.get_coefficients()
    c1 = ccc1[1].copy()
    c2 = ccc1[2].copy()
    c4 = ccc1[4].copy()
    c8 = ccc1[8].copy()
    #######  rotation of the oe around y   #############################################################################

    a = np.cos(beta)
    b = np.sin(beta)


    ccc2 = np.array([c2 * b ** 2, c1, c2 * a ** 2, -c4 * b, c4 * a, -2 * c2 * a * b, -c8 * b, 0., c8 * a, 0.])


    oe2 = Optical_element.initialize_as_surface_conic_from_coefficients(ccc2)
    oe2.p = p
    oe2.q = q
    oe2.theta = theta
    oe2.alpha = 0.
    oe2.type = "Surface conical mirror"


    oe1 = Optical_element.initialize_as_surface_conic_paraboloid_from_focal_distances(p=p, q=q, theta=theta, alpha=0., infinity_location="p", focal=q, cylindrical=1)

    ccc = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0., -q])

    screen = Optical_element.initialize_as_surface_conic_from_coefficients(ccc)
    screen.set_parameters(p, q, 0., 0., "Surface conical mirror")

    theta = np.pi/2 - theta

###########################################################################################################################
###
###    position = Vector(beam.x, beam.y, beam.z)
###    position.rotation(-theta, 'x')
###    position.rotation(-theta, 'z')
###
###    [beam.x, beam.y, beam.z] = [position.x, position.y, position.z]
###
###
###########################################################################################################################
###    y = p/np.sqrt(1+2*np.tan(theta)**2)
###    z = y* np.tan(theta)
###    x = -y * np.tan(theta)
###
###    v = Vector(-x, y, -z)
###    v.normalization()
###
###    beam.x += x
###    beam.y -= y
###    beam.z += z
###
###
###
###
###    beam.vx = beam.vx*0 + v.x
###    beam.vy = beam.vy*0 + v.y
###    beam.vz = beam.vz*0 + v.z
###
###
###########################################################################################################################


    vector = Vector(0., 1., 0.)
    vector.rotation(-theta, 'x')
    print(vector.info())

    print("theta' = %f, theta'' = %f" %(np.arctan(vector.x/vector.y)*180/np.pi, np.arctan(vector.z/vector.y)*180/np.pi))

    ny = -vector.z/np.sqrt(vector.y**2+vector.z**2)
    nz = vector.y/np.sqrt(vector.y**2+vector.z**2)

    n = Vector(0, ny, nz)

    vrot = vector.rodrigues_formula(n, -theta)
    vrot.normalization()


    print("theta' = %f, theta'' = %f" %(np.arctan(vrot.x/vrot.y)*180/np.pi, np.arctan(vrot.z/vrot.y)*180/np.pi))

    print(vrot.info())

#########################################################################################################################


    position = Vector(beam.x, beam.y, beam.z)
    mod_position = position.modulus()
    velocity = Vector(beam.vx, beam.vy, beam.vz)

    position.rotation(-theta, 'x')
    velocity.rotation(-theta, 'x')

    position = position.rodrigues_formula(n, -theta)
    velocity = velocity.rodrigues_formula(n, -theta)
    velocity.normalization()

    position.normalization()
    position.x = position.x * mod_position
    position.y = position.y * mod_position
    position.z = position.z * mod_position


    [beam.x, beam.y, beam.z] = [position.x, position.y, position.z]
    [beam.vx, beam.vy, beam.vz] = [velocity.x, velocity.y, velocity.z]



####### translation  ###################################################################################################


    vector_point = Vector(0, p, 0)

    vector_point.rotation(-(np.pi / 2 - oe2.theta - 0*np.pi/4), "x")
    vector_point = vector_point.rodrigues_formula(n, -theta)
    vector_point.normalization()


    beam.x = beam.x - vector_point.x * p
    beam.y = beam.y - vector_point.y * p
    beam.z = beam.z - vector_point.z * p

########################################################################################################################


    q=15.


    beam1 = beam.duplicate()
    beam2 = beam.duplicate()
    beam3 = beam.duplicate()


    [beam1, t1] = oe1.intersection_with_optical_element(beam1)
    oe1.output_direction_from_optical_element(beam1)

    t = (q-beam1.y)/beam1.vy
    beam1.x += beam1.vx * t
    beam1.y += beam1.vy * t
    beam1.z += beam1.vz * t

    beam1.plot_xz()


    [beam2, t2] = oe2.intersection_with_optical_element(beam2)
    oe2.output_direction_from_optical_element(beam2)

    t = (q-beam2.y)/beam2.vy
    beam2.x += beam2.vx * t
    beam2.y += beam2.vy * t
    beam2.z += beam2.vz * t

    beam2.plot_xz()


    [beam3, t3] = oe1.intersection_with_optical_element(beam3)
    oe1.output_direction_from_optical_element(beam3)
    [beam3, t3] = oe2.intersection_with_optical_element(beam3)
    oe2.output_direction_from_optical_element(beam3)

    Nn = 5000
    qqq = np.ones(Nn)
    dx = np.ones(Nn)

    for i in range (Nn):

        qqq[i] = q - 0.1 + 0.2 *i / Nn

        t = (qqq[i]-beam3.y)/beam3.vy
        beam3.x += beam3.vx * t
        beam3.y += beam3.vy * t
        beam3.z += beam3.vz * t

        dx[i] = max(beam3.z) - min(beam3.z)

        beam3.x -= beam3.vx * t
        beam3.y -= beam3.vy * t
        beam3.z -= beam3.vz * t


    plt.figure()
    plt.plot(qqq, dx, 'r.')





    index = np.where(min(dx))
    t = (qqq[index]-beam3.y)/beam3.vy
    beam3.x += beam3.vx * t
    beam3.y += beam3.vy * t
    beam3.z += beam3.vz * t


    beam3.plot_xz(0)
    beam3.histogram()


    plt.figure()
    plt.plot(beam1.x, beam1.z, 'b.')
    plt.plot(beam2.x, beam2.z, 'y.')
    plt.plot(beam3.x, beam3.z, 'g.')


    plt.show()


