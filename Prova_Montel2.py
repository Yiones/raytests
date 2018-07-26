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


    shadow_beam = shadow_source()
    beam = Beam()
    beam.initialize_from_arrays(
        shadow_beam.getshonecol(1),
        shadow_beam.getshonecol(2),
        shadow_beam.getshonecol(3),
        shadow_beam.getshonecol(4),
        shadow_beam.getshonecol(5),
        shadow_beam.getshonecol(6),
        shadow_beam.getshonecol(10),
        0
    )



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

#########################################################################################################################
#
#
#    q=15.
#
#
#    beam1 = beam.duplicate()
#    beam2 = beam.duplicate()
#    beam3 = beam.duplicate()
#
#
#    [beam1, t1] = oe1.intersection_with_optical_element(beam1)
#    oe1.output_direction_from_optical_element(beam1)
#
#    t = (q-beam1.y)/beam1.vy
#    beam1.x += beam1.vx * t
#    beam1.y += beam1.vy * t
#    beam1.z += beam1.vz * t
#
#    beam1.plot_xz(0)
#
#
#    [beam2, t2] = oe2.intersection_with_optical_element(beam2)
#    oe2.output_direction_from_optical_element(beam2)
#
#    t = (q-beam2.y)/beam2.vy
#    beam2.x += beam2.vx * t
#    beam2.y += beam2.vy * t
#    beam2.z += beam2.vz * t
#
#    beam2.plot_xz(0)

########################################################################################################################

    beam1 = beam.duplicate()
    beam2 = beam.duplicate()
    beam3 = beam.duplicate()


    [beam1, t1] = oe1.intersection_with_optical_element(beam1)
    [beam2, t2] = oe2.intersection_with_optical_element(beam2)
    [beam3, t3] = screen.intersection_with_optical_element(beam3)



    ###########  Good rays beam1

    indices = np.where(beam1.x > xmax)
    beam1.flag[indices] = -1
    indices = np.where(beam1.x < xmin)
    beam1.flag[indices] = -1

    indices = np.where(beam1.y > ymax)
    beam1.flag[indices] = -1
    indices = np.where(beam1.y < ymin)
    beam1.flag[indices] = -1

    indices = np.where(beam1.z > zmax)
    beam1.flag[indices] = -1
    indices = np.where(beam1.z < zmin)
    beam1.flag[indices] = -1


    ##########  Good rays beam2

    indices = np.where(beam2.x > xmax)
    beam2.flag[indices] = -1
    indices = np.where(beam2.x < xmin)
    beam2.flag[indices] = -1

    indices = np.where(beam2.y > ymax)
    beam2.flag[indices] = -1
    indices = np.where(beam2.y < ymin)
    beam2.flag[indices] = -1

    indices = np.where(beam2.z > zmax)
    beam2.flag[indices] = -1
    indices = np.where(beam2.z < zmin)
    beam2.flag[indices] = -1

    ######first iteration##################################################################################################

    indices1 = np.where(beam1.flag < 0)
    indices2 = np.where(beam2.flag < 0)


    maxim = max(abs(np.max(t1)), abs(np.max(t2)))


    t1[indices1] = 1e12 * np.ones(np.size(indices1))
    t2[indices2] = 1e12 * np.ones(np.size(indices1))

    t = np.minimum(t1, t2)
    origin = np.ones(beam1.N)

    indices = np.where(t2 < t1)
    origin[indices] += 1

    indices = np.where(t == 1e12)
    origin[indices] = 3

    beam3.flag += -1
    beam3.flag[indices] = 0

    indices = np.where(beam3.flag >= 0)


    t[indices] = t3[indices]

    #####3 good beam####################################################################################################

    indices1 = np.where(origin == 1)
    beam01 = beam1.part_of_beam(indices1)
    indices1 = np.where(origin == 2)
    beam02 = beam2.part_of_beam(indices1)
    indices1 = np.where(origin == 3)
    beam03 = beam3.part_of_beam(indices1)

    print("The rays arriving at the different oe are: %f, %f, %f" %(beam01.N, beam02.N, beam03.N))
    print(origin)


    #######  Starting the for cicle   ###################################################################################

    beam1_list = [beam01.duplicate(), Beam(), Beam(), Beam(), Beam()]
    beam2_list = [beam02.duplicate(), Beam(), Beam(), Beam(), Beam()]
    beam3_list = [beam03.duplicate(), Beam(), Beam(), Beam(), Beam()]

    for i in range(0, 2):
        ##### oe1 beam

        oe1.output_direction_from_optical_element(beam1_list[i])
        beam1_list[i].flag *= 0

        beam2_list[i + 1] = beam1_list[i].duplicate()
        beam3_list[i + 1] = beam1_list[i].duplicate()

        [beam2_list[i + 1], t2] = oe2.intersection_with_optical_element(beam2_list[i + 1])
        [beam3_list[i + 1], t3] = screen.intersection_with_optical_element(beam3_list[i + 1])

        origin = 2 * np.ones(beam1_list[i].N)

        indices = np.where(beam2_list[i + 1].x > xmax)
        beam2_list[i + 1].flag[indices] = -1
        indices = np.where(beam2_list[i + 1].x < xmin)
        beam2_list[i + 1].flag[indices] = -1

        indices = np.where(beam2_list[i + 1].y > ymax)
        beam2_list[i + 1].flag[indices] = -1
        indices = np.where(beam2_list[i + 1].y < ymin)
        beam2_list[i + 1].flag[indices] = -1

        indices = np.where(beam2_list[i + 1].z > zmax)
        beam2_list[i + 1].flag[indices] = -1
        indices = np.where(beam2_list[i + 1].z < zmin)
        beam2_list[i + 1].flag[indices] = -1

        maxim = 2 * max(max(abs(t2)), max(abs(t3)))

        indices = np.where(beam2_list[i + 1].flag < 0)
        t2[indices] += 2 * maxim

        t = np.minimum(t2, t3)

        indices = np.where(t3 < t2)
        origin[indices] = 3

        print(np.size(np.where(origin==2)), np.size(np.where(origin==3)))


        beam03 = beam3_list[i + 1].part_of_beam(indices)

        indices = np.where(origin == 2)
        beam2_list[i + 1] = beam2_list[i + 1].part_of_beam(indices)

        ####oe2 beam

        oe2.output_direction_from_optical_element(beam2_list[i])
        beam2_list[i].flag *= 0

        beam1_list[i + 1] = beam2_list[i].duplicate()
        beam3_list[i + 1] = beam2_list[i].duplicate()

        [beam1_list[i + 1], t1] = oe1.intersection_with_optical_element(beam1_list[i + 1])
        [beam3_list[i + 1], t3] = screen.intersection_with_optical_element(beam3_list[i + 1])

        origin02 = np.ones(beam2_list[i].N)

        indices = np.where(beam1_list[i + 1].x > xmax)
        beam1_list[i + 1].flag[indices] = -1
        indices = np.where(beam1_list[i + 1].x < xmin)
        beam1_list[i + 1].flag[indices] = -1

        indices = np.where(beam1_list[i + 1].y > ymax)
        beam1_list[i + 1].flag[indices] = -1
        indices = np.where(beam1_list[i + 1].y < ymin)
        beam1_list[i + 1].flag[indices] = -1

        indices = np.where(beam1_list[i + 1].z > zmax)
        beam1_list[i + 1].flag[indices] = -1
        indices = np.where(beam1_list[i + 1].z < zmin)
        beam1_list[i + 1].flag[indices] = -1

        maxim = 2 * max(max(abs(t1)), max(abs(t3)))

        indices = np.where(beam1_list[i + 1].flag < 0)
        t1[indices] += 2 * maxim

        t = t1.copy()
        t[indices] = t3[indices]
        print(t1, t3)

        origin02[indices] += 2

        indices = np.where(origin02 == 3)

        print(np.size(np.where(origin02==1)), np.size(np.where(origin02==3)))

        beam003 = beam3_list[i + 1].part_of_beam(indices)
        beam3_list[i + 1] = beam03.merge(beam003)

        indices = np.where(origin02 == 1)
        beam1_list[i + 1] = beam1_list[i + 1].part_of_beam(indices)


    plt.figure()
    plt.plot(beam3_list[0].x, beam3_list[0].z, 'ro')
    plt.plot(beam3_list[1].x, beam3_list[1].z, 'bo')
    plt.plot(beam3_list[2].x, beam3_list[2].z, 'go')
    #plt.plot(beam1_list[3].x, beam1_list[3].z, 'yo')
    #plt.plot(beam1_list[4].x, beam1_list[4].z, 'ko')
    plt.xlabel('x axis')
    plt.ylabel('z axis')
    plt.axis('equal')

    print("No reflection: %d\nOne reflection: %d\nTwo reflection: %d" %(beam3_list[0].N, beam3_list[1].N, beam3_list[2].N))

    beam3_list[0].plot_xz(0)
    beam3_list[2].plot_xz(0)


    plt.show()