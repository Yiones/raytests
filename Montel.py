from OpticalElement import Optical_element
from SurfaceConic import SurfaceConic
from Beam import Beam
from Vector import Vector
import matplotlib.pyplot as plt
import numpy as np


case = "Normal"            ####  "Rotated"    or      "Normal"



def shadow_source():
    #
    # Python script to run shadow3. Created automatically with ShadowTools.make_python_script_from_list().
    #
    import Shadow
    import numpy

    # write (1) or not (0) SHADOW files start.xx end.xx star.xx
    iwrite = 0

    #
    # initialize shadow3 source (oe0) and beam
    #
    beam = Shadow.Beam()
    oe0 = Shadow.Source()

    #
    # Define variables. See meaning of variables in:
    #  https://raw.githubusercontent.com/srio/shadow3/master/docs/source.nml
    #  https://raw.githubusercontent.com/srio/shadow3/master/docs/oe.nml
    #

    oe0.FDISTR = 1
    oe0.FSOUR = 0
    oe0.F_PHOT = 0
    oe0.HDIV1 = 0.001
    oe0.HDIV2 = 0.001
    oe0.IDO_VX = 0
    oe0.IDO_VZ = 0
    oe0.IDO_X_S = 0
    oe0.IDO_Y_S = 0
    oe0.IDO_Z_S = 0
    oe0.NPOINT = 25000
    oe0.PH1 = 1000.0
    oe0.VDIV1 = 0.01
    oe0.VDIV2 = 0.01

    # Run SHADOW to create the source

    if iwrite:
        oe0.write("start.00")

    beam.genSource(oe0)

    if iwrite:
        oe0.write("end.00")
        beam.write("begin.dat")


    return beam


main = "__main__2__"

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






if main == "__main__2__":



    beam = Beam(25000)
    beam.set_flat_divergence(dx=0.001, dz=0.001)
    #beam.set_divergences_collimated()

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

    p = 5.
    q = 15.
    theta = 89.5 * np.pi / 180
    beta = (90. + 0.) * np.pi / 180
    alpha = 87. * np.pi / 180

    xmax =  0.
    xmin = -0.9
    ymax =  0.9
    ymin = -0.9
    zmax =  0.9
    # print("zmax = %f" %(zmax))
    # zmax = 0.4
    zmin = 0.

    oe1 = Optical_element.initialize_as_surface_conic_ellipsoid_from_focal_distances(p=p, q=q, theta=theta, alpha=0.,
                                                                                     cylindrical=1)


    ccc1 = oe1.ccc_object.get_coefficients()
    c1 = ccc1[1]
    c2 = ccc1[2]
    c4 = ccc1[4]
    c8 = ccc1[8]
    #######  rotation of the oe around y   #############################################################################

    a = np.cos(beta)
    b = np.sin(beta)

    ccc2 = np.array([c2 * b ** 2, c1, c2 * a ** 2, -c4 * b, c4 * a, -2 * c1 * a * b, -c8 * b, 0., c8 * a, 0.])

    oe2 = Optical_element.initialize_as_surface_conic_from_coefficients(ccc2)
    oe2.p = p
    oe2.q = q
    oe2.theta = theta
    oe2.alpha = 0.
    oe2.type = "Surface conical mirror"


    # ccc = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0., -(q-0.055)])
    ccc = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0., -q])

    screen = Optical_element.initialize_as_surface_conic_from_coefficients(ccc)
    screen.set_parameters(p, q, 0., 0., "Surface conical mirror")

    # oe2 = Optical_element.initialize_as_surface_conic_ellipsoid_from_focal_distances(p=p, q=q, theta=theta, alpha=0., cylindrical=1)

    # beam.plot_xpzp()

########################################################################################################################
    theta = np.pi/2 - theta

    vector = Vector(0., 1., 0.)
    vector.rotation(-theta, 'x')
    print(vector.info())


    ny = -vector.z / np.sqrt(vector.y ** 2 + vector.z ** 2)
    nz = vector.y / np.sqrt(vector.y ** 2 + vector.z ** 2)

    n = Vector(0, ny, nz)

    vrot = vector.rodrigues_formula(n, -theta)
    vrot.normalization()

    print("theta' = %f, theta'' = %f" % (
    np.arctan(vrot.x / vrot.y) * 180 / np.pi, np.arctan(vrot.z / vrot.y) * 180 / np.pi))


    ####################################################################################################################

    position = Vector(beam.x, beam.y, beam.z)
    mod_position = position.modulus()
    velocity = Vector(beam.vx, beam.vy, beam.vz)

    position.rotation(-theta, 'x')
    velocity.rotation(-theta, 'x')

    position = position.rodrigues_formula(n, -theta)
    velocity = velocity.rodrigues_formula(n, -theta)
    velocity.normalization()

    #position.normalization()
    position.x = position.x #* mod_position
    position.y = position.y #* mod_position
    position.z = position.z #* mod_position

    [beam.x, beam.y, beam.z] = [position.x, position.y, position.z]
    [beam.vx, beam.vy, beam.vz] = [velocity.x, velocity.y, velocity.z]

    ####### translation  ###############################################################################################

    vector_point = Vector(0, p, 0)

    vector_point.rotation(-(np.pi / 2 - oe2.theta), "x")
    vector_point = vector_point.rodrigues_formula(n, -theta)
    vector_point.normalization()

    beam.x = beam.x - vector_point.x * p
    beam.y = beam.y - vector_point.y * p
    beam.z = beam.z - vector_point.z * p

    beam1 = beam.duplicate()
    beam2 = beam.duplicate()
    beam3 = beam.duplicate()
    beam0 = beam.duplicate()


    #####################################################################################################################

    print("theta' = %f, theta'' = %f" %(np.arctan(np.mean(beam.x)/np.mean(beam.y)),np.arctan(np.mean(beam.z)/np.mean(beam.y)) ))

    #######  Before for        ##############################################################################################


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

        indices = np.where(beam2_list[i + 1].z > ymax)
        beam2_list[i + 1].flag[indices] = -1
        indices = np.where(beam2_list[i + 1].z < ymin)
        beam2_list[i + 1].flag[indices] = -1

        indices = np.where(beam2_list[i + 1].y > zmax)
        beam2_list[i + 1].flag[indices] = -1
        indices = np.where(beam2_list[i + 1].y < zmin)
        beam2_list[i + 1].flag[indices] = -1

        maxim = 2 * max(max(abs(t2)), max(abs(t3)))

        indices = np.where(beam2_list[i + 1].flag < 0)
        t2[indices] += 2 * maxim

        t = np.minimum(t2, t3)

        indices = np.where(t3 < t2)
        origin[indices] = 3

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

        origin02[indices] += 2

        indices = np.where(origin02 == 3)

        beam003 = beam3_list[i + 1].part_of_beam(indices)
        beam3_list[i + 1] = beam03.merge(beam003)

        indices = np.where(origin02 == 1)
        beam1_list[i + 1] = beam1_list[i + 1].part_of_beam(indices)

    plt.figure()
    plt.plot(beam3_list[0].x, beam3_list[0].z, 'ro')
    plt.plot(beam3_list[1].x, beam3_list[1].z, 'bo')
    plt.plot(beam3_list[2].x, beam3_list[2].z, 'go')
    # plt.plot(beam1_list[3].x, beam1_list[3].z, 'yo')
    # plt.plot(beam1_list[4].x, beam1_list[4].z, 'ko')
    plt.xlabel('x axis')
    plt.ylabel('z axis')
    plt.axis('equal')
    plt.title('final plot')


    beam3_list[0].plot_xz(0)
    beam3_list[2].plot_xz(0)

    print(beam3_list[0].N, beam3_list[1].N, beam3_list[2].N)

    plt.show()