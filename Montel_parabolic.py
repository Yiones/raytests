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


    oe0.FDISTR = 3
    oe0.F_PHOT = 0
    oe0.HDIV1 = 1.0
    oe0.HDIV2 = 1.0
    oe0.IDO_VX = 0
    oe0.IDO_VZ = 0
    oe0.IDO_X_S = 0
    oe0.IDO_Y_S = 0
    oe0.IDO_Z_S = 0
    oe0.NPOINT = 25000
    oe0.PH1 = 10000.0
    oe0.SIGDIX = 8.84999972e-05
    oe0.SIGDIZ = 7.1999998e-06
    oe0.SIGMAX = 5.7000001e-05
    oe0.SIGMAZ = 1.04e-05
    oe0.VDIV1 = 1.0
    oe0.VDIV2 = 1.0


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

    #beam = Beam()
    #beam.set_flat_divergence(0.01,0.01)



    beam_prova = beam.duplicate()

    p = 5.
    q = 15.
    theta = 88. * np.pi / 180
    beta = (90. + 0.) * np.pi / 180
    alpha = 87. * np.pi / 180

    xmax = 0.
    xmin = -0.4
    ymax =  0.4
    ymin = -0.4
    zmax =  0.4
    # print("zmax = %f" %(zmax))
    # zmax = 0.4
    zmin = 0.

    oe1 = Optical_element.initialize_as_surface_conic_paraboloid_from_focal_distances(p=p, q=q, theta=theta, alpha=0., infinity_location="p", focal=q, cylindrical=1)


    ccc1 = oe1.ccc_object.get_coefficients()
    print(ccc1)
    c1 = ccc1[1]
    c2 = ccc1[2]
    c4 = ccc1[4]
    c8 = ccc1[8]
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

    # print("\n")
    # print(oe1.info())
    # print("\n")
    # print(oe2.info())
    # print("\n")


    ccc = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0., -q])

    screen = Optical_element.initialize_as_surface_conic_from_coefficients(ccc)
    screen.set_parameters(p, q, 0., 0., "Surface conical mirror")

    # oe2 = Optical_element.initialize_as_surface_conic_ellipsoid_from_focal_distances(p=p, q=q, theta=theta, alpha=0., cylindrical=1)

    # beam.plot_xpzp()

    ##########   rotation     ##############################################################################################

    position = Vector(beam.x, beam.y, beam.z)
    velocity = Vector(beam.vx, beam.vy, beam.vz)

    position.rotation(-(np.pi / 2 - oe1.theta), "x")
    velocity.rotation(-(np.pi / 2 - oe1.theta), "x")
    position.rotation(-(np.pi / 2 - oe2.theta), "z")
    velocity.rotation(-(np.pi / 2 - oe2.theta), "z")

    [beam.x, beam.y, beam.z] = [position.x, position.y, position.z]
    [beam.vx, beam.vy, beam.vz] = [velocity.x, velocity.y, velocity.z]

    ########  translation    ###############################################################################################

    vector_point = Vector(0, oe2.p, 0)

    vector_point.rotation(-(np.pi / 2 - oe2.theta), "z")
    vector_point.rotation(-(np.pi / 2 - oe2.theta), "x")

    beam.x = beam.x - vector_point.x
    beam.y = beam.y - vector_point.y
    beam.z = beam.z - vector_point.z


    # print("beam.x = %f, beam.y = %f, beam.z = %f" % (np.mean(beam.x), np.mean(beam.y), np.mean(beam.z)))
    # print("beam.vx = %f, beam.vy = %f, beam.vz = %f" % (np.mean(beam.vx), np.mean(beam.vy), np.mean(beam.vz)))

    # print("theta angle = %f"  %((np.arctan(np.mean(beam.z/beam.y))*180/np.pi)))
    # print("fi angle = %f"  %((np.arctan(np.mean(beam.x/beam.y))*180/np.pi)))

    ###### beam separation   ###############################################################################################

    beam1 = beam.duplicate()
    beam2 = beam.duplicate()
    beam3 = beam.duplicate()
    beam0 = beam.duplicate()

#####    #####################################################################################################################
#
#    print(np.mean(beam1.x), np.mean(beam1.y), np.mean(beam1.z))
#    print(np.mean(beam1.vx), np.mean(beam1.vy), np.mean(beam1.vz))
#    print(np.mean(beam1.x**2)-(np.mean(beam1.x))**2, np.mean(beam1.y**2)-(np.mean(beam1.y))**2, np.mean(beam1.z**2)-(np.mean(beam1.z))**2)
#
#    [beam1, t] = oe1. intersection_with_optical_element(beam1)
#    oe1.output_direction_from_optical_element(beam1)
#
#
#    t = (q - beam1.y)/beam1.vy
#    #t = - beam1.x/beam1.vx
#    beam1.x = beam1.x + beam1.vx * t
#    beam1.y = beam1.y + beam1.vy * t
#    beam1.z = beam1.z + beam1.vz * t
#
#
#
#    beam1.plot_xz()
#    plt.title("oe1")
    #######  Before for        ##############################################################################################


    #beam.z = - beam.x.copy()
    #velocity = Vector(-np.mean(beam.x), -np.mean(beam.y),-np.mean(beam.z))
    #velocity.normalization()
    #beam.vx = beam.vx*0 + velocity.x
    #beam.vy = beam.vy*0 + velocity.y
    #beam.vz = beam.vz*0 + velocity.z



    #beam = Beam(25000)
    #yp = p / np.sqrt(1+2*np.tan(90*np.pi/180-theta)**2)
    #xp = - yp * np.cos(theta)
    #zp =  yp * np.cos(theta)

    #beam.set_point(xp, yp, zp)
    #beam.set_circular_spot(1e-3)
    #beam.plot_xz()

    #velocity = Vector(-xp, -yp, -zp)
    #velocity.normalization()
    #beam.vx = beam.vx * 0 + velocity.x
    #beam.vy = beam.vy * 0 + velocity.y
    #beam.vz = beam.vz * 0 + velocity.z


    #print("yp = %f, xp = %f, p = %f" %(yp, xp, p))



    [beam1, t1] = oe1.intersection_with_optical_element(beam1)
    [beam2, t2] = oe2.intersection_with_optical_element(beam2)


    print("t2 < t1: %d" %(np.size(np.where(t2<t1))))

    #t = (q-beam1.y)/beam1.vy
    #beam1.x = beam1.x + beam1.vx * t
    #beam1.x = beam1.x + beam1.vx * t
    #beam1.x = beam1.x + beam1.vx * t

    #beam1.plot_xz(0)
    #plt.title("beam1")

    #t = (q-beam2.y)/beam2.vy
    #beam2.x = beam2.x + beam2.vx * t
    #beam2.x = beam2.x + beam2.vx * t
    #beam2.x = beam2.x + beam2.vx * t

    #beam2.plot_xz(0)
    #plt.title("beam2")
    #plt.show()


    print("times")
    print(t1, t2, t1-t2)


    # beam2.plot_xz()

    [beam3, t3] = screen.intersection_with_optical_element(beam3)
    print("times")
    print(t1, t2, t3)

    # print(t1, t1[0])
    # print(t2, t2[0])
    # print(t3, t3[0])
    # print(beam.x[0], beam.y[0], beam.z[0])
    # print(beam.vx[0], beam.vy[0], beam.vz[0])

    beam1.plot_xz(0)
    plt.title("beam1 before")
    beam1.plot_yx(0)
    plt.title("beam1 before")

    #beam2.plot_xz(0)
    #plt.title("beam2 before")
    #beam2.plot_yx(0)
    #plt.title("beam2 before")

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

    print(beam1.flag)

    beam1.plot_good_xz(0)
    plt.title("beam1 after")
    beam1.plot_good_yx(0)
    plt.title("beam1 after")

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

    #beam2.plot_good_xz(0)
    #plt.title("beam2 after")
    #beam2.plot_good_yx(0)
    #plt.title("beam2 after")

    ######first iteration##################################################################################################

    indices1 = np.where(beam1.flag < 0)
    indices2 = np.where(beam2.flag < 0)

    # print("max(t1) = %f, max(t2) = %f" % (np.max(t1), np.max(t2)))

    maxim = max(abs(np.max(t1)), abs(np.max(t2)))

    # print("\nhello world")

    # print("Mean time: t1 = %f, t2 = %f" % (np.mean(t1), np.mean(t2)))

    t1[indices1] = t1[indices1]*0 + 1e12
    t2[indices2] = t2[indices2]*0 + 1e12

    t = np.minimum(t1, t2)
    origin = np.ones(beam1.N)

    indices = np.where(t2 < t1)
    origin[indices] += 1

    indices = np.where(t == 1e12)
    origin[indices] = 3

    beam3.flag += -1
    beam3.flag[indices] = 0

    indices = np.where(beam3.flag >= 0)

    # print("beam3 good rays = %d, bad rays = %d" % (np.size(indices), beam3.N - np.size(indices)))

    t[indices] = t3[indices]

    print("Rays on oe1 = %f" % (np.size(np.where(origin == 1))))
    print("Rays on oe2 = %f" % (np.size(np.where(origin == 2))))
    print("Rays on screen (No reflection)= %f" % (np.size(np.where(origin == 3))))

    # print(origin)

    # beam3.plot_xz()

    #####3 good beam####################################################################################################

    indices1 = np.where(origin == 1)
    beam01 = beam1.part_of_beam(indices1)
    indices1 = np.where(origin == 2)
    beam02 = beam2.part_of_beam(indices1)
    indices1 = np.where(origin == 3)
    beam03 = beam3.part_of_beam(indices1)

    # print("The total size of the three beam is: %f + %f + %f = %f" % (beam01.N, beam02.N, beam03.N, beam01.N + beam02.N + beam03.N))


    #######  Starting the for cicle   ###################################################################################

    beam1_list = [beam01.duplicate(), Beam(), Beam(), Beam(), Beam()]
    beam2_list = [beam02.duplicate(), Beam(), Beam(), Beam(), Beam()]
    beam3_list = [beam03.duplicate(), Beam(), Beam(), Beam(), Beam()]

    print("Rays, at the first step, arriving on\n-oe1 = %d\n-oe2 = %d\n-screen = %d" %(beam01.N, beam02.N, beam03.N))



    #for i in range(0, 2):

    ##### oe1 beam

    i = 0


    oe1.output_direction_from_optical_element(beam1_list[i])
    beam1_list[i].flag *= 0

    print("Ray on oe1 = %d"  %(beam1_list[i].N))

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


    #beam2_list[i+1].plot_xz(0)
    #beam2_list[i+1].plot_yx(0)
    #beam2_list[i+1].plot_zy(0)


    maxim = 2 * max(max(abs(t2)), max(abs(t3)))

    indices = np.where(beam2_list[i + 1].flag < 0)
    t2[indices] += 2 * maxim

    t = np.minimum(t2, t3)

    indices = np.where(t3 < t2)
    origin[indices] = 3

    # print("Now the value of i is %d" %(i))
    beam03 = beam3_list[i + 1].part_of_beam(indices)


    print("One reflection (FHM) = %d"  %beam03.N)

    indices = np.where(origin == 2)
    beam2_list[i + 1] = beam2_list[i + 1].part_of_beam(indices)

    ####oe2 beam

    oe2.output_direction_from_optical_element(beam2_list[i])
    beam2_list[i].flag *= 0


    print("Ray on oe2 = %d"  %(beam2_list[i].N))

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

    #beam1_list[i+1].plot_xz(0)
    #beam1_list[i+1].plot_yx(0)
    #beam1_list[i+1].plot_zy(0)
    #plt.show()



    #print(i)
    #print(beam1_list[i+1].flag)
    #print(t1, t3)


    maxim = 2 * max(max(abs(t1)), max(abs(t3)))



    indices = np.where(beam1_list[i + 1].flag < 0)
    t1[indices] += 2 * maxim

    t = t1.copy()
    t[indices] = t3[indices]

    # indices = np.where(t3 < t1)

    origin02[indices] += 2

    indices = np.where(origin02 == 3)

    beam003 = beam3_list[i + 1].part_of_beam(indices)


    print("One reflection (FVM) = %d"  %beam003.N)

    # print("Now the value of i is %d" %(i))


    beam3_list[i + 1] = beam03.merge(beam003)
    beam3_list[i + 1] = beam03.merge(beam003)




    # beam3_list[i + 1] = beam003.duplicate()
    # beam3_list[i+1].plot_xz()

    indices = np.where(origin02 == 1)
    beam1_list[i + 1] = beam1_list[i + 1].part_of_beam(indices)

    #beam3_list[i+1].plot_xz()
    #plt.title("pro")


######## last iteration ################################################################################################


    i = 1

    oe1.output_direction_from_optical_element(beam1_list[i])
    beam1_list[i].flag *= 0

    print("Ray on oe1 = %d" % (beam1_list[i].N))

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

    # print("Now the value of i is %d" %(i))
    beam03 = beam3_list[i + 1].part_of_beam(indices)

    indices = np.where(origin == 2)
    beam2_list[i + 1] = beam2_list[i + 1].part_of_beam(indices)

    ####oe2 beam

    oe2.output_direction_from_optical_element(beam2_list[i])
    beam2_list[i].flag *= 0

    print("Ray on oe2 = %d" % (beam2_list[i].N))

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

    # indices = np.where(t3 < t1)

    origin02[indices] += 2

    indices = np.where(origin02 == 3)

    beam003 = beam3_list[i + 1].part_of_beam(indices)

    # print("Now the value of i is %d" %(i))
    beam3_list[i + 1] = beam03.merge(beam003)

    # beam3_list[i + 1] = beam003.duplicate()
    # beam3_list[i+1].plot_xz()

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

    beam3_list[2].histogram()

    beam3_list[2].plot_xz()

    print("No reflection: %d\nOne reflection: %d\nTwo reflection: %d"  %(beam3_list[0].N, beam3_list[1].N, beam3_list[2].N))


    # beam003.plot_xz()




plt.show()


