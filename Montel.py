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




if main == "__main__":

    beam = Beam()
    beam.set_flat_divergence(dx=0.1, dz=0.1)
    #beam.set_divergences_collimated()

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

    position.rotation(-(np.pi / 2 - oe2.theta), "z")
    velocity.rotation(-(np.pi / 2 - oe2.theta), "z")
    position.rotation(-(np.pi / 2 - oe2.theta), "x")
    velocity.rotation(-(np.pi / 2 - oe2.theta), "x")

    [beam.x, beam.y, beam.z] = [position.x, position.y, position.z]
    [beam.vx, beam.vy, beam.vz] = [velocity.x, velocity.y, velocity.z]


########  translation    ################################################################################################

    vector_point = Vector(0, oe2.p, 0)

    vector_point.rotation(-(np.pi / 2 - oe2.theta), "z")
    vector_point.rotation(-(np.pi / 2 - oe2.theta), "x")


    beam.x = beam.x - vector_point.x
    beam.y = beam.y - vector_point.y
    beam.z = beam.z - vector_point.z

########  others    ####################################################################################################

    [beam, t] = oe2.intersection_with_optical_element(beam)
    oe2.output_direction_from_optical_element(beam)


######  translation to the screen   ####################################################################################
#
#    oe2.translation_to_the_screen(beam)
#
######  intersection to the screen   ###################################################################################

    t = (q-beam.y)/beam.vy
    beam.x = beam.x + beam.vx * t
    beam.y = beam.y + beam.vy * t
    beam.z = beam.z + beam.vz * t

#########################################################################################################################


    #beam.plot_xz(0)
    #plt.title(case)


    print("mean(beam.x)=%f, mean(beam.y)=%f, mean(beam.z)=%f" %(np.mean(beam.x), np.mean(beam.y), np.mean(beam.z)))


    #plt.show()


if main == "__main__2__":


    beam = Beam(25000)
    beam.set_flat_divergence(dx=0.01, dz=0.01)
    beam.set_divergences_collimated()

    shadow_beam = shadow_source()
    beam=Beam()
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
    theta = 88. * np.pi / 180
    beta = 90 * np.pi / 180
    alpha = 87.*np.pi/180


    xmax =  0.
    xmin = -0.4
    ymax =  0.4
    ymin = -0.4
    zmax =  0.4
    zmin =  0.

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


    print("\n")
    print(oe1.info())
    print("\n")
    print(oe2.info())
    print("\n")

    ccc = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0., -q])

    screen = Optical_element.initialize_as_surface_conic_from_coefficients(ccc)
    screen.set_parameters(p, q, 0., 0., "Surface conical mirror")



    #oe2 = Optical_element.initialize_as_surface_conic_ellipsoid_from_focal_distances(p=p, q=q, theta=theta, alpha=0., cylindrical=1)

    #beam.plot_xpzp()

##########   rotation     ##############################################################################################

    position = Vector(beam.x, beam.y, beam.z)
    velocity = Vector(beam.vx, beam.vy, beam.vz)

    position.rotation(-(np.pi / 2 - oe2.theta), "x")
    velocity.rotation(-(np.pi / 2 - oe2.theta), "x")
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


    beam.x = -beam.z.copy()
    beam.vx = np.sqrt(1-beam.vy**2-beam.vz**2)

    print("beam.x = %f, beam.y = %f, beam.z = %f" % (np.mean(beam.x), np.mean(beam.y), np.mean(beam.z)))
    print("beam.vx = %f, beam.vy = %f, beam.vz = %f" % (np.mean(beam.vx), np.mean(beam.vy), np.mean(beam.vz)))

    print("theta angle = %f"  %((np.arctan(np.mean(beam.z/beam.y))*180/np.pi)))
    print("fi angle = %f"  %((np.arctan(np.mean(beam.x/beam.y))*180/np.pi)))

###### beam separation   ###############################################################################################

    beam1 = beam.duplicate()
    beam2 = beam.duplicate()
    beam3 = beam.duplicate()
    beam0 = beam.duplicate()


#########################################################################################################################
#
#    print("beam.x = %f, beam.y = %f, beam.z = %f" %(np.mean(beam.x), np.mean(beam.y), np.mean(beam.z)))
#    print("beam.vx = %f, beam.vy = %f, beam.vz = %f" %(np.mean(beam.vx), np.mean(beam.vy), np.mean(beam.vz)))
#
#    [beam, t] = oe2.intersection_with_optical_element(beam)
#
#    print("After the intersection\nbeam.x = %f, beam.y = %f, beam.z = %f" %(np.mean(beam.x), np.mean(beam.y), np.mean(beam.z)))
#    print("beam.vx = %f, beam.vy = %f, beam.vz = %f" %(np.mean(beam.vx), np.mean(beam.vy), np.mean(beam.vz)))
#
#    oe2.output_direction_from_optical_element(beam)
#
#    #position = Vector(beam.x, beam.y, beam.z)
#    #velocity = Vector(beam.vx, beam.vy, beam.vz)
#    #position.rotation(-(np.pi / 2 - oe2.theta), "z")
#    #velocity.rotation(-(np.pi / 2 - oe2.theta), "z")
#    #[beam.x, beam.y, beam.z] = [position.x, position.y, position.z]
#    #[beam.vx, beam.vy, beam.vz] = [velocity.x, velocity.y, velocity.z]
#
#    #oe2.translation_to_the_screen(beam)
#    #oe2.intersection_with_the_screen(beam)
#
#    t = (1. - beam.y) / beam.vy
#    print(np.mean(t))
#    beam.x = beam.x + beam.vx * t
#    beam.y = beam.y + beam.vy * t
#    beam.z = beam.z + beam.vz * t
#
#    print("When y = 1\nbeam.x = %f, beam.y = %f, beam.z = %f" %(np.mean(beam.x), np.mean(beam.y), np.mean(beam.z)))
#    print("beam.vx = %f, beam.vy = %f, beam.vz = %f" %(np.mean(beam.vx), np.mean(beam.vy), np.mean(beam.vz)))
#
#    screen.intersection_with_optical_element(beam)
#
#    #t = (q-beam.y)/beam.vy
#    #print(np.mean(t))
#    #beam.x = beam.x + beam.vx * t
#    #beam.y = beam.y + beam.vy * t
#    #beam.z = beam.z + beam.vz * t
#
#
#    beam.plot_xz()
#
#
#######  time comparasion  ##############################################################################################



    [beam1, t1] = oe1.intersection_with_optical_element(beam1)
    [beam2, t2] = oe2.intersection_with_optical_element(beam2)
    [beam3, t3] = screen.intersection_with_optical_element(beam3)


    print(t1, t1[0])
    print(t2, t2[0])
    print(t3, t3[0])
    print(beam.x[0], beam.y[0], beam.z[0])
    print(beam.vx[0], beam.vy[0], beam.vz[0])


    #beam1.plot_xz(0)
    #plt.title("beam1 before")
    #beam1.plot_yx(0)
    #plt.title("beam1 before")


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


    #beam1.plot_good_xz(0)
    #plt.title("beam1 after")
    #beam1.plot_good_yx(0)
    #plt.title("beam1 after")

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

    indices1 = np.where(beam1.flag<0)
    indices2 = np.where(beam2.flag<0)

    print("max(t1) = %f, max(t2) = %f" %(np.max(t1), np.max(t2)))

    maxim = max(abs(np.max(t1)), abs(np.max(t2)))

    print("\nhello world")

    print("Mean time: t1 = %f, t2 = %f" %(np.mean(t1), np.mean(t2)))

    t1[indices1] = 1e12 * np.ones(np.size(indices1))
    t2[indices2] = 1e12 * np.ones(np.size(indices1))


    t = np.minimum(t1,t2)
    origin = np.ones(beam1.N)

    indices = np.where(t2<t1)
    origin[indices] += 1




    indices = np.where(t==1e12)
    origin[indices] = 3

    beam3.flag += -1
    beam3.flag[indices] = 0

    indices = np.where(beam3.flag>=0)

    print("beam3 good rays = %d, bad rays = %d" %(np.size(indices), beam3.N-np.size(indices)))

    t[indices] = t3[indices]


    print("Rays on oe1 = %f" %(np.size(np.where(origin==1))))
    print("Rays on oe2 = %f" %(np.size(np.where(origin==2))))
    print("Rays on screen (No reflection)= %f" %(np.size(np.where(origin==3))))

    print(origin)

    #####3 good beam####################################################################################################

    indices1 = np.where(origin==1)
    beam01 = beam1.part_of_beam(indices1)
    indices1 = np.where(origin==2)
    beam02 = beam2.part_of_beam(indices1)
    indices1 = np.where(origin==3)
    beam03 = beam3.part_of_beam(indices1)

    print("The total size of the three beam is: %f + %f + %f = %f"  %(beam01.N, beam02.N, beam03.N, beam01.N+beam02.N+beam03.N))


    ### Working with beam01  ############################################################################################

    oe1.output_direction_from_optical_element(beam01)
    beam01.flag *= 0

    print("Rays that have reaced the oe1: %f"  %(beam01.N))

    beam2 = beam01.duplicate()
    beam3 = beam01.duplicate()

    [beam2, t2] = oe2.intersection_with_optical_element(beam2)
    [beam3, t3] = screen.intersection_with_optical_element(beam3)



    indices = np.where(beam2.x > xmax)
    beam2.flag[indices] = -1
    indices = np.where(beam2.x < xmin)
    beam2.flag[indices] = -1

    indices = np.where(beam2.z > ymax)
    beam2.flag[indices] = -1
    indices = np.where(beam2.z < ymin)
    beam2.flag[indices] = -1

    indices = np.where(beam2.y > zmax)
    beam2.flag[indices] = -1
    indices = np.where(beam2.y < zmin)
    beam2.flag[indices] = -1

    indices = np.where(beam2.flag<0)
    print("Bad rays that reach oe2 from oe1 = %f"  %(np.size(indices)))
    maxim = np.max(abs(t2))

    #print(t2, t3)

    #beam2.plot_zy(0)
    #beam2.plot_good_zy(0)


    t = t2.copy()

    t[indices] = t3[indices]

    origin01 = 2*np.ones(beam01.N)
    origin01[indices] += 1

    print("The number of ray that reach screen (only one reflection VFM) = %f\n\n"  %(np.size(indices)))


    indices1 = np.where(origin01==3)
    beam003 = beam3.part_of_beam(indices1)
    indices1 = np.where(origin01==2)
    beam002 = beam2.part_of_beam(indices1)

    #plt.figure()
    #plt.plot(beam03.x, beam03.z, 'ro')
    #plt.plot(beam003.x, beam003.z, 'bo')
    #plt.axis('equal')

    ### Working with beam02  ############################################################################################

    oe2.output_direction_from_optical_element(beam02)


    print("Rays that have reaced the oe2: %f"  %(beam02.N))

    #beam02.plot_xz()
    #beam02.plot_zy()
    #plt.show()

    beam02.flag *= 0

    beam1 = beam02.duplicate()
    beam3 = beam02.duplicate()

    [beam1, t1] = oe1.intersection_with_optical_element(beam1)
    [beam3, t3] = screen.intersection_with_optical_element(beam3)

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

    indices = np.where(beam1.flag<0)
    print("Bad rays that reach oe2 from oe1 = %f"  %(np.size(indices)))
    maxim = np.max(abs(t2))

    #beam1.plot_xz()
    #plt.title("beam1 with all rays")
    #beam1.plot_yx()
    #plt.title("beam1 with all rays")


    #beam1.plot_good_xz()
    #plt.title("beam1 with good rays")
    #beam1.plot_good_yx()
    #plt.title("beam1 with good rays")


    t = t1.copy()
    print(np.size(indices))

    t[indices] = t3[indices]


    print("The number of ray that reach screen (only one reflection HFM) = %f"  %(np.size(indices)))

    origin02 = np.ones(beam02.N)
    origin02[indices] += 2


    indices1 = np.where(origin02==3)
    beam0003 = beam3.part_of_beam(indices1)
    indices1 = np.where(origin02==1)
    beam0001 = beam1.part_of_beam(indices1)

    print(np.size(beam0003.x))

    #plt.figure()
    #plt.plot(beam03.x, beam03.z, 'ro')
    #plt.plot(beam003.x, beam003.z, 'bo')
    #plt.plot(beam0003.x, beam0003.z, 'yo')
    #plt.axis('equal')




########################################################################################################################

    oe1.output_direction_from_optical_element(beam0001)
    oe2.output_direction_from_optical_element(beam002)

    beam002.flag *= 0.
    beam0001.flag *= 0.

    screen.intersection_with_optical_element(beam0001)
    screen.intersection_with_optical_element(beam002)

    plt.figure()
    plt.plot(beam03.x, beam03.z, 'ro')
    plt.plot(beam003.x, beam003.z, 'bo')
    plt.plot(beam0003.x, beam0003.z, 'yo')
    plt.plot(beam002.x, beam002.z, 'go')
    plt.plot(beam0001.x, beam0001.z, 'go')
    plt.xlabel('x axis')
    plt.ylabel('z axis')
    plt.axis('equal')

    print("\n\nNo-reflection red\nLength = %f, Height = %f, area = %f, number of rays = %f"  %(max(beam03.x)-min(beam03.x), max(beam03.z)-min(beam03.z),(max(beam03.x)-min(beam03.x))*(max(beam03.z)-min(beam03.z)), beam03.N))
    print("One-reflecction (HFM) blue\nLength = %f, Height = %f, area = %f, number of rays = %f"  %(max(beam003.x)-min(beam003.x), max(beam003.z)-min(beam003.z),(max(beam003.x)-min(beam003.x))*(max(beam003.z)-min(beam003.z)), beam003.N))
    print("One-reflecction (VFM) yellow\nLength = %f, Height = %f, area = %f, number of rays = %f"  %(max(beam0003.x)-min(beam0003.x), max(beam0003.z)-min(beam0003.z),(max(beam0003.x)-min(beam0003.x))*(max(beam0003.z)-min(beam0003.z)), beam0003.N))

    gxmax = max(max(beam0001.x),max(beam002.x))
    gxmin = min(min(beam0001.x),min(beam002.x))
    gzmax = max(max(beam0001.z),max(beam002.z))
    gzmin = min(min(beam0001.z),min(beam002.z))
    beam_f = beam0001.merge(beam002)

    print("Double-reflection green\nLength = %f, Height = %f, area = %f, number of rays = %f"  %(gxmax-gxmin, gzmax-gzmin, (gxmax-gxmin)*(gzmax*gzmin), beam_f.N))
    print("Total rays = %f\n\n" %(beam03.N+beam003.N+beam0003.N+beam_f.N))


    plt.figure()
    plt.plot(beam_f.x, beam_f.z, 'go')
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    #beam_f.histogram()


    plt.show()