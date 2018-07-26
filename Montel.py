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

    #print("oe1 coefficients")
    #print(oe1.ccc_object.get_coefficients())
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


    #print(oe2.ccc_object.get_coefficients())

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


    #print("mean(beam.x)=%f, mean(beam.y)=%f, mean(beam.z)=%f" %(np.mean(beam.x), np.mean(beam.y), np.mean(beam.z)))


    #plt.show()


if main == "__main__2__":

    fwhmx = np.ones(40)
    fwhmz = np.ones(40)

    qqq = np.ones(40)


    for iii in range (0, 40):

        beam = Beam(25000)
        beam.set_flat_divergence(dx=0.001, dz=0.001)
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
        beta = (90.+0.) * np.pi / 180
        alpha = 87.*np.pi/180


        xmax =  0.
        xmin = -0.4
        ymax =  0.4
        ymin = -0.4
        zmax =  0.4*abs(np.cos(np.pi/2-beta))
        #print("zmax = %f" %(zmax))
        #zmax = 0.4
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

        ccc2 = np.array([c2 * b ** 2, c1, c2 * a ** 2, -c4 * b, c4 * a, -2 * c2 * a * b, -c8 * b, 0., c8 * a, 0.])

        oe2 = Optical_element.initialize_as_surface_conic_from_coefficients(ccc2)
        oe2.p = p
        oe2.q = q
        oe2.theta = theta
        oe2.alpha = 0.
        oe2.type = "Surface conical mirror"


        #print("\n")
        #print(oe1.info())
        #print("\n")
        #print(oe2.info())
        #print("\n")

        #ccc = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0., -(q-0.055)])
        mimim = min (qqq)
        jjj = np.where(qqq==mimim)
        #ccc = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0., -(q-0.2+0.4*0.1*iii/40)])
        #qqq[iii] = q-0.2+0.4*0.1*iii/40

        #ccc = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0., -(q - 0.4 + 0.8 * iii/40)])
        #qqq[iii] = q - 5 + 10* iii/40

        ccc = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0., -(q - 0.4 + 0.8 * iii / 40)])
        qqq[iii] = q - 5 + 10 * iii / 40


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

        #print("beam.x = %f, beam.y = %f, beam.z = %f" % (np.mean(beam.x), np.mean(beam.y), np.mean(beam.z)))
        #print("beam.vx = %f, beam.vy = %f, beam.vz = %f" % (np.mean(beam.vx), np.mean(beam.vy), np.mean(beam.vz)))

        #print("theta angle = %f"  %((np.arctan(np.mean(beam.z/beam.y))*180/np.pi)))
        #print("fi angle = %f"  %((np.arctan(np.mean(beam.x/beam.y))*180/np.pi)))

    ###### beam separation   ###############################################################################################

        beam1 = beam.duplicate()
        beam2 = beam.duplicate()
        beam3 = beam.duplicate()
        beam0 = beam.duplicate()

        print(oe1.info())
        print(oe2.info())


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
    #######  Before for        ##############################################################################################


        beam1 = beam.duplicate()
        beam2 = beam.duplicate()
        beam3 = beam.duplicate()

        [beam1, t1] = oe1.intersection_with_optical_element(beam1)
        [beam2, t2] = oe2.intersection_with_optical_element(beam2)

        #beam2.plot_xz()


        [beam3, t3] = screen.intersection_with_optical_element(beam3)

        #print(t1, t1[0])
        #print(t2, t2[0])
        #print(t3, t3[0])
        #print(beam.x[0], beam.y[0], beam.z[0])
        #print(beam.vx[0], beam.vy[0], beam.vz[0])

        # beam1.plot_xz(0)
        # plt.title("beam1 before")
        # beam1.plot_yx(0)
        # plt.title("beam1 before")

        # beam2.plot_xz(0)
        # plt.title("beam2 before")
        # beam2.plot_yx(0)
        # plt.title("beam2 before")

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

        # beam1.plot_good_xz(0)
        # plt.title("beam1 after")
        # beam1.plot_good_yx(0)
        # plt.title("beam1 after")

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

        # beam2.plot_good_xz(0)
        # plt.title("beam2 after")
        # beam2.plot_good_yx(0)
        # plt.title("beam2 after")

        ######first iteration##################################################################################################

        indices1 = np.where(beam1.flag < 0)
        indices2 = np.where(beam2.flag < 0)

        #print("max(t1) = %f, max(t2) = %f" % (np.max(t1), np.max(t2)))

        maxim = max(abs(np.max(t1)), abs(np.max(t2)))

        #print("\nhello world")

        #print("Mean time: t1 = %f, t2 = %f" % (np.mean(t1), np.mean(t2)))

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

        #print("beam3 good rays = %d, bad rays = %d" % (np.size(indices), beam3.N - np.size(indices)))

        t[indices] = t3[indices]

        #print("Rays on oe1 = %f" % (np.size(np.where(origin == 1))))
        #print("Rays on oe2 = %f" % (np.size(np.where(origin == 2))))
        #print("Rays on screen (No reflection)= %f" % (np.size(np.where(origin == 3))))

        #print(origin)

        #beam3.plot_xz()

        #####3 good beam####################################################################################################

        indices1 = np.where(origin == 1)
        beam01 = beam1.part_of_beam(indices1)
        indices1 = np.where(origin == 2)
        beam02 = beam2.part_of_beam(indices1)
        indices1 = np.where(origin == 3)
        beam03 = beam3.part_of_beam(indices1)

        #print("The total size of the three beam is: %f + %f + %f = %f" % (beam01.N, beam02.N, beam03.N, beam01.N + beam02.N + beam03.N))


       #######  Starting the for cicle   ###################################################################################

        beam1_list = [beam01.duplicate(), Beam(), Beam(), Beam(), Beam()]
        beam2_list = [beam02.duplicate(), Beam(), Beam(), Beam(), Beam()]
        beam3_list = [beam03.duplicate(), Beam(), Beam(), Beam(), Beam()]





        for i in range (0, 2):

            ##### oe1 beam

           oe1.output_direction_from_optical_element(beam1_list[i])
           beam1_list[i].flag *= 0

           beam2_list[i+1] = beam1_list[i].duplicate()
           beam3_list[i+1] = beam1_list[i].duplicate()

           [beam2_list[i+1], t2] = oe2.intersection_with_optical_element(beam2_list[i+1])
           [beam3_list[i+1], t3] = screen.intersection_with_optical_element(beam3_list[i+1])

           origin = 2*np.ones(beam1_list[i].N)

           indices = np.where(beam2_list[i+1].x > xmax)
           beam2_list[i+1].flag[indices] = -1
           indices = np.where(beam2_list[i+1].x < xmin)
           beam2_list[i+1].flag[indices] = -1

           indices = np.where(beam2_list[i+1].z > ymax)
           beam2_list[i+1].flag[indices] = -1
           indices = np.where(beam2_list[i+1].z < ymin)
           beam2_list[i+1].flag[indices] = -1

           indices = np.where(beam2_list[i+1].y > zmax)
           beam2_list[i+1].flag[indices] = -1
           indices = np.where(beam2_list[i+1].y < zmin)
           beam2_list[i+1].flag[indices] = -1

           maxim = 2*max(max(abs(t2)), max(abs(t3)))

           indices = np.where(beam2_list[i+1].flag <0)
           t2[indices] += 2*maxim

           t = np.minimum(t2,t3)

           indices = np.where(t3<t2)
           origin[indices] = 3

           #print("Now the value of i is %d" %(i))
           beam03 = beam3_list[i+1].part_of_beam(indices)

           indices = np.where(origin==2)
           beam2_list[i+1] = beam2_list[i+1].part_of_beam(indices)


           ####oe2 beam


           oe2.output_direction_from_optical_element(beam2_list[i])
           beam2_list[i].flag *= 0

           beam1_list[i + 1] = beam2_list[i].duplicate()
           beam3_list[i + 1] = beam2_list[i].duplicate()


           [beam1_list[i + 1], t1] = oe1.intersection_with_optical_element(beam1_list[i + 1])
           [beam3_list[i + 1], t3] = screen.intersection_with_optical_element(beam3_list[i + 1])

           origin02 =  np.ones(beam2_list[i].N)

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

           #indices = np.where(t3 < t1)

           origin02[indices] += 2

           indices = np.where(origin02==3)

           beam003 = beam3_list[i + 1].part_of_beam(indices)


           #print("Now the value of i is %d" %(i))
           beam3_list[i + 1] = beam03.merge(beam003)

           #beam3_list[i + 1] = beam003.duplicate()
           #beam3_list[i+1].plot_xz()

           indices = np.where(origin02 == 1)
           beam1_list[i + 1] = beam1_list[i + 1].part_of_beam(indices)






        #oe1.output_direction_from_optical_element(beam1_list[2])
        #oe2.output_direction_from_optical_element(beam2_list[2])
        #print(beam1_list[2].N, beam2_list[2].N, beam2_list[2].x)

        #beam1_list[2].flag = np.zeros(beam1_list[2].N)
        #beam2_list[2].flag = np.zeros(beam2_list[2].N)

        #print(beam1_list[2].flag)
        #print(beam2_list[2].flag)


        #screen.intersection_with_optical_element(beam1_list[2])
        #screen.intersection_with_optical_element(beam2_list[2])

        #beam1_list[2] = beam1_list[2].merge(beam2_list[2])



        #plt.figure()
        #plt.plot(beam3_list[0].x, beam3_list[0].z, 'ro')
        #plt.plot(beam3_list[1].x, beam3_list[1].z, 'bo')
        #plt.plot(beam3_list[2].x, beam3_list[2].z, 'go')
        ##plt.plot(beam1_list[3].x, beam1_list[3].z, 'yo')
        ##plt.plot(beam1_list[4].x, beam1_list[4].z, 'ko')
        #plt.xlabel('x axis')
        #plt.ylabel('z axis')
        #plt.axis('equal')

        #beam3_list[2].histogram()

        #beam3_list[2].plot_xz()

        #beam003.plot_xz()


        #print("FWHM(x) = %f"  %(max(beam3_list[2].x)-min(beam3_list[2].x)))
        y = np.zeros(110)
        y1 = np.zeros(110)
        x = np.ones(110)
        z = np.ones(110)


        for i in range (0, 110):


            #exchange between beam003 and beam3_list[2]

            mx = max (beam03.x) + 0.00001
            mn = min (beam03.x) - 0.00001

            mzx = max(beam003.z) + 0.00001
            mzn = min(beam003.z) - 0.00001

            mv =  max(beam03.x)/2
            mzv = max(beam003.z)/2

            eps = (mx-mn) / 110
            epsz = (mzx-mzn) / 110

            indices =  np.where(beam03.x < mn + i*eps)
            indicesz = np.where(beam003.z < mn + i*eps)

            #print(np.size(indices))

            if i ==0:
                y[i] = np.size(indices)
                y1[i] = np.size(indicesz)
            else:
                y[i] = np.size(indices)  - np.sum(y)
                y1[i] = np.size(indices1)  - np.sum(y1)

            x[i] = mn + i * eps
            z[i] = mzn + i * epsz

        #plt.figure()
        #plt.plot(x, y)

        xminfwhm = 0
        xmaxfwhm = 0
        zminfwhm = 0
        zmaxfwhm = 0

        for i in range (0, 110):

            mx2 =  max(y)/2
            mz2 = max(y1)/2

            if y[i] > mx2:
                if xminfwhm == 0:
                    xminfwhm = i
            elif y[i] < mx2:
                if y[i] > mx2*0.8:
                    xmaxfwhm= i

            if y1[i] > mz2:
                if zminfwhm == 0:
                    zminfwhm = i
            elif y1[i] < mz2:
                if y1[i] > mz2 * 0.8:
                    zmaxfwhm = i


        y2 = [mx2, mx2]
        x2 = [x[xminfwhm], x[xmaxfwhm]]

        #plt.figure()
        #plt.plot(x2, y2, 'ro')

        fwhmx[iii] = x[xmaxfwhm]-x[xminfwhm]
        fwhmz[iii] = z[xmaxfwhm]-z[xminfwhm]



    print(fwhmx)
    print(fwhmz)

    plt.figure()
    plt.plot(qqq, fwhmx, 'bo')
    plt.plot(qqq, fwhmz, 'r.')

    hx = min(fwhmx)
    hz = min(fwhmz)

    indices = np.where(fwhmz==hz)

    print(hx, hz)
    print(qqq[indices])

    beam = Beam(25000)
    beam.set_flat_divergence(dx=0.01, dz=0.01)
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

    p = 5.
    q = 15.
    theta = 88. * np.pi / 180
    beta = (90. + 0.) * np.pi / 180
    alpha = 87. * np.pi / 180

    xmax = 0.
    xmin = -0.4
    ymax = 0.4
    ymin = -0.4
    zmax = 0.4 * abs(np.cos(np.pi / 2 - beta))
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

    # print("\n")
    # print(oe1.info())
    # print("\n")
    # print(oe2.info())
    # print("\n")

    # ccc = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0., -(q-0.055)])
    mimim = min (fwhmx)
    jjj = np.where(fwhmx==mimim)
    qq = qqq[jjj]
    ccc = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0., -qq[0]])


    screen = Optical_element.initialize_as_surface_conic_from_coefficients(ccc)
    screen.set_parameters(p, q, 0., 0., "Surface conical mirror")

    # oe2 = Optical_element.initialize_as_surface_conic_ellipsoid_from_focal_distances(p=p, q=q, theta=theta, alpha=0., cylindrical=1)

    # beam.plot_xpzp()

#    ##########   rotation     ##############################################################################################
#
#    position = Vector(beam.x, beam.y, beam.z)
#    velocity = Vector(beam.vx, beam.vy, beam.vz)
#
#    position.rotation(-(np.pi / 2 - oe2.theta), "z")
#    velocity.rotation(-(np.pi / 2 - oe2.theta), "z")
#    position.rotation(-(np.pi / 2 - oe2.theta), "x")
#    velocity.rotation(-(np.pi / 2 - oe2.theta), "x")
#
#    [beam.x, beam.y, beam.z] = [position.x, position.y, position.z]
#    [beam.vx, beam.vy, beam.vz] = [velocity.x, velocity.y, velocity.z]
#
#    ########  translation    ###############################################################################################
#
#    vector_point = Vector(0, oe2.p, 0)
#
#    vector_point.rotation(-(np.pi / 2 - oe2.theta), "x")
#    vector_point.rotation(-(np.pi / 2 - oe2.theta), "z")
#
#    beam.x = beam.x - vector_point.x
#    beam.y = beam.y - vector_point.y
#    beam.z = beam.z - vector_point.z
#
#    beam.x = -beam.z.copy()
#    beam.vx = np.sqrt(1 - beam.vy ** 2 - beam.vz ** 2)
#
#    # print("beam.x = %f, beam.y = %f, beam.z = %f" % (np.mean(beam.x), np.mean(beam.y), np.mean(beam.z)))
#    # print("beam.vx = %f, beam.vy = %f, beam.vz = %f" % (np.mean(beam.vx), np.mean(beam.vy), np.mean(beam.vz)))
#
#    # print("theta angle = %f"  %((np.arctan(np.mean(beam.z/beam.y))*180/np.pi)))
#    # print("fi angle = %f"  %((np.arctan(np.mean(beam.x/beam.y))*180/np.pi)))
#
#    ###### beam separation   ###############################################################################################

    theta = np.pi/2 - theta

    vector = Vector(0., 1., 0.)
    vector.rotation(-theta, 'x')
    print(vector.info())

    print("theta' = %f, theta'' = %f" % (
    np.arctan(vector.x / vector.y) * 180 / np.pi, np.arctan(vector.z / vector.y) * 180 / np.pi))

    ny = -vector.z / np.sqrt(vector.y ** 2 + vector.z ** 2)
    nz = vector.y / np.sqrt(vector.y ** 2 + vector.z ** 2)

    n = Vector(0, ny, nz)

    vrot = vector.rodrigues_formula(n, -theta)
    vrot.normalization()

    print("theta' = %f, theta'' = %f" % (
    np.arctan(vrot.x / vrot.y) * 180 / np.pi, np.arctan(vrot.z / vrot.y) * 180 / np.pi))

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

    vector_point.rotation(-(np.pi / 2 - oe2.theta - 0 * np.pi / 4), "x")
    vector_point = vector_point.rodrigues_formula(n, -theta)
    vector_point.normalization()

    beam.x = beam.x - vector_point.x * p
    beam.y = beam.y - vector_point.y * p
    beam.z = beam.z - vector_point.z * p

    #########################################################################################################################

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
    #######  Before for        ##############################################################################################

    beam1 = beam.duplicate()
    beam2 = beam.duplicate()
    beam3 = beam.duplicate()

    [beam1, t1] = oe1.intersection_with_optical_element(beam1)
    [beam2, t2] = oe2.intersection_with_optical_element(beam2)

    # beam2.plot_xz()

    [beam3, t3] = screen.intersection_with_optical_element(beam3)

    # print(t1, t1[0])
    # print(t2, t2[0])
    # print(t3, t3[0])
    # print(beam.x[0], beam.y[0], beam.z[0])
    # print(beam.vx[0], beam.vy[0], beam.vz[0])

    # beam1.plot_xz(0)
    # plt.title("beam1 before")
    # beam1.plot_yx(0)
    # plt.title("beam1 before")

    # beam2.plot_xz(0)
    # plt.title("beam2 before")
    # beam2.plot_yx(0)
    # plt.title("beam2 before")

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

    # beam1.plot_good_xz(0)
    # plt.title("beam1 after")
    # beam1.plot_good_yx(0)
    # plt.title("beam1 after")

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

    # beam2.plot_good_xz(0)
    # plt.title("beam2 after")
    # beam2.plot_good_yx(0)
    # plt.title("beam2 after")

    ######first iteration##################################################################################################

    indices1 = np.where(beam1.flag < 0)
    indices2 = np.where(beam2.flag < 0)

    # print("max(t1) = %f, max(t2) = %f" % (np.max(t1), np.max(t2)))

    maxim = max(abs(np.max(t1)), abs(np.max(t2)))

    # print("\nhello world")

    # print("Mean time: t1 = %f, t2 = %f" % (np.mean(t1), np.mean(t2)))

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

    # print("beam3 good rays = %d, bad rays = %d" % (np.size(indices), beam3.N - np.size(indices)))

    t[indices] = t3[indices]

    # print("Rays on oe1 = %f" % (np.size(np.where(origin == 1))))
    # print("Rays on oe2 = %f" % (np.size(np.where(origin == 2))))
    # print("Rays on screen (No reflection)= %f" % (np.size(np.where(origin == 3))))

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

        # print("Now the value of i is %d" %(i))
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

    # oe1.output_direction_from_optical_element(beam1_list[2])
    # oe2.output_direction_from_optical_element(beam2_list[2])
    # print(beam1_list[2].N, beam2_list[2].N, beam2_list[2].x)

    # beam1_list[2].flag = np.zeros(beam1_list[2].N)
    # beam2_list[2].flag = np.zeros(beam2_list[2].N)

    # print(beam1_list[2].flag)
    # print(beam2_list[2].flag)

    # screen.intersection_with_optical_element(beam1_list[2])
    # screen.intersection_with_optical_element(beam2_list[2])

    # beam1_list[2] = beam1_list[2].merge(beam2_list[2])

    plt.figure()
    plt.plot(beam3_list[0].x, beam3_list[0].z, 'ro')
    plt.plot(beam3_list[1].x, beam3_list[1].z, 'bo')
    plt.plot(beam3_list[2].x, beam3_list[2].z, 'go')
    #plt.plot(beam1_list[3].x, beam1_list[3].z, 'yo')
    #plt.plot(beam1_list[4].x, beam1_list[4].z, 'ko')
    plt.xlabel('x axis')
    plt.ylabel('z axis')
    plt.axis('equal')

    beam3_list[2].histogram()

    beam3_list[2].plot_xz()

    # beam003.plot_xz()

    print(screen.info())

    print (xminfwhm, xmaxfwhm, xmaxfwhm-xminfwhm)


    plt.show()