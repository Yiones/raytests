from OpticalElement import Optical_element
from SurfaceConic import SurfaceConic
from Beam import Beam
from Vector import Vector
import matplotlib.pyplot as plt
import numpy as np
import Shadow
from Shape import  BoundaryRectangle
from CompoundOpticalElement import CompoundOpticalElement

main = "__main4__"
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



def time_comparision(beam1, elements, oe):

    origin = np.ones(beam1.N)
    tf = 1e35 * np.ones(beam1.N)

    for i  in range (0, len(elements)):


        beam = beam1.duplicate()
        [beam, t] = oe[i].intersection_with_optical_element(beam)

        indices = np.where(beam.flag < 0)
        t[indices] = 1e30

        print(indices)


        tf = np.minimum(t, tf)
        indices = np.where(t == tf)
        origin[indices] = elements[i]

    return origin

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



    beam.flag *= 0

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

    bound1 = BoundaryRectangle(xmax=xmax, xmin=xmin, ymax=ymax, ymin=ymin, zmax=zmax, zmin=zmin)
    oe1.bound = bound1


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

    bound2 = BoundaryRectangle(xmax=xmax, xmin=xmin, ymax=ymax, ymin=ymin, zmax=zmax, zmin=zmin)
    oe2.bound = bound2


    ccc = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0., -q])

    screen = Optical_element.initialize_as_surface_conic_from_coefficients(ccc)
    screen.set_parameters(p, q, 0., 0., "Surface conical mirror")




###########################################################################################################################

    print(beam.y)

    theta = np.pi/2 - theta


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

    print(beam.y)

    element = [1, 2, 3]
    oe = [oe1, oe2, screen]

    origin = time_comparision(beam, element, oe)

    print(origin)

    indices = np.where(origin==1)
    beam1 = beam.part_of_beam(indices)
    indices = np.where(origin==2)
    beam2 = beam.part_of_beam(indices)
    indices = np.where(origin==3)
    beam3 = beam.part_of_beam(indices)

    print(beam1.N, beam2.N, beam3.N)

    [beam3, t] = screen.intersection_with_optical_element(beam3)

    beam1_list = [beam1.duplicate(), Beam(), Beam()]
    beam2_list = [beam2.duplicate(), Beam(), Beam()]
    beam3_list = [beam3.duplicate(), Beam(), Beam()]





    for i in range (0, 2):

        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Iteration number %d" %i)
        print(beam1_list[i].N, beam2_list[i].N)

        [beam1_list[i], t] = oe1.intersection_with_optical_element(beam1_list[i])
        oe1.output_direction_from_optical_element(beam1_list[i])

        origin = time_comparision(beam1_list[i], [2, 3], [oe2, screen])
        indices = np.where(origin==2)
        beam2_list[i+1] = beam1_list[i].part_of_beam(indices)
        indices = np.where(origin==3)
        beam03 = beam1_list[i].part_of_beam(indices)

        [beam2_list[i], t] = oe2.intersection_with_optical_element(beam2_list[i])
        oe2.output_direction_from_optical_element(beam2_list[i])

        origin = time_comparision(beam2_list[i], [1, 3], [oe1, screen])
        indices = np.where(origin == 1)
        beam1_list[i+1] = beam2_list[i].part_of_beam(indices)
        indices = np.where(origin == 3)
        beam003 = beam2_list[i].part_of_beam(indices)


        beam3_list[i+1] = beam03.merge(beam003)
        [beam3_list[i+1], t] = screen.intersection_with_optical_element(beam3_list[i+1])



    plt.figure()
    plt.plot(beam3_list[0].x, beam3_list[0].z, 'ro')
    plt.plot(beam3_list[1].x, beam3_list[1].z, 'bo')
    plt.plot(beam3_list[2].x, beam3_list[2].z, 'go')
    plt.xlabel('x axis')
    plt.ylabel('z axis')
    plt.axis('equal')





    plt.show()


if main == "__main2__":



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

    beam = Beam(25000)
    beam.set_flat_divergence(25*1e-6, 25*1e-6)
    beam.set_rectangular_spot(xmax=25*1e-6, xmin=-25*1e-6, zmax=5*1e-6, zmin=-5*1e-6)
    beam.set_gaussian_divergence(25*1e-6, 25*1e-6)
    beam.set_divergences_collimated()

    beam.flag *= 0

    p = 5.
    q = 15.
    theta = 88. * np.pi /180.

    xmax = 0.
    xmin = -0.3
    ymax =  0.3
    ymin = -0.3
    zmax =  0.3
    zmin = 0.

    bound1 = BoundaryRectangle(xmax, xmin, ymax, ymin, zmax, zmin)
    bound2 = BoundaryRectangle(xmax, xmin, ymax, ymin, zmax, zmin)

    Nn = 25
    qq = np.ones(Nn)
    dx = np.ones(Nn)




    for i in range (0, Nn):
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Iteration %d" %i)
        beam1 = beam.duplicate()
        qq[i] = q - 1. + 2. * i / Nn
        print(qq[i])

        montel = CompoundOpticalElement.initialize_as_montel_parabolic(p=p, q=q, theta=theta, bound1=bound1, bound2=bound2, distance_of_the_screen=qq[i])
        beam03 = montel.trace_montel(beam1)

        if beam03[2].N != 0:
            dx[i] = max(beam03[2].x) - min(beam03[2].x)
        else:
            dx[i] = 100


    plt.figure()
    plt.plot(qq, dx)

    min_of_qq = min(dx)
    indice = np.where(dx == min_of_qq)
    dos = qq[indice]
    print(min_of_qq)
    print("The best distance of the screen is %f"  %dos)

    montel = CompoundOpticalElement.initialize_as_montel_parabolic(p=p, q=q, theta=theta, bound1=bound1, bound2=bound2,distance_of_the_screen=q)
    beam03 = montel.trace_montel(beam)

    print(qq,dx)

    print(beam03[2].N/25000)

    plt.figure()
    plt.plot(beam03[0].x, beam03[0].z, 'ro')
    plt.plot(beam03[1].x, beam03[1].z, 'bo')
    plt.plot(beam03[2].x, beam03[2].z, 'go')
    plt.xlabel('x axis')
    plt.ylabel('z axis')
    plt.axis('equal')

    beam03[2].plot_xpzp(0)
    beam03[2].plot_ypzp(0)

    plt.show()

if main == "__main3__":

    beam = Beam(25000)
    beam.set_circular_spot(1e-4)
    beam.set_flat_divergence(0.01, 0.01)


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

    beam.flag *= 0

    p = 5.
    q = 15.
    theta = 88.*np.pi/180

    xmax = 0.
    xmin = -0.4
    ymax =  0.4
    ymin = -0.4
    zmax =  0.4
    zmin = 0.


    bound1 = BoundaryRectangle(xmax, xmin, ymax, ymin, zmax, zmin)
    bound2 = BoundaryRectangle(xmax, xmin, ymax, ymin, zmax, zmin)


    Nn = 10
    qq = np.ones(Nn)
    dx = np.ones(Nn)


    for i in range (0, Nn):

        beam1 = beam.duplicate()
        qq[i] = q - 0.1 + 0.2 * i / Nn

        montel = CompoundOpticalElement.initialize_as_montel_ellipsoid(p=p, q=q, theta=theta, bound1=bound1, bound2=bound2, distance_of_the_screen=qq[i])
        beam03 = montel.trace_montel(beam1)

        dx[i] = max(beam03[2].x) - min(beam03[2].x)



    plt.figure()
    plt.plot(qq, dx)

    min_of_qq = min(dx)
    indice = np.where(dx == min_of_qq)
    dos = qq[indice]
    print("The best distance of the screen is %f"  %dos)

    montel = CompoundOpticalElement.initialize_as_montel_ellipsoid(p=p, q=q, theta=theta, bound1=bound1, bound2=bound2)
    beam03 = montel.trace_montel(beam)


    plt.figure()
    plt.plot(beam03[0].x, beam03[0].z, 'ro')
    plt.plot(beam03[1].x, beam03[1].z, 'bo')
    plt.plot(beam03[2].x, beam03[2].z, 'go')
    plt.xlabel('x axis')
    plt.ylabel('z axis')
    plt.axis('equal')

    plt.show()

if main == "__main4__":

    beam = Beam.initialize_as_person()
    beam.x *= 1e-4
    beam.z *= 1e-4
    beam.set_flat_divergence(0.0001, 0.0001)


    #beam = Beam(25000)
    #beam.set_flat_divergence(25*1e-6, 25*1e-6)
    #beam.set_rectangular_spot(xmax=25*1e-6, xmin=-25*1e-6, zmax=5*1e-6, zmin=-5*1e-6)
    #beam.set_gaussian_divergence(0.01/3, 0.01/3)
    #beam.set_divergences_collimated()

    p = 5.
    q = 15.
    theta = 88. * np.pi / 180.

    xmax = 0.
    xmin = -0.4
    ymax =  0.1
    ymin = -0.1
    zmax =  0.4
    zmin = 0.

    bound = BoundaryRectangle(xmax=xmax, xmin=xmin, ymax=ymax, ymin=ymin, zmax=zmax, zmin=zmin)

    Nn = 25
    qq = np.ones(Nn)
    dx = np.ones(Nn)


    for i in range (0, Nn):
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Iteration %d"  %i)

        beam1 = beam.duplicate()
        qq[i] = q - 4 + 8. * i / Nn

        montel = CompoundOpticalElement.initialize_as_montel_ellipsoid(p=p, q=q, theta=theta, bound1=bound, bound2=bound, distance_of_the_screen=qq[i])
        beam03 = montel.trace_montel(beam1)

        dx[i] = max(beam03[2].x) - min(beam03[2].x)



    plt.figure()
    plt.plot(qq, dx)

    min_of_qq = min(dx)
    indice = np.where(dx == min_of_qq)
    dos = qq[indice]
    print("The best distance of the screen is %f\the dx is %f"  %(dos, dx[indice]))


    montel = CompoundOpticalElement.initialize_as_montel_ellipsoid(p=p, q=q, theta=theta, bound1=bound, bound2=bound)
    print(montel.info())

    beam03 = montel.trace_montel(beam)


    plt.figure()
    plt.plot(beam03[0].x, beam03[0].z, 'ro')
    plt.plot(beam03[1].x, beam03[1].z, 'bo')
    plt.plot(beam03[2].x, beam03[2].z, 'go')
    plt.xlabel('x axis')
    plt.ylabel('z axis')
    plt.axis('equal')

    plt.show()


if main == "__main5__":

    p = 5.
    q = 15.
    theta = 88*np.pi/180
    alpha = 90*np.pi/180

    oe1 = Optical_element.initialize_as_surface_conic_ellipsoid_from_focal_distances(p=p, q=q, theta=theta, alpha=0, cylindrical=1)


    oe2 = oe1.duplicate()
    oe2.rotation_surface_conic(alpha, 'y')

    print(oe1.info(), oe2.info())
