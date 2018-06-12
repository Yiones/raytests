from Beam import Beam
from OpticalElement import Optical_element
from Shape import BoundaryRectangle
import numpy as np
import matplotlib.pyplot as plt
from numpy.testing import assert_almost_equal
from CompoundOpticalElement import CompoundOpticalElement
from Vector import Vector
from SurfaceConic import SurfaceConic

#def test_wolter1():
#
#    beam1 = Beam()
#    beam1.set_point(15., 0., 15.)
#    op_ax = Beam (1)
#    op_ax.set_point(15., 0., 15.)
#    # beam1.set_rectangular_spot(55/2*1,-55/2*1,55/2*1,-55/2*1)
#    beam1.set_circular_spot(25*1e-3)
#
#    beam=op_ax.merge(beam1)
#    beam.set_divergences_collimated()
#
#    #beam.plot_xz()
#
#
#    p = 36.
#    q = 25.
#    focal = 25.
#    z0 = 15
#    theta = 0 * np.pi / 180
#    alpha = 0 * np.pi / 180
#
#    paraboloid = Optical_element.initialize_as_surface_conic_paraboloid_from_focal_distances(p, 0, theta, alpha, "p", focal)
#
#    beam = paraboloid.trace_Wolter_1(beam, z0)
#
#    beam.retrace(10.)
#
#
#    beam.plot_xz()
#    plt.show()



#def test_compound_wolter1():
#
#    p=0.01
#    beam1 = Beam()
#    beam1.set_point(p, 0., p)
#
#    op_ax = Beam (1)
#    op_ax.set_point(p, 0., p)
#    beam1.set_circular_spot(25*1e-3)
#
#    beam=op_ax.merge(beam1)
#    beam.set_divergences_collimated()
#
#    beam.plot_xz()
#
#
#    p = 2.
#    q = 25.
#    z0 = 1.
#
#    wolter1 = CompoundOpticalElement.initialiaze_as_wolter_1(p, q, z0)
#
#    beam = wolter1.trace_compound(beam)
#    beam.plot_xz()
#    beam.retrace(10.)
#    beam.plot_xz()
#    plt.show()



#def test_compound_wolter1():
#
#    p=100.
#    beam1 = Beam.initialize_as_person()
#    #beam1.x *= 1000
#    #beam1.z *= 1000
#    beam1.set_point(p, 0., p)
#
#    op_ax = Beam (1)
#    op_ax.set_point(p, 0., p)
#
#    beam=op_ax.merge(beam1)
#    beam.set_divergences_collimated()
#
#    beam.plot_xz()
#
#
#    p = 36.
#    q = 25.
#    z0 = 0.
#
#
#    wolter1 = CompoundOpticalElement.initialiaze_as_wolter_1(p, q, z0)
#    print(wolter1.oe[0].ccc_object.get_coefficients())
#    print(wolter1.oe[1].ccc_object.get_coefficients())
#
#    beam = wolter1.trace_compound(beam)
#    beam.plot_xz()
#    beam.retrace(10.)
#    beam.plot_xz()
#    plt.show()






#def test_compound_wolter1_with_hole():
#
#    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>   test_compound_wolter1_with_hole")
#
#
#    p=21.
#    beam1 = Beam.initialize_as_person(100000)
#    beam1.x *= 1.
#    beam1.z *= 1.
#    beam1.set_point(p, 0., p)
#
#    op_ax = Beam (1)
#    op_ax.set_point(p, 0., p)
#
#    beam=op_ax.merge(beam1)
#    beam.set_divergences_collimated()
#
#    beam.plot_xz()
#
#    print(beam.flag)
#
#    p = 3600.
#    q = 25.
#    z0 = 15.
#    wolter1 = CompoundOpticalElement.initialiaze_as_wolter_1(p, q, z0)
#
#    beam = wolter1.trace_with_hole(beam)
#
#    beam.plot_xz()
#    beam.retrace(10.)
#    beam.plot_good_xz(0)
#
#    #print(wolter1.info())
#
#    plt.show()

#def test_wolter1():
#    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  test_wolter_1")
#
#    p = 40.
#    beam1 = Beam.initialize_as_person()
#    beam1.set_point(p, 0., p)
#
#    op_ax = Beam(1)
#    op_ax.set_point(p, 0., p)
#
#    beam = op_ax.merge(beam1)
#    beam.set_divergences_collimated()
#
#    beam.plot_xz()
#
#    p = 3600000.
#    q = 45.
#    z0 = 30.
#
#    wolter1 = CompoundOpticalElement.initialiaze_as_wolter_1(p, q, z0)
#
#    beam = wolter1.trace_compound(beam)
#    #beam.plot_yx()
#
#    t = -beam.x/beam.vx
#    beam.x = beam.x + t * beam.vx
#    beam.y = beam.y + t * beam.vy
#    beam.z = beam.z + t * beam.vz
#
#
#
#    #beam.retrace(10.)
#    beam.plot_xz()
#    beam.plot_yx()
#    wolter1.oe[1].output_frame_wolter(beam)
#    beam.retrace(10.)
#    beam.plot_xz()
#    beam.plot_yx()
#    plt.title("Wolter 1")
#
#
#
#    print(wolter1.info())
#
#    print(np.mean(beam.y))
#
#    plt.show()

#def test_wolter1():
#    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  test_wolter_1")
#
#    p = 10000.
#    beam1 = Beam.initialize_as_person()
#    beam1.x *= 100
#    beam1.z *= 100
#    beam1.set_point(p, 0., p)
#
#    op_ax = Beam(1)
#    op_ax.set_point(p, 0., p)
#
#    beam = op_ax.merge(beam1)
#    beam.set_divergences_collimated()
#
#    beam.plot_xz()
#
#    p = 3600000.
#    q = 100.
#    z0 = 80.
#
#    wolter1 = CompoundOpticalElement.initialiaze_as_wolter_1(p, q, z0)
#    print(wolter1.info())
#
#    oe1 = Optical_element.initialize_as_surface_conic_paraboloid_from_focal_distances(p=p, q=0, theta=0., alpha=0., infinity_location="p", focal=2*z0-q)
#    oe2 = Optical_element.initialize_my_hyperboloid(p=0., q=100., theta=90.*np.pi/180, alpha=0., wolter=1, z0=80, distance_of_focalization=2*z0-q)
#
#    #oe1.rotation_to_the_optical_element(beam)
#    #oe1.translation_to_the_optical_element(beam)
#    #[beam, t] = oe1.intersection_with_optical_element(beam)
#    #oe1.output_direction_from_optical_element(beam)
#
#    #[beam, t] = oe2.intersection_with_optical_element(beam)
#    #oe2. output_direction_from_optical_element(beam)
#
#    #oe2.theta = 0.
#    #oe2.rotation_to_the_screen(beam)
#    #oe2.translation_to_the_screen(beam)
#    #oe2.intersection_with_the_screen(beam)
#
#    #wolter1.oe[0].rotation_to_the_optical_element(beam)
#    #wolter1.oe[0].translation_to_the_optical_element(beam)
#    #[beam, t] = wolter1.oe[0].intersection_with_optical_element(beam)
#    #wolter1.oe[0].output_direction_from_optical_element(beam)
#
#    #wolter1.oe[0].effect_of_optical_element(beam)
#
#    #wolter1.oe[1].effect_of_optical_element(beam)
#
#    #wolter1.oe[1].theta = 0.
#    #wolter1.oe[1].effect_of_the_screen(beam)
#
#    beam = wolter1.trace_compound(beam)
#    #t = -beam.x / beam.vx
#    #beam.x = beam.x + t * beam.vx
#    #beam.y = beam.y + t * beam.vy
#    #beam.z = beam.z + t * beam.vz
#
#    beam.plot_xz()
#    print("mean(beam.x)=%f, mean(beam.y)=%f, mean(beam.z)=%f" %(np.mean(beam.x),np.mean(beam.y),np.mean(beam.z)))
#    oe2.output_frame_wolter(beam)
#
#    beam.retrace(10.)
#    beam.plot_xz()
#
#    plt.show()




#def test_wolter1():
#    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  test_wolter_1")
#
#    p = 1000.
#    beam1 = Beam.initialize_as_person()
#    beam1.x *= 1e-3
#    beam1.z *= 1e-3
#    beam1.set_point(p, 0., p)
#
#    op_ax = Beam(1)
#    op_ax.set_point(p, 0., p)
#
#    beam = op_ax.merge(beam1)
#    beam.set_divergences_collimated()
#
#    beam.plot_xz()
#
#    p = 3600000.
#    theta =1e-3*np.pi/180
#    q=100
#    D=50
#
#
#    b = D*np.tan(2*theta)/2
#    c = -4*D**2
#
#    focal1 = -b + np.sqrt(b**2 - c)
#    focal2 = -b - np.sqrt(b**2 - c)
#
#    focal = max(focal1, focal2)
#
#    oe1 = Optical_element.initialize_as_surface_conic_paraboloid_from_focal_distances(p=p, q=0., theta=0., alpha=0., infinity_location="p", focal=focal)
#
#    s1 = D/np.tan(2*theta)
#    s2 = q*np.sin(np.arccos(D/q))
#    d = s1+s2
#    z0 = focal+d/2
#
#    print("s1=%f, s2=%f, d=%f, z0=%f, focal=%f" %(s1,s2,d,z0,focal))


#   #plt.show()


#def test_wolter1():
#    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  test_wolter_1")
#
#    p = 0.
#    beam1 = Beam.initialize_as_person()
#    beam1.x *= 100000.
#    beam1.z *= 100000.
#    beam1.set_point(p, 0., p)
#
#    op_ax = Beam(1)
#    op_ax.set_point(p, 0., p)
#
#    beam = op_ax.merge(beam1)
#    beam.set_divergences_collimated()
#
#    beam.plot_xz()
#
#    p = 3600000.
#    q = 100.
#    z0 = 80.
#
#    wolter1 = CompoundOpticalElement.initialiaze_as_wolter_1(p, q, z0)
#
#    beam = wolter1.trace_compound(beam)
#
#    beam.plot_xz()
#
#    beam.retrace(10.)
#    beam.plot_xz()
#
#    plt.show()



#def test_wolter1():
#    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  test_wolter_1")
#
#    p = 1000.
#    beam1 = Beam.initialize_as_person()
#    beam1.x *= 1.
#    beam1.z *= 1.
#    beam1.set_point(p, 0., p)
#
#    op_ax = Beam(1)
#    op_ax.set_point(p, 0., p)
#
#    beam = op_ax.merge(beam1)
#    beam.set_divergences_collimated()
#
#    beam.plot_xz()
#
#
#    p = 6*1e24
#    R = 100.
#    theta = 1e-3*np.pi/180.
#    q=120.
#
#    cp1 = -2*R/np.tan(theta)
#    cp2 = 2*R*np.tan(theta)
#    cp = max(cp1,cp2)
#    f = cp/4
#    print("focal=%f" %(f))
#
#    oe1 = Optical_element.initialize_as_surface_conic_paraboloid_from_focal_distances(p=p, q=f, theta=0., alpha=0., infinity_location="p")
#
#    s1 = R/np.tan(2*theta)
#    s2 = q*np.sin(np.arccos(R/q))
#    c=(s1-s2)/2
#    z0 = f+c
#
#    print(c)
#
#    b1 = np.sqrt(0.5*c**2+0.5*R**2+0.5*R**4/cp**2-R**2*z0/cp+0.5*z0**2-0.5/cp**2*np.sqrt((-c**2*cp**2-cp**2*R**2-R**4+2*cp*R**2*z0-cp**2*z0**2)**2-4*cp**2*(c**2*R**4-2*c**2*cp*R**2*z0+c**2*cp**2*z0**2)))
#    b2 = np.sqrt(0.5*c**2+0.5*R**2+0.5*R**4/cp**2-R**2*z0/cp+0.5*z0**2+0.5/cp**2*np.sqrt((-c**2*cp**2-cp**2*R**2-R**4+2*cp*R**2*z0-cp**2*z0**2)**2-4*cp**2*(c**2*R**4-2*c**2*cp*R**2*z0+c**2*cp**2*z0**2)))
#    b = min(b1,b2)
#    a=np.sqrt(c**2-b**2)
#
#    ccc = np.array([-1/a**2, -1/a**2, 1/b**2, 0., 0., 0., 0., 0., -2*z0/b**2, z0**2/b**2-1 ])
#    oe2 = Optical_element.initialize_as_surface_conic_from_coefficients(ccc)
#    oe2.type = "My hyperbolic mirror"
#    oe2.p = 0.
#    oe2.q = z0+c
#    oe2.theta = 90*np.pi/180
#    oe2.alpha = 0.
#
#    oe1.effect_of_optical_element(beam)
#    oe2.intersection_with_optical_element(beam)
#    oe2.output_direction_from_optical_element(beam)
#    oe2.theta = 0.
#    #oe2.effect_of_the_screen(beam)
#
#    t=-beam.x/beam.vx
#    beam.x = beam.x + beam.vx * t
#    beam.y = beam.y + beam.vy * t
#    beam.z = beam.z + beam.vz * t
#
#    beam.plot_xz()
#
#    print("mean(beam.x)=%f, mean(beam.y)=%f, mean(beam.z)=%f" %(np.mean(beam.x),np.mean(beam.y),np.mean(beam.z)))
#    print("variance(beam.x)=%f, variance(beam.y)=%f, variance(beam.z)=%f" %(np.mean(beam.x**2),np.mean(beam.y**2),np.mean(beam.z**2)))
#    print(z0+c)
#
#    plt.show()


#def test_wolter1():        #####Good and last one with only two parameters R and theta
#    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  test_wolter_1")
#
#    p = 1000.
#    beam1 = Beam.initialize_as_person()
#    beam1.x *= 1.
#    beam1.z *= 1.
#    beam1.set_point(p, 0., p)
#    op_ax = Beam(1)
#    op_ax.set_point(p, 0., p)
#    beam = op_ax.merge(beam1)
#    beam.set_divergences_collimated()
#    beam.plot_xz()
#
#
#    p = 6*1e15
#    R = 100.
#    theta = 1e-3*np.pi/180.
#
#    cp1 = -2*R/np.tan(theta)
#    cp2 = 2*R*np.tan(theta)
#    cp = max(cp1,cp2)
#    f = cp/4
#    print("focal=%f" %(f))
#
#    oe1 = Optical_element.initialize_as_surface_conic_paraboloid_from_focal_distances(p=p, q=f, theta=0., alpha=0., infinity_location="p")
#
#    s1 = R/np.tan(2*theta)
#    s2 = R/np.tan(4*theta)
#    c=(s1-s2)/2
#    z0 = f+c
#
#    print(c)
#
#    b1 = np.sqrt(0.5*c**2+0.5*R**2+0.5*R**4/cp**2-R**2*z0/cp+0.5*z0**2-0.5/cp**2*np.sqrt((-c**2*cp**2-cp**2*R**2-R**4+2*cp*R**2*z0-cp**2*z0**2)**2-4*cp**2*(c**2*R**4-2*c**2*cp*R**2*z0+c**2*cp**2*z0**2)))
#    b2 = np.sqrt(0.5*c**2+0.5*R**2+0.5*R**4/cp**2-R**2*z0/cp+0.5*z0**2+0.5/cp**2*np.sqrt((-c**2*cp**2-cp**2*R**2-R**4+2*cp*R**2*z0-cp**2*z0**2)**2-4*cp**2*(c**2*R**4-2*c**2*cp*R**2*z0+c**2*cp**2*z0**2)))
#    b = min(b1,b2)
#    a=np.sqrt(c**2-b**2)
#
#
#    ccc = np.array([-1/a**2, -1/a**2, 1/b**2, 0., 0., 0., 0., 0., -2*z0/b**2, z0**2/b**2-1 ])
#    oe2 = Optical_element.initialize_as_surface_conic_from_coefficients(ccc)
#    oe2.type = "My hyperbolic mirror"
#    oe2.p = 0.
#    oe2.q = z0+c
#    oe2.theta = 90*np.pi/180
#    oe2.alpha = 0.
#
#    print(oe2.info())
#
#    oe1.effect_of_optical_element(beam)
#    oe2.intersection_with_optical_element(beam)
#    oe2.output_direction_from_optical_element(beam)
#    oe2.theta = 0.
#    oe2.effect_of_the_screen(beam)
#
#    #t=-beam.x/beam.vx
#    #beam.x = beam.x + beam.vx * t
#    #beam.y = beam.y + beam.vy * t
#    #beam.z = beam.z + beam.vz * t
#
#    oe2.output_frame_wolter(beam)
#
#    beam.plot_xz()
#    beam.retrace(10.)
#    beam.plot_xz()
#
#    #print("mean(beam.x)=%f, mean(beam.y)=%f, mean(beam.z)=%f" %(np.mean(beam.x),np.mean(beam.y),np.mean(beam.z)))
#    #print("variance(beam.x)=%f, variance(beam.y)=%f, variance(beam.z)=%f" %(np.mean(beam.x**2),np.mean(beam.y**2),np.mean(beam.z**2)))
#    #print(z0+c)
#
#    plt.show()
#
#    print(oe1.info())
#    print(oe2.info())




def test_compound_wolter1_with_hole():

    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>   test_compound_wolter1_with_hole")



    p = 100.
    beam1 = Beam.initialize_as_person(100000)
    beam1.x *= 50.
    beam1.z *= 50.
    beam1.set_point(p, 0., p)
    op_ax = Beam(1)
    op_ax.set_point(p, 0., p)
    beam = op_ax.merge(beam1)
    beam.set_divergences_collimated()
    beam.plot_xz()



    p=6*1e8
    R=100.
    theta=0.001*np.pi/180

    wolter = CompoundOpticalElement.initialiaze_as_wolter_1_with_two_parameters(p1=p, R=R, theta=theta)


    #beam = wolter.trace_compound(beam)
    beam = wolter.trace_with_hole(beam)

    beam.plot_good_xz()
    beam.retrace(10.)
    beam.plot_good_xz()

    plt.show()