from Beam import Beam
from OpticalElement import Optical_element
import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_almost_equal
from SurfaceConic import SurfaceConic

do_plot = False

def run_shadow_source():
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
    oe0.HDIV1 = 0.005
    oe0.HDIV2 = 0.005
    oe0.IDO_VX = 0
    oe0.IDO_VZ = 0
    oe0.IDO_X_S = 0
    oe0.IDO_Y_S = 0
    oe0.IDO_Z_S = 0
    oe0.PH1 = 1000.0
    oe0.VDIV1 = 0.05
    oe0.VDIV2 = 0.05

    # Run SHADOW to create the source

    if iwrite:
        oe0.write("start.00")

    beam.genSource(oe0)

    if iwrite:
        oe0.write("end.00")
        beam.write("begin.dat")
    return beam

def run_shadow_spherical_mirror(beam):
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

    oe1 = Shadow.OE()

    #
    # Define variables. See meaning of variables in:
    #  https://raw.githubusercontent.com/srio/shadow3/master/docs/source.nml
    #  https://raw.githubusercontent.com/srio/shadow3/master/docs/oe.nml
    #


    oe1.DUMMY = 100.0
    oe1.FMIRR = 1
    oe1.FWRITE = 1
    oe1.T_IMAGE = 1.0
    oe1.T_INCIDENCE = 41.0
    oe1.T_REFLECTION = 41.0
    oe1.T_SOURCE = 2.0


    #
    # run optical element 1
    #
    print("    Running optical element: %d" % (1))
    if iwrite:
        oe1.write("start.01")
    beam.traceOE(oe1, 1)
    if iwrite:
        oe1.write("end.01")
        beam.write("star.01")

    return beam

def run_shadow_elliptical_mirror(beam):
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
    oe1 = Shadow.OE()

    #
    # Define variables. See meaning of variables in:
    #  https://raw.githubusercontent.com/srio/shadow3/master/docs/source.nml
    #  https://raw.githubusercontent.com/srio/shadow3/master/docs/oe.nml
    #

    oe0.FDISTR = 1
    oe0.FSOUR = 0
    oe0.F_PHOT = 0
    oe0.HDIV1 = 0.005
    oe0.HDIV2 = 0.005
    oe0.IDO_VX = 0
    oe0.IDO_VZ = 0
    oe0.IDO_X_S = 0
    oe0.IDO_Y_S = 0
    oe0.IDO_Z_S = 0
    oe0.PH1 = 1000.0
    oe0.VDIV1 = 0.05
    oe0.VDIV2 = 0.05

    oe1.DUMMY = 100.0
    oe1.FMIRR = 2
    oe1.FWRITE = 1
    oe1.T_IMAGE = 10.0
    oe1.T_INCIDENCE = 50.0
    oe1.T_REFLECTION = 50.0
    oe1.T_SOURCE = 20.0

    # Run SHADOW to create the source

    if iwrite:
        oe0.write("start.00")

    beam.genSource(oe0)

    if iwrite:
        oe0.write("end.00")
        beam.write("begin.dat")

    #
    # run optical element 1
    #
    print("    Running optical element: %d" % (1))
    if iwrite:
        oe1.write("start.01")
    beam.traceOE(oe1, 1)
    if iwrite:
        oe1.write("end.01")
        beam.write("star.01")

    return beam

def run_shadow_parabolic_mirror(beam):
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

    oe1 = Shadow.OE()

    #
    # Define variables. See meaning of variables in:
    #  https://raw.githubusercontent.com/srio/shadow3/master/docs/source.nml
    #  https://raw.githubusercontent.com/srio/shadow3/master/docs/oe.nml
    #

    oe1.DUMMY = 100.0
    oe1.FMIRR = 4
    oe1.FWRITE = 1
    oe1.T_INCIDENCE = 72.0
    oe1.T_REFLECTION = 72.0

    #
    # run optical element 1
    #
    print("    Running optical element: %d" % (1))
    if iwrite:
        oe1.write("start.01")
    beam.traceOE(oe1, 1)
    if iwrite:
        oe1.write("end.01")
        beam.write("star.01")

    return beam

def run_shadow_hyperbolic_mirror(beam):
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

    oe1 = Shadow.OE()

    #
    # Define variables. See meaning of variables in:
    #  https://raw.githubusercontent.com/srio/shadow3/master/docs/source.nml
    #  https://raw.githubusercontent.com/srio/shadow3/master/docs/oe.nml
    #

    oe1.DUMMY = 100.0
    oe1.FMIRR = 7
    oe1.FWRITE = 1
    oe1.T_IMAGE = 28.0
    oe1.T_INCIDENCE = 28.0
    oe1.T_REFLECTION = 28.0


    #
    # run optical element 1
    #
    print("    Running optical element: %d" % (1))
    if iwrite:
        oe1.write("start.01")
    beam.traceOE(oe1, 1)
    if iwrite:
        oe1.write("end.01")
        beam.write("star.01")


    return beam



def test_spherical_mirror():

    print(">>>>>>>>>>>>>>> test_spherical_mirror")
    shadow_beam = run_shadow_source()

    beam1=Beam()
    beam1.initialize_from_arrays(
        shadow_beam.getshonecol(1),
        shadow_beam.getshonecol(2),
        shadow_beam.getshonecol(3),
        shadow_beam.getshonecol(4),
        shadow_beam.getshonecol(5),
        shadow_beam.getshonecol(6),
        shadow_beam.getshonecol(10),
        0
    )

    #beam1 = Beam(5000)
    #beam1.set_point(0, 0, 0)
    #beam1.set_flat_divergence(5e-3, 5e-2)
    p=2.
    q=1.
    theta=41*np.pi/180

    shadow_beam = run_shadow_source()

    spherical_mirror=Optical_element.initialize_as_surface_conic_sphere_from_focal_distances(p,q, theta)

    beam1=spherical_mirror.trace_optical_element(beam1)


    if do_plot:
        beam1.plot_xz()
        beam1.plot_xpzp()
        plt.title("Spherical mirror with p=2, q=1, theta=41")
        plt.show()

    shadow_beam = run_shadow_spherical_mirror(shadow_beam)


    assert_almost_equal(beam1.x, shadow_beam.getshonecol(1), 8)
    assert_almost_equal(beam1.y, shadow_beam.getshonecol(2), 8)
    assert_almost_equal(beam1.z, shadow_beam.getshonecol(3), 8)



def test_ellipsoidal_mirror():

    print(">>>>>>>>>>>>>>> test_ellipsoidal_mirror")

    #beam1=Beam(5000)
    #beam1.set_point(0,0,0)
    #beam1.set_flat_divergence(5e-3,5e-2)

    shadow_beam = run_shadow_source()

    beam1=Beam()
    beam1.initialize_from_arrays(
        shadow_beam.getshonecol(1),
        shadow_beam.getshonecol(2),
        shadow_beam.getshonecol(3),
        shadow_beam.getshonecol(4),
        shadow_beam.getshonecol(5),
        shadow_beam.getshonecol(6),
        shadow_beam.getshonecol(10),
        0
    )

    p=20.
    q=10.
    theta=50*np.pi/180

    spherical_mirror=Optical_element.initialize_as_surface_conic_ellipsoid_from_focal_distances(p,q,theta)

    beam1=spherical_mirror.trace_optical_element(beam1)

    if do_plot:
        beam1.plot_xz()
        beam1.plot_xpzp()
        plt.title("Ellipsoidal mirror with p=20, q=10, theta=50")
        plt.show()

    shadow_beam = run_shadow_elliptical_mirror(beam1)


    assert_almost_equal(beam1.vx, shadow_beam.getshonecol(4), 1)
    assert_almost_equal(beam1.vy, shadow_beam.getshonecol(5), 1)
    assert_almost_equal(beam1.vz, shadow_beam.getshonecol(6), 1)


def test_paraboloid_mirror():

    print(">>>>>>>>>>>>>>> test_paraboloid_mirror")
    #beam1=Beam(5000)
    #beam1.set_point(0,0,0)
    #beam1.set_flat_divergence(5e-3,5e-2)

    shadow_beam = run_shadow_source()

    beam1=Beam()
    beam1.initialize_from_arrays(
        shadow_beam.getshonecol(1),
        shadow_beam.getshonecol(2),
        shadow_beam.getshonecol(3),
        shadow_beam.getshonecol(4),
        shadow_beam.getshonecol(5),
        shadow_beam.getshonecol(6),
        shadow_beam.getshonecol(10),
        0
    )

    p=10.
    q=20.
    theta=72*np.pi/180
    alpha=0*np.pi/180
    spherical_mirror=Optical_element.initialize_as_surface_conic_paraboloid_from_focal_distances(p,q,theta,alpha)
    beam1=spherical_mirror.trace_optical_element(beam1)

    if do_plot:
        beam1.plot_xz()
        beam1.plot_xpzp()
        plt.title("Paraboloid mirror  with p=10, q=20, theta=72")
        print(spherical_mirror.ccc_object.get_coefficients())
        plt.show()

    shadow_beam = run_shadow_parabolic_mirror(shadow_beam)


    assert_almost_equal(beam1.x, shadow_beam.getshonecol(1), 7)
    assert_almost_equal(beam1.y, shadow_beam.getshonecol(2), 7)
    assert_almost_equal(beam1.z, shadow_beam.getshonecol(3), 7)




####     This is problematic     #######################################################################################
#
#def test_hyperboloid_mirror():
#    #beam1=Beam(5000)
#    #beam1.set_point(0,0,0)
#    #beam1.set_flat_divergence(5e-3,5e-2)
#
#    shadow_beam=run_shadow_source()
#
#    beam1=Beam(5000)
#    beam1.initialize_from_arrays(
#        shadow_beam.getshonecol(1),
#        shadow_beam.getshonecol(2),
#        shadow_beam.getshonecol(3),
#        shadow_beam.getshonecol(4),
#        shadow_beam.getshonecol(5),
#        shadow_beam.getshonecol(6),
#        shadow_beam.getshonecol(10),
#        0
#    )
#
#    p=1.
#    q=2.
#    theta = 76*np.pi/180
#    spherical_mirror=Optical_element.initialize_as_hyperboloid_from_focal_distances(p,q,theta)
#    beam1=spherical_mirror.trace_surface_conic(beam1)
#    beam1.plot_xz()
#    beam1.plot_xpzp()
#    plt.show()
#
#    shadow_beam=run_shadow_hyperbolic_mirror(shadow_beam)
#
#
########################################################################################################################