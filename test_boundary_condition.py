from Beam import Beam
from OpticalElement import Optical_element
from Shape import BoundaryRectangle
import numpy as np
import matplotlib.pyplot as plt
from numpy.testing import assert_almost_equal


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
    oe1 = Shadow.OE()
    oe2 = Shadow.OE()

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
    oe0.NPOINT = 10000
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

    #
    # run optical element 1

    return beam


def trace_shadow(beam):
    #
    # Python script to run shadow3. Created automatically with ShadowTools.make_python_script_from_list().
    #
    import Shadow
    import numpy

    # write (1) or not (0) SHADOW files start.xx end.xx star.xx
    iwrite = 0

    #

    oe1 = Shadow.OE()
    oe2 = Shadow.OE()


    oe1.DUMMY = 100.0
    oe1.FHIT_C = 1
    oe1.FWRITE = 1
    oe1.RLEN1 = 0.05
    oe1.RLEN2 = 0.05
    oe1.RWIDX1 = 0.005
    oe1.RWIDX2 = 0.005
    oe1.T_IMAGE = 1.0
    oe1.T_INCIDENCE = 65.0
    oe1.T_REFLECTION = 65.0
    oe1.T_SOURCE = 2.0

    oe2.ALPHA = 90
    oe2.DUMMY = 100.0
    oe2.FHIT_C = 1
    oe2.FMIRR = 4
    oe2.FWRITE = 1
    oe2.RLEN1 = 0.1
    oe2.RLEN2 = 0.1
    oe2.RWIDX1 = 0.01
    oe2.RWIDX2 = 0.01
    oe2.T_IMAGE = 2.0
    oe2.T_INCIDENCE = 28.0
    oe2.T_REFLECTION = 28.0
    oe2.T_SOURCE = 5.0

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

    #
    # run optical element 2
    #
    print("    Running optical element: %d" % (2))
    if iwrite:
        oe2.write("start.02")
    beam.traceOE(oe2, 2)
    if iwrite:
        oe2.write("end.02")
        beam.write("star.02")

    return beam

def test_boundary_condition():
    #beam1 = Beam(10000)
    #beam1.set_point(0, 0, 0)
    #beam1.set_flat_divergence(5e-3, 5e-2)

    shadow_beam = run_shadow_source()

    beam1 = Beam(10000)
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


    bound1=BoundaryRectangle(xmax=0.005,xmin=-0.005,ymax=0.05,ymin=-0.05)
    bound2=BoundaryRectangle(xmax=0.01,xmin=-0.01,ymax=0.1,ymin=-0.1)

    plane_mirror=Optical_element.initialize_as_plane_mirror(2,1,65*np.pi/180,0)
    parabolic_mirror=Optical_element.initialize_as_surface_conic_paraboloid_from_focal_distances(5,2,28*np.pi/180,90*np.pi/180)


    plane_mirror.rectangular_bound(bound1)
    parabolic_mirror.rectangular_bound(bound2)

    beam1=plane_mirror.trace_optical_element(beam1)
    beam1=parabolic_mirror.trace_optical_element(beam1)

    beam1.plot_xz()
    plt.title("Total points plot")
    beam1.plot_good_xz()
    plt.title("Good points plot")

    print(beam1.flag)

    indices=np.where(beam1.flag>0)

    print("The good number of ray are:    %f"   %(beam1.flag[indices].size))

    plt.show()


    shadow_beam=trace_shadow(shadow_beam)


    assert_almost_equal(beam1.x, shadow_beam.getshonecol(1), 8)
    assert_almost_equal(beam1.y, shadow_beam.getshonecol(2), 8)
    assert_almost_equal(beam1.z, shadow_beam.getshonecol(3), 8)

