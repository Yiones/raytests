#
# Python script to run shadow3. Created automatically with ShadowTools.make_python_script_from_list().
#
import Shadow
import numpy as np
import matplotlib.pylab as plt

from OpticalElement import Optical_element
from Beam import Beam

def run_shadow():


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
    oe0.PH1 = 1000.0
    oe0.VDIV1 = 0.05
    oe0.VDIV2 = 0.05

    oe1.DUMMY = 100.0
    oe1.FMIRR = 1
    oe1.FWRITE = 1
    oe1.T_IMAGE = 1.0
    oe1.T_INCIDENCE = 45.0
    oe1.T_REFLECTION = 45.0
    oe1.T_SOURCE = 1.0

    oe2.DUMMY = 100.0
    oe2.FWRITE = 3
    oe2.F_REFRAC = 2
    oe2.F_SCREEN = 1
    oe2.N_SCREEN = 1
    oe2.T_IMAGE = 0.0
    oe2.T_INCIDENCE = 0.0
    oe2.T_REFLECTION = 180.0
    oe2.T_SOURCE = 0.0

    # Run SHADOW to create the source

    if iwrite:
        oe0.write("start.00")

    beam.genSource(oe0)

    # if iwrite:
    #     oe0.write("end.00")
    #     beam.write("begin.dat")
    #
    # #
    # # run optical element 1
    # #
    # print("    Running optical element: %d" % (1))
    # if iwrite:
    #     oe1.write("start.01")
    # beam.traceOE(oe1, 1)
    # if iwrite:
    #     oe1.write("end.01")
    #     beam.write("star.01")
    #
    # #
    # # run optical element 2
    # #
    # print("    Running optical element: %d" % (2))
    # if iwrite:
    #     oe2.write("start.02")
    # beam.traceOE(oe2, 2)
    # if iwrite:
    #     oe2.write("end.02")
    #     beam.write("star.02")
    #
    # Shadow.ShadowTools.plotxy(beam, 1, 3, nbins=101, nolost=1, title="Real space")
    # Shadow.ShadowTools.plotxy(beam,1,4,nbins=101,nolost=1,title="Phase space X")
    # Shadow.ShadowTools.plotxy(beam,3,6,nbins=101,nolost=1,title="Phase space Z")

    return beam

if __name__ == "__main__":

    beam_shadow = run_shadow()




    beam1=Beam()
    # beam1.set_point(0,0,0)
    # beam1.set_flat_divergence(5e-3,5e-2)

    beam1.initialize_from_arrays(
        beam_shadow.getshonecol(1),
        beam_shadow.getshonecol(2),
        beam_shadow.getshonecol(3),
        beam_shadow.getshonecol(4),
        beam_shadow.getshonecol(5),
        beam_shadow.getshonecol(6),
        beam_shadow.getshonecol(10),
    )

    #### Data of the plane mirron

    p=1.
    q=1.
    theta=45
    alpha=90
    R=2*p*q/(q+p)/np.cos(theta)
    spherical_mirror=Optical_element.initialize_as_spherical_mirror(p,q,theta,alpha,R)

    beam1=spherical_mirror.trace_optical_element(beam1)
    beam1.plot_xz()
    plt.title("xz diagram on the image plane")
    beam1.plot_xpzp()
    plt.title("xpzp diagram on the image plane")



    plt.show()