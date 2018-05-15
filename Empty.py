from Beam import Beam
from OpticalElement import Optical_element
from Shape import BoundaryRectangle
import numpy as np
from SurfaceConic import SurfaceConic
import matplotlib.pyplot as plt


def generate_shadow_beam():
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
    oe0.FSOUR = 1
    oe0.F_PHOT = 0
    oe0.HDIV1 = 0.0
    oe0.HDIV2 = 0.0
    oe0.IDO_VX = 0
    oe0.IDO_VZ = 0
    oe0.IDO_X_S = 0
    oe0.IDO_Y_S = 0
    oe0.IDO_Z_S = 0
    oe0.NPOINT = 10000
    oe0.PH1 = 1000.0
    oe0.VDIV1 = 0.0
    oe0.VDIV2 = 0.0
    oe0.WXSOU = 5.0
    oe0.WZSOU = 5.0


    # Run SHADOW to create the source

    if iwrite:
        oe0.write("start.00")

    beam.genSource(oe0)

    if iwrite:
        oe0.write("end.00")
        beam.write("begin.dat")

    return beam


beam=Beam()
beam.set_divergences_collimated()
beam.set_point(0.,0.,1000.)
beam.set_rectangular_spot(5/2,-5/2,5/2,-5/2)


#shadow_beam=generate_shadow_beam()
#
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

beam.plot_xz()
beam.plot_xpzp()

p = 10000.
q = 25.
theta = 0 * np.pi / 180
alpha = 0 * np.pi / 180

#par_mirro=Optical_element.initialize_as_surface_conic_paraboloid_from_focal_distances(p,q,theta,alpha,"p")
#beam=par_mirro.trace_optical_element(beam)

#ccc=SurfaceConic([1.,1.,0,0,0,0,0,0,-1000.,0])
#prova=Optical_element(ccc)
#prova.set_parameters(p,q,theta,alpha)
#prova.type="Surface conical mirror"

prova=Optical_element.initialize_as_surface_conic_paraboloid_from_focal_distances(p,q,theta,alpha,"p")
print(prova.ccc_object.get_coefficients())

beam=prova.trace_optical_element(beam)
#
beam.plot_xz()
plt.show()