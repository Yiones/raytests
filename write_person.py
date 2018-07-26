from Beam import Beam
import matplotlib.pyplot as plt
import Shadow
from Shadow.ShadowTools import plotxy

beam = Beam.initialize_as_person()

# beam.plot_xz()
#
# plt.show()



#
# initialize shadow3 source (oe0) and beam
#
beam_shadow = Shadow.Beam()
oe0 = Shadow.Source()

#
# Define variables. See meaning of variables in:
#  https://raw.githubusercontent.com/srio/shadow3/master/docs/source.nml
#  https://raw.githubusercontent.com/srio/shadow3/master/docs/oe.nml
#

oe0.FDISTR = 1
oe0.FSOUR = 0
oe0.F_PHOT = 0
oe0.HDIV1 = 1e-12
oe0.HDIV2 = 1e-12
oe0.IDO_VX = 0
oe0.IDO_VZ = 0
oe0.IDO_X_S = 0
oe0.IDO_Y_S = 0
oe0.IDO_Z_S = 0
oe0.NPOINT = beam.N
oe0.PH1 = 1000.0
oe0.VDIV1 = 1e-12
oe0.VDIV2 = 1e-12



#Run SHADOW to create the source


beam_shadow.genSource(oe0)

beam_shadow.rays[:,0] = beam.x * 1e-3
beam_shadow.rays[:,2] = beam.z * 1e-3


plotxy(beam_shadow,1,3,nbins=500)

# if iwrite:
#     oe0.write("end.00")
beam_shadow.write("/users/yaouadi/Oasys/begin.dat")