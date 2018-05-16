from Beam import Beam
from OpticalElement import Optical_element
from Shape import BoundaryRectangle
import numpy as np
from SurfaceConic import SurfaceConic
import matplotlib.pyplot as plt



beam=Beam()
beam.set_divergences_collimated()
beam.set_point(20.,0.,20.)
beam.set_rectangular_spot(5/2*1e-6,-5/2*1e-6,5/2*1e-6,-5/2*1e-6)

print(beam.vy)

p = 100.
q = 25.
theta = 0 * np.pi / 180
alpha = 0 * np.pi / 180


prova = Optical_element.initialize_as_surface_conic_paraboloid_from_focal_distances(p,q,theta,alpha,"p")
print(prova.ccc_object.get_coefficients())

beam=prova.trace_Wolter_1(beam)

print(prova.ccc_object.get_coefficients())

beam.plot_xy()
beam.plot_xz()
print(np.mean(beam.x))
plt.show()











#beam=Beam()
#zou=100.
#beam.set_point(0.,0.,100.)
#
#p=1000.
#q=25+2*np.sqrt(2)
#theta=0*np.pi/180
#alpha=0
#
#hyp=Optical_element.initialize_my_hyperboloid(p,q,theta,alpha)
#
#beam=hyp.trace_optical_element(beam)
#
#beam.plot_xy()
#beam.plot_xz()
#plt.show()

#
#beam=Beam()
#beam.set_divergences_collimated()
#beam.set_point(350.,0.,350.)
#beam.set_rectangular_spot(5/2*1e-4,-5/2*1e-4,5/2*1e-4,-5/2*1e-4)
#
#
#p = 1000.
#q = 25.
#theta = 0 * np.pi / 180
#alpha = 0 * np.pi / 180
#
#prova = Optical_element.initialize_as_surface_conic_paraboloid_from_focal_distances(p,q,theta,alpha,"p")
#beam=prova.trace_optical_element(beam)
#print(prova.ccc_object.get_coefficients())
#beam.plot_xy()
#beam.plot_xz()
#print(np.mean(beam.z))
#
#beam.plot_xy()
#beam.plot_xz()
#plt.show()