from Beam import Beam
from OpticalElement import Optical_element
from Shape import BoundaryRectangle
import numpy as np
import matplotlib.pyplot as plt
from numpy.testing import assert_almost_equal


beam=Beam()
beam.set_flat_divergence(0.005,0.0005)
p=20.
q=0.
spherical_mirror=Optical_element.initialize_as_spherical_mirror(p,q,theta=0,alpha=0,R=20)
beam=spherical_mirror.trace_optical_element(beam)
#t=20/beam.vy
#beam.x = beam.x + beam.vx * t
#beam.y = beam.y + beam.vy * t
#beam.z = beam.z + beam.vz * t
p=20.-np.sqrt(2)
q=np.sqrt(2)
theta=0*np.pi/180
hyp_mirror=Optical_element.initialize_my_hyperboloid(p,q,theta)
beam=hyp_mirror.trace_optical_element(beam)
#print (hyp_mirror.ccc_object.get_coefficients())
print(beam.x,beam.y,beam.z)
beam.plot_xz()
plt.show()