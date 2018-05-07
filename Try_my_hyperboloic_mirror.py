from Beam import Beam
from OpticalElement import Optical_element
from Shape import BoundaryRectangle
import numpy as np
import matplotlib.pyplot as plt
from numpy.testing import assert_almost_equal


beam=Beam(1)
#beam.set_flat_divergence(0.000005,0.001)
beam.set_point(0,0,20+np.sqrt(2))

p=20.
q=np.sqrt(2)
theta=0*np.pi/180

hyp_mirror=Optical_element.initialize_my_hyperboloid(p,q,theta)
#ideal_lens=Optical_element()
#ideal_lens.set_parameters(10.+np.sqrt(2),10.+np.sqrt(2))


beam.plot_xz()
#print(beam.x,beam.vx)
#beam=ideal_lens.trace_ideal_lens(beam)
#print(beam.x,beam.vx)
#

beam=hyp_mirror.trace_optical_element(beam)


#t = (20.+np.sqrt(2)) / beam.vy
#beam.x = beam.x + beam.vx * t
#beam.y = beam.y + beam.vy * t
#beam.z = beam.z + beam.vz * t


beam.plot_xz()
print (hyp_mirror.ccc_object.get_coefficients())

print(beam.x,beam.y,beam.z)

#beam.histogram()

plt.show()