from Beam import Beam
from OpticalElement import Optical_element
from Shape import BoundaryRectangle
import numpy as np
import matplotlib.pyplot as plt



beam1 = Beam()
beam1.set_point(0, 0, 0)
beam1.set_flat_divergence(5e-3, 5e-2)
p=2.
q=1.
theta=0*np.pi/180

bound1=BoundaryRectangle(xmax=0.005,xmin=-0.005,ymax=0.05,ymin=-0.05)
bound2=BoundaryRectangle(xmax=0.01,xmin=-0.01,ymax=0.1,ymin=-0.1)

spherical_mirror=Optical_element.initialize_as_plane_mirror(2,1,theta,0)
plane_mirror=Optical_element.initialize_as_surface_conic_ellipsoid_from_focal_distances(1,1,28*np.pi/180)


beam1=spherical_mirror.trace_optical_element(beam1,bound1)
beam1=plane_mirror.trace_optical_element(beam1,bound2)

beam1.plot_xz()
plt.title("Total points plot")
beam1.plot_good_xz()
plt.title("Good points plot")

print(beam1.flag)

plt.show()



