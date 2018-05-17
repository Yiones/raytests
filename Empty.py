from Beam import Beam
from OpticalElement import Optical_element
from Shape import BoundaryRectangle
import numpy as np
from SurfaceConic import SurfaceConic
import matplotlib.pyplot as plt



#beam=Beam()
#beam.set_divergences_collimated()
#beam.set_point(100.,0.,100.)
##beam.set_rectangular_spot(55/2*1,-55/2*1,55/2*1,-55/2*1)
#beam.set_circular_spot(25)
#
#beam.plot_xz()
#
#p = 25.
#q = 25.
#theta = 0 * np.pi / 180
#alpha = 0 * np.pi / 180
#
#
#prova = Optical_element.initialize_as_surface_conic_paraboloid_from_focal_distances(p,q,theta,alpha,"p")
#
#beam=prova.trace_Wolter_1(beam)
#
#beam.plot_xy()
#beam.plot_xz()
#plt.show()
#





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
#beam.set_point(0.,0.,0.)
#beam.set_rectangular_spot(5/2*1e-4,-5/2*1e-4,5/2*1e-4,-5/2*1e-4)
#
#
#p = 10.
#q = 25.
#theta = 44 * np.pi / 180
#alpha = 90 * np.pi / 180
#
#prova = Optical_element.initialize_as_surface_conic_paraboloid_from_focal_distances(p,q,theta,alpha,"p")
#beam=prova.trace_optical_element(beam)
#print(prova.ccc_object.get_coefficients())
#
#beam.plot_xy()
#beam.plot_xz()
#plt.show()


#
#beam=Beam()
#beam.set_divergences_collimated()
#beam.set_point(15.,0.,15.)
#beam.set_rectangular_spot(5/2*1e-1,-5/2*1e-6,5/2*1e-1,-5/2*1e-1)
#
#
#p = 2000.
#q = 1.
#theta = 0 * np.pi / 180
#alpha = 0 * np.pi / 180
#
#
#prova = Optical_element.initialize_as_surface_conic_paraboloid_from_focal_distances(p,q,theta,alpha,"p")
#print(prova.ccc_object.get_coefficients())
#
#beam=prova.trace_Wolter_2(beam)
#
#print(np.mean(beam.z))
#beam.plot_xy()
#beam.plot_xz()
#plt.show()



beam1=Beam()
beam1.set_divergences_collimated()
beam1.set_point(0.+100,0.,20.+100)
beam1.set_circular_spot(5.)

beam2=Beam()
beam2.set_divergences_collimated()
beam2.set_point(0.+100,0.,0.+100)
beam2.set_rectangular_spot(20.,-20.,15.,10.)


beam=beam1.merge(beam2)

beam3=Beam()
beam3.set_divergences_collimated()
beam3.set_point(0.+100,0.,0.+100)
beam3.set_rectangular_spot(5.,-5.,10.,-40.)

beam=beam.merge(beam3)
beamd=beam.duplicate()


p = 10.
q = 25.
theta = 44 * np.pi / 180
alpha = 90 * np.pi / 180
prova = Optical_element.initialize_as_surface_conic_paraboloid_from_focal_distances(p,q,theta,alpha,"p")
beamd=prova.trace_optical_element(beamd)
print(prova.ccc_object.get_coefficients())

t = (100-beam.y)/beam.vy

beamd.x = beamd.x + beamd.vx * t
beamd.y = beamd.y + beamd.vy * t
beamd.z = beamd.z + beamd.vz * t


beamd.plot_xz()


beam.plot_xz()
beam.plot_xpzp()


p = 1000.
q = 25.
theta = 0 * np.pi / 180
alpha = 0 * np.pi / 180
prova = Optical_element.initialize_as_surface_conic_paraboloid_from_focal_distances(p,q,theta,alpha,"p")
beam=prova.trace_Wolter_1(beam)
beam.plot_xy()
beam.plot_xz()

plt.show()