import numpy as np
from Vector import Vector
from OpticalElement import Optical_element
from SurfaceConic import SurfaceConic
from Beam import Beam
import matplotlib.pyplot as plt
from Shape import BoundaryRectangle


######################   ellipse     ###################################################################################


p = 13.4
q = 0.67041707
theta = 88.8 * np.pi/180

a = (p+q)/2
b = np.sqrt(p*q)*np.sin(np.pi/2-theta)
f = np.sqrt(a**2-b**2)

beta = np.arccos((p**2+4*f**2-q**2)/(4*p*f))

ccc1 = np.array([1./b**2, 1./b**2, 1/a**2, 0., 0., 0., 0., 0., 0., -1])

y = - p * np.sin(beta)
z = f - p * np.cos(beta)

print("beta = %f, y = %f, z = %f"  %(beta,y,z))

oe1 = Optical_element.initialize_as_surface_conic_from_coefficients(ccc1)


######################   hyperbole   ###################################################################################

d = 0.1082882
qf = 0.300
theta1 = 89. * np.pi/180

p1 = q - d
q1 = qf

ah = (p1 - q1)/2
bh = np.sqrt(p1*q1)*np.cos(theta1)
z0 = np.sqrt(a**2 - b**2) - np.sqrt(ah**2 - bh**2)
print("f-z0 = %f" %(np.sqrt(a**2-b**2)-z0))

ccc2 = np.array([-1./ah**2, -1./ah**2, 1/bh**2, 0., 0., 0., 0., 0., 2*z0/bh**2, z0**2/bh**2-1])

oe2 = Optical_element.initialize_as_surface_conic_from_coefficients(ccc2)


v = Vector(0., y/p, (z-np.sqrt(a**2-b**2))/p)
v.normalization()


beam2 = Beam(1)

beam2.z += f

beam2.vx = v.x
beam2.vy = v.y
beam2.vz = v.z

oe1.intersection_with_optical_element(beam2)
oe1.output_direction_from_optical_element(beam2)
by1 = beam2.y
bz1 = beam2.z
oe2.intersection_with_optical_element(beam2)
by2 = beam2.y
bz2 = beam2.z

bound1 = BoundaryRectangle(xmax=0.0075, xmin=-0.0075, ymax=10000, ymin=-10000, zmax=bz1+0.5, zmin=bz2-0.2)
oe1.set_bound(bound1)

bound2 = BoundaryRectangle(xmax=0.0075, xmin=-0.0075, ymax=10000, ymin=-10000, zmax=bz2+0.5, zmin=bz2-0.2)
oe2.set_bound(bound2)

####################   Beam generation    ##############################################################################


beam = Beam()
beam.set_gaussian_divergence(5*1e-5,0.00025)
beam.set_rectangular_spot( xmax=200*1e-6, xmin=-200*1e-6, zmax=10*1e-6, zmin=-10*1e-6)

beam.plot_xz()
beam.plot_xpzp(0)


print("vector v")
print(v.info())


v0 = Vector(0., 0., -1.)
alpha = np.arccos(v.dot(v0))
v0.rotation(-alpha,'x')

velocity = Vector(beam.vx, beam.vz, -beam.vy)
print("velocity")
print(np.mean(velocity.x), np.mean(velocity.y), np.mean(velocity.z))
velocity.rotation(-alpha, 'x')

print(np.mean(velocity.x), np.mean(velocity.y), np.mean(velocity.z))
print("\n")
beam.z += f


beam.vx = velocity.x
beam.vy = velocity.y
beam.vz = velocity.z





#################   Beam propagation  ##################################################################################


oe1.intersection_with_optical_element(beam)
print(np.mean(beam.x),np.mean(beam.y), np.mean(beam.z))
oe1.output_direction_from_optical_element(beam)
oe2.intersection_with_optical_element(beam)
print(np.mean(beam.x),np.mean(beam.y), np.mean(beam.z))

oe2.output_direction_from_optical_element(beam)

print("ellipse\na = %f, b = %f\nhyperbola\na = %f, b = %f, z0 = %f" %(a, b, ah, bh, z0))


#t = - beam.y / beam.vy
t = (-np.sqrt(ah**2+bh**2)-z0-beam.z)/beam.vz


beam.x += beam.vx * t
beam.y += beam.vy * t
beam.z += beam.vz * t

beam.x *= 1e6
beam.y *= 1e6
beam.z *= 1e6

beam.plot_yx(0)
beam.plot_good_yx(0)


print(beam.flag, np.size(np.where(beam.flag<0)))

print(np.mean(beam.y), np.mean(beam.z))
print("focus = %f" %(-np.sqrt(ah**2+bh**2)-z0))
print("dz = %f" %(max(beam.z)-min(beam.z)))


plt.show()