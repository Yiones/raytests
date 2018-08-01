import numpy as np
from Vector import Vector
from OpticalElement import Optical_element
from SurfaceConic import SurfaceConic
from Beam import Beam
import matplotlib.pyplot as plt
from Shape import BoundaryRectangle
from CompoundOpticalElement import CompoundOpticalElement


def shadow_source():
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

    oe0.FDISTR = 3
    oe0.FSOUR = 1
    oe0.F_PHOT = 0
    oe0.HDIV1 = 5.9999998e-05
    oe0.HDIV2 = 5.9999998e-05
    oe0.IDO_VX = 0
    oe0.IDO_VZ = 0
    oe0.IDO_X_S = 0
    oe0.IDO_Y_S = 0
    oe0.IDO_Z_S = 0
    oe0.NPOINT = 25000
    oe0.PH1 = 1000.0
    oe0.SIGDIX = 4.6699999e-05
    oe0.SIGDIZ = 0.00025499999
    oe0.VDIV1 = 0.00028499999
    oe0.VDIV2 = 0.00028499999
    oe0.WXSOU = 0.04
    oe0.WZSOU = 0.002

    # Run SHADOW to create the source

    if iwrite:
        oe0.write("start.00")

    beam.genSource(oe0)

    if iwrite:
        oe0.write("end.00")
        beam.write("begin.dat")

    return beam

#######################   ellipse     ###################################################################################
#
#
#p = 13.4
#q = 0.67041707
#theta = 88.8 * np.pi/180
#
#ae = (p+q)/2
#be = np.sqrt(p*q)*np.cos(theta)
#print("b of sin = %f, b of cos = %f" %(be, np.sqrt(p*q)*np.cos(theta)))
#f = np.sqrt(ae**2-be**2)
#
#beta = np.arccos((p**2+4*f**2-q**2)/(4*p*f))
#
#ccc1 = np.array([1./be**2, 1./be**2, 1/ae**2, 0., 0., 0., 0., 0., 0., -1])
#
#y = - p * np.sin(beta)
#z = f - p * np.cos(beta)
#
#print("beta = %f, y = %f, z = %f"  %(beta,y,z))
#print(y**2/be**2+z**2/ae**2)
#oe1 = Optical_element.initialize_as_surface_conic_from_coefficients(ccc1)
#
#
#######################   hyperbole   ###################################################################################
#
#d = 0.1082882
#qf = 0.300
#theta1 = 89. * np.pi/180
#
#p1 = q - d
#q1 = qf
#
#print("p1 = %f, q1 = %f" %(p1,q1))
#
#ah = (p1 - q1)/2
#bh = np.sqrt(p1*q1)*np.cos(theta1)
#z0 = np.sqrt(ae**2 - be**2) - np.sqrt(ah**2 + bh**2)
#print("f-z0 = %f" %(np.sqrt(ae**2-be**2)-z0))
#
#ccc2 = np.array([-1./ah**2, -1./ah**2, 1/bh**2, 0., 0., 0., 0., 0., 2*z0/bh**2, z0**2/bh**2-1])
#
#oe2 = Optical_element.initialize_as_surface_conic_from_coefficients(ccc2)
#
#
#v = Vector(0., y/p, (z-np.sqrt(ae**2-be**2))/p)
#v.normalization()
#
#print("velocity: v.y = %f, v.z = %f" %(v.y, v.z))
#
#
#beam2 = Beam(1)
#
#beam2.z += f
#
#beam2.vx = v.x
#beam2.vy = v.y
#beam2.vz = v.z
#
#print(beam2.z, beam2.vx, beam2.vy, beam2.vz)
#
#oe1.intersection_with_optical_element(beam2)
#
#normal = Vector(0., 2*y/be**2, 2*z/ae**2)
#normal.normalization()
#vel_vec = Vector(beam2.vx, beam2.vy, beam2.vz)
#print(normal.dot(vel_vec))
#print("angle = %f"  %(np.arccos(normal.dot(vel_vec))*180/np.pi))
#
#oe1.output_direction_from_optical_element(beam2)
#by1 = beam2.y
#bz1 = beam2.z
#oe2.intersection_with_optical_element(beam2)
#by2 = beam2.y
#bz2 = beam2.z
#
#
#bound1 = BoundaryRectangle(xmax=0.0075, xmin=-0.0075, ymax=10000, ymin=-10000, zmax=bz1+0.5, zmin=bz2-0.2)
##oe1.set_bound(bound1)
#
#bound2 = BoundaryRectangle(xmax=0.0075, xmin=-0.0075, ymax=10000, ymin=-10000, zmax=bz2+0.5, zmin=bz2-0.2)
##oe2.set_bound(bound2)
#
#####################   Beam generation    ##############################################################################
#
#shadow_beam = shadow_source()
#
#beam = Beam()
#beam.set_gaussian_divergence(5*1e-5,0.00025)
#beam.set_rectangular_spot( xmax=200*1e-6, xmin=-200*1e-6, zmax=10*1e-6, zmin=-10*1e-6)
#
#beam = Beam(25000)
#beam.initialize_from_arrays(
#    shadow_beam.getshonecol(1),
#    shadow_beam.getshonecol(3),
#    shadow_beam.getshonecol(2),
#    shadow_beam.getshonecol(4),
#    shadow_beam.getshonecol(5),
#    shadow_beam.getshonecol(6),
#    shadow_beam.getshonecol(10),
#    0
#)
#
#beam.z = - beam.z + f
#beam.x *= 1e-2*1e6
#beam.y *= 1e-2*1e6
#
#beam.plot_yx(0)
#
#
#beam.x /= 1e6
#beam.y /= 1e6
#
#print("vector v")
#print(v.info())
#
#
#v0 = Vector(0., 0., -1.)
#alpha = np.arccos(v.dot(v0))
#v0.rotation(-alpha,'x')
#
#velocity = Vector(beam.vx, beam.vz, -beam.vy)
#print("velocity")
#print(np.mean(velocity.x), np.mean(velocity.y), np.mean(velocity.z))
#velocity.rotation(-alpha, 'x')
#
#print(np.mean(velocity.x), np.mean(velocity.y), np.mean(velocity.z))
#print("\n")
#
#
#beam.vx = velocity.x
#beam.vy = velocity.y
#beam.vz = velocity.z
#
#
#
#
#
##################   Beam propagation  ##################################################################################
#
#
#oe1.intersection_with_optical_element(beam)
#print(np.mean(beam.x),np.mean(beam.y), np.mean(beam.z))
#oe1.output_direction_from_optical_element(beam)
#oe2.intersection_with_optical_element(beam)
#print(np.mean(beam.x),np.mean(beam.y), np.mean(beam.z))
#
#oe2.output_direction_from_optical_element(beam)
#
#print("ellipse\na = %f, b = %f\nhyperbola\na = %f, b = %f, z0 = %f" %(ae, be, ah, bh, z0))
#
#
##t = - beam.y / beam.vy
#
#Nn = 1000
#qqq = np.ones(Nn)
#dy = np.ones(Nn)
#
#for i in range (0,Nn):
#
#    qqq[i] = np.sqrt(ah**2+bh**2)- z0 - 0.0001 + 0.0002 * i / Nn
#
#    t = (qqq[i]-beam.z)/beam.vz
#
#
#    beam.x += beam.vx * t
#    beam.y += beam.vy * t
#    beam.z += beam.vz * t
#
#
#    dy[i] = max(beam.y)-min(beam.y)
#
#
#    beam.x -= beam.vx * t
#    beam.y -= beam.vy * t
#    beam.z -= beam.vz * t
#
#
#plt.figure()
#plt.plot(qqq, dy, 'b.')
#
#qq = qqq[np.where(dy==min(dy))]
#t = (qq-beam.z) / beam.vz
#
#beam.x += beam.vx * t
#beam.y += beam.vy * t
#beam.z += beam.vz * t
#
#beam.x *= 1e6
#beam.y *= 1e6
#
#beam.plot_yx(0)
#
#print("-z0+fh = %f, qq = %f" %(-z0+np.sqrt(ah**2+bh**2), qq))
#
#plt.show()


#print(beam.flag, np.size(np.where(beam.flag<0)))
#
#print(np.mean(beam.y), np.mean(beam.z))
#print("focus = %f" %(-np.sqrt(ah**2+bh**2)-z0))
#
#print("dx = %f" %(max(beam.x)-min(beam.x)))
#print("dy = %f" %(max(beam.y)-min(beam.y)))
#
#oe1.output_frame_wolter(beam)
#
#by = beam.y
#bz = beam.z
#
#beam.y = bz
#beam.z = by
#
#beam.retrace(100)
#beam.plot_xz()
#
#plt.show()

#######################################################################################################################

p = 13.4
q =  0.300
d = 0.1082882
q1 = 0.67041707
theta1 = 88.8*np.pi/180
theta2 = 89.*np.pi/180




wolter_jap = CompoundOpticalElement.wolter_for_japanese(p=p, q=q, d=d, q1=q1, theta1=theta1, theta2=theta2)

shadow_beam = shadow_source()
beam = Beam()
beam.set_gaussian_divergence(5*1e-5,0.00025)
beam.set_rectangular_spot( xmax=200*1e-6, xmin=-200*1e-6, zmax=10*1e-6, zmin=-10*1e-6)
beam = Beam(25000)
beam.initialize_from_arrays(
    shadow_beam.getshonecol(1),
    shadow_beam.getshonecol(3),
    shadow_beam.getshonecol(2),
    shadow_beam.getshonecol(4),
    shadow_beam.getshonecol(5),
    shadow_beam.getshonecol(6),
    shadow_beam.getshonecol(10),
    0
)

beam.z = - beam.z + np.sqrt(7.035209**2-0.062770**2)
beam.x *= 1e-2
beam.y *= 1e-2

beam.plot_yx(0)
beam.plot_ypzp(0)


wolter_jap.velocity_wolter_japanes(beam)

print("velocity: vy = %f, vz = %f" %(np.mean(beam.vy), np.mean(beam.vz)))


#################   Beam propagation  ##################################################################################


wolter_jap.oe[0].intersection_with_optical_element(beam)
print(np.mean(beam.x),np.mean(beam.y), np.mean(beam.z))
wolter_jap.oe[0].output_direction_from_optical_element(beam)
wolter_jap.oe[1].intersection_with_optical_element(beam)
print(np.mean(beam.x),np.mean(beam.y), np.mean(beam.z))

wolter_jap.oe[1].output_direction_from_optical_element(beam)


ccc= wolter_jap.oe[1].ccc_object.get_coefficients()
print(ccc)
print("Sciau belu")
ah = (-ccc[0])**-0.5
bh = ccc[2]**-0.5
print("ah = %f, bh = %f" %(ah, bh))
#t = - beam.y / beam.vy

t = (np.sqrt(ah**2+bh**2)-6.903668-beam.z)/beam.vz

beam.x += beam.vx * t
beam.y += beam.vy * t
beam.z += beam.vz * t

beam.x *= 1e6
beam.y *= 1e6

beam.plot_yx(0)


print(beam.flag, np.size(np.where(beam.flag<0)))

print(np.mean(beam.y), np.mean(beam.z))

print("dx = %f" %(max(beam.x)-min(beam.x)))
print("dy = %f" %(max(beam.y)-min(beam.y)))

#wolter_jap.oe[0].output_frame_wolter(beam)
#
#by = beam.y
#bz = beam.z
#
#beam.y = bz
#beam.z = by
#
#beam.retrace(100)
#beam.plot_xz()

plt.show()

