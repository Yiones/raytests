from Vector import Vector
import numpy as np


x=0.
y=0.
z=20.

### rotation   #####

position=Vector(x,y,z)
position.rotation(-90*np.pi/180)
(x,y,z)=(position.x,position.y,position.z)

velocity=Vector(0.,20.+2*np.sqrt(2),-20.)
velocity.normalization()
velocity.rotation(-90*np.pi/180)
vx=velocity.x
vy=velocity.y
vz=velocity.z

print("We have an initial velocity of")
print(vx,vy,vz)

### translation   #####

p=Vector(0.,20.+np.sqrt(2),0.)
p.rotation(-90*np.pi/180)
x=x-p.x
y=y-p.y
z=z-p.z

#print(x,y,z)

### intersection   #####

a=-vx**2-vy**2+vz**2
b=(-x*vx-y*vy+z*vz)
c=-x**2-y**2+z**2-1

print("a, b, c are respectively:\n%f   %f    %f" % (a, b, c))

if a!=0:
    t=(-b-np.sqrt(b**2-a*c))/a
else:
    t=-c/b
print("The time of flight is:      %f" %(t))


x = x + vx*t
y = y + vy*t
z = z + vz*t

#print(x,y,z)

### output direction ###

position=Vector(x,y,z)
velocity=Vector(vx,vy,vz)

print(velocity.info())

normal = Vector(-2*x,-2*y, 2*z)
normal.normalization()
vpep = velocity.perpendicular_component(normal)
v2=velocity.sum(vpep)
v2=v2.sum(vpep)

print(v2.info())

(vx,vy,vz)=(v2.x,v2.y,v2.z)

### rotation to the screen ###

position=Vector(x,y,z)
velocity=Vector(vx,vy,vz)
position.rotation(-(90*np.pi/180),"x")
velocity.rotation(-(90*np.pi/180),"x")
(x,y,z)=(position.x,position.y,position.z)
(vx,vy,vz)=(velocity.x,velocity.y,velocity.z)

### translation and intersection to the screen ###

y=y-np.sqrt(2)

t=-y/vy

x = x + vx*t
y = y + vy*t
z = z + vz*t


print(x,y,z)