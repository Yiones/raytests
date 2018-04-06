#
#
#
import numpy as np
import matplotlib.pyplot as plt


def creation_source_point_uniform (dx,dz,N=10000,x0=0,y0=0,z0=0):
    x=np.zeros(N)
    y=np.zeros(N)
    z=np.zeros(N)
    
    vx=dx*(np.random.random(N)-0.5)*2
    vz=dz*(np.random.random(N)-0.5)*2
    vy=np.random.random(N)
    for i in range(0,N):
        vy[i]=np.sqrt(1-vx[i]**2-vz[i]**2)
    return np.array([x,y,z,vx,vy,vz])


def creation_source_gaussian(sx,sz,N=10000,x0=0,y0=0,z0=0):
    x=np.zeros(N)
    y=np.zeros(N)
    z=np.zeros(N)
    
    vx=sx*(np.random.randn(N))*2
    vz=sz*(np.random.randn(N))*2
    vy=np.random.random(N)
    for i in range(0,N):
        vy[i]=np.sqrt(1-vx[i]**2-vz[i]**2)
    return np.array([x,y,z,vx,vy,vz])


def free_propagation(beam,p):
    N=beam.size/6
    print(N)
    for i in range (0,N):
        beam[0,i]=beam[3,i]*beam[4,i]/p
        beam[2,i]=beam[5,i]*beam[4,i]/p
        beam[1,i]=p
    return beam


beam0=creation_source_gaussian(3e-3/2.35,5e-6/2.35)
beam1=creation_source_point_uniform(3e-3,5e-6)
beam10=free_propagation(beam0,10)
beam11=free_propagation(beam1,10)

plt.figure(1)
plt.plot(beam10[0,:],beam10[2,:], 'ro')
plt.plot(beam11[0,:],beam11[2,:], 'bo')
plt.xlabel('x axis')
plt.ylabel('z axis')
plt.title('xz plot at p=10')

plt.figure(2)
plt.hist(beam10[0,:])
plt.title('x position for gaussian distribution')

plt.figure(3)
plt.hist(beam11[0,:])
plt.title('x position for uniform distribution')
plt.show()

print(beam11)

