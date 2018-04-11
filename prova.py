import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

def moltiplication_beetween_2_matrix(a,b):
    (ra,ca)=a.shape
    (rb,cb)=b.shape
    c=np.zeros((ra,cb))
    for i in range(0,ra):
        for j in range(0, cb):
            c[i,j]=np.dot(a[i,:],np.transpose(b[:,j]))
    return c

def moltiplication_beetween_matrix_and_array(a,b):
    (ra,ca)=a.shape
    c=np.zeros((ra))
    for i in range(0,ra):
        c[i]=np.dot(a[i,:],b)
    return c

def rotation(beam,tetha,alpha):         #rotation along x axis of an angle 1-alpha, and along y with an angle tetha
    tetha=pi/180.0*tetha
    alpha=pi/180.0*alpha
    beta=pi-tetha
    sinalpha=np.sin(alpha)
    cosalpha=np.cos(alpha)
    sinbeta=np.sin(beta)
    cosbeta=np.cos(beta)
    m2=np.array([[cosalpha,0,sinalpha],[0,1,0],[-sinalpha,0,cosalpha]])
    m1=np.array([[1,0,0],[0,cosbeta,-sinbeta],[0,sinbeta,cosbeta]])
    for i in range(0,3):
        [beam[0,i],beam[1,i],beam[2,i]]=moltiplication_beetween_matrix_and_array(m1,[beam[0,i],beam[1,i],beam[2,i]])
        [beam[0, i], beam[1, i], beam[2, i]]=moltiplication_beetween_matrix_and_array(m2,[beam[0,i],beam[1,i],beam[2,i]])
    return

def translation(beam,p):
    beam[0,:]=beam[0,:]-p[0]
    beam[1,:]=beam[1,:]-p[1]
    beam[2,:]=beam[2,:]-p[2]
    return





beam=np.array([[0.0,0.0,1.0],[0.0,1.0,0.0],[1.0,0.0,0.0]])
print beam


rotation(beam,45,0)
print("After rotation")
print beam
translation(beam,[0,5,0])

print("After translation")
print beam

m1=np.array([[1,0,0],[0,-2,-2],[0,2,-2]])
p=[0,0,1]
