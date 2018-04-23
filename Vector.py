import numpy as np
import matplotlib.pyplot as plt
from numpy import pi

def normalization3d(v):
    mod=np.sqrt(v[0]**2+v[1]**2+v[2]**2)
    if mod==0:
        v[0]=0
        v[1]=0
        v[2]=0
    else:
        v=v/mod
    return v


def moltiplication_beetween_matrix_and_array(a,b):
    (ra,ca)=a.shape
    c=np.zeros((ra))
    for i in range(0,ra):
        c[i]=np.dot(a[i,:],b)
    return c




class Vector(object):

    def __init__(self,x,y,z):

        self.x = x
        self.y = y
        self.z = z


    def rotation_x(self,alpha):
        x=self.x
        y=self.y
        z=self.z
        self.x = x
        self.y = y*np.cos(alpha)-z*np.sin(alpha)
        self.z = y*np.sin(alpha)+z*np.cos(alpha)

    def rotation_y(self,beta):
        x=self.x
        y=self.y
        z=self.z
        self.x = x*np.cos(beta)+z*np.sin(beta)
        self.y = y
        self.z = -x*np.sin(beta)+z*np.cos(beta)


    def rotation_z(self,gamma):
        x=self.x
        y=self.y
        z=self.z
        self.x = x*np.cos(gamma)-y*np.sin(gamma)
        self.y = x*np.sin(gamma)+y*np.cos(gamma)
        self.z = z





    def rotation(self,alpha,axis="x"):
        """
        rotate a vector an angle alpha
        :param alpha: rotation angle in degrees (counterclockwise)
        :param axis: "x", "y" or "z"
        :return:
        """

        angle=pi/180.0*alpha
        if axis == "x":
            self.rotation_x(angle)
        elif axis=="y":
            self.rotation_y(angle)
        elif axis=="z":
            self.rotation_z(angle)

    def info(self):
        return "x: %f, y: %f, z: %f\n"%(self.x,self.y,self.z)


