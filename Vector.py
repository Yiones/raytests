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

    def plane_normal(self):
        normal=Vector(self.x,self.y,self.z)
        normal.x=0*self.x
        normal.y=0*self.y
        normal.z=1*self.z

        return normal


    def spherical_normal(self,R):
        normal=Vector(self.x,self.y,self.z)
        normal.x=2*self.x
        normal.y=2*self.y
        normal.z=2*self.z-2*R

        return normal


    def normalization(self):
        mod=np.sqrt(self.x**2+self.y**2+self.z**2)
        self.x = self.x / mod
        self.y = self.y / mod
        self.z = self.z / mod

    def dot(self,v2):
        dot = np.array(self.x*v2.x+self.y*v2.y+self.z*v2.z)
        return dot

    def perpendicular_component(self,normal):
        a=-self.dot(normal)
        normal.x=normal.x*a
        normal.y=normal.y*a
        normal.z=normal.z*a
        return normal

    def sum(self,v2):
        sum = Vector(self.x,self.y,self.z)
        sum.x = self.x+v2.x
        sum.y = self.y+v2.y
        sum.z = self.z+v2.z

        return sum

    def after_reflection(self,v2):
        after_reflection = Vector(self.x,self.y,self.z)
        after_reflection = self.sum(v2)

        return after_reflection







#        for i in range(0,beam.N):
#            normal  = Vector(2*beam.x, 2*beam.y, 2*beam.z-2*self.R)
#            normal  = normalization3d(normal)
#            vector  = [beam.vx[i],beam.vy[i],beam.vz[i]]
#            vperp   = -np.dot(vector, normal)*normal
#            vparall = vector+vperp
#            [beam.vx[i], beam.vy[i], beam.vz[i]] = vperp+vparall





    def info(self):
        return "x: %f, y: %f, z: %f\n"%(self.x,self.y,self.z)


