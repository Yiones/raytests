
import numpy as np
import matplotlib.pyplot as plt
from Vector import Vector


class Optical_element(object):

    def __init__(self,ccc=np.zeros(10)):
        self.p=0
        self.q=0
        self.theta=0
        self.alpha=0
        self.R=0
        self.type=""

        if ccc is not None:
            self.ccc = ccc.copy()
        else:
            self.ccc = np.zeros(10)


    def set_parameters(self,p,q,theta,alpha,R=0):
            self.p = p
            self.q = q
            self.theta = theta
            self.alpha = alpha
            self.R=R



    @classmethod
    def initialize_as_plane_mirror(cls, p, q,theta, alpha):

        plane_mirror=Optical_element()
        plane_mirror.p=p
        plane_mirror.q=q
        plane_mirror.theta=theta
        plane_mirror.alpha=alpha
        plane_mirror.type="Plane mirror"

        return plane_mirror


    @classmethod
    def initialize_as_spherical_mirror(cls, p, q,theta, alpha, R):

        spherical_mirror=Optical_element()
        spherical_mirror.p=p
        spherical_mirror.q=q
        spherical_mirror.theta=theta
        spherical_mirror.alpha=alpha
        spherical_mirror.R=R
        spherical_mirror.type="Spherical mirror"

        return spherical_mirror


    @classmethod
    def initialize_from_coefficients(cls, ccc):
        if np.array(ccc).size != 10:
            raise Exception("Invalid coefficients (dimension must be 10)")
        return Optical_element(ccc=ccc)

    @classmethod
    def initialize_as_plane(cls):
        return Optical_element(np.array([0, 0, 0, 0, 0, 0, 0, 0, -1., 0]))
    #
    # initializers from focal distances
    #

    @classmethod
    def initialize_as_sphere_from_focal_distances(cls, p, q, theta1, cylindrical=0, cylangle=0.0,
                                                  switch_convexity=0):
        ccc = Optical_element()
        ccc.set_sphere_from_focal_distances(p, q, theta1)
        if cylindrical:
            ccc.set_cylindrical(cylangle)
        if switch_convexity:
            ccc.switch_convexity()
        return ccc


    @classmethod
    def initialize_as_ellipsoid_from_focal_distances(cls, p, q, theta1, cylindrical=0, cylangle=0.0,
                                                     switch_convexity=0):
        ccc = Optical_element()
        ccc.set_ellipsoid_from_focal_distances(p, q, theta1)
        if cylindrical:
            ccc.set_cylindrical(cylangle)
        if switch_convexity:
            ccc.switch_convexity()
        return ccc


    @classmethod
    def initialize_as_paraboloid_from_focal_distances(cls, p, q, theta1, cylindrical=0, cylangle=0.0,
                                                      switch_convexity=0):
        ccc = Optical_element()
        ccc.set_paraboloid_from_focal_distances(p, q, theta1)
        if cylindrical:
            ccc.set_cylindrical(cylangle)
        if switch_convexity:
            ccc.switch_convexity()
        return ccc


    @classmethod
    def initialize_as_hyperboloid_from_focal_distances(cls, p, q, theta1, cylindrical=0, cylangle=0.0,
                                                       switch_convexity=0):
        ccc = Optical_element()
        ccc.set_hyperboloid_from_focal_distances(p, q, theta1)
        if cylindrical:
            ccc.set_cylindrical(cylangle)
        if switch_convexity:
            ccc.switch_convexity()
        return ccc
    #
    # initializars from surface parameters
    #


    @classmethod
    def initialize_as_sphere_from_curvature_radius(cls, radius, cylindrical=0, cylangle=0.0, switch_convexity=0):
        ccc = Optical_element()
        ccc.set_sphere_from_curvature_radius(radius)
        if cylindrical:
            ccc.set_cylindrical(cylangle)
        if switch_convexity:
            ccc.switch_convexity()
        return ccc


    def set_cylindrical(self, CIL_ANG):
        COS_CIL = np.cos(CIL_ANG)
        SIN_CIL = np.sin(CIL_ANG)
        A_1 = self.ccc[1 - 1]
        A_2 = self.ccc[2 - 1]
        A_3 = self.ccc[3 - 1]
        A_4 = self.ccc[4 - 1]
        A_5 = self.ccc[5 - 1]
        A_6 = self.ccc[6 - 1]
        A_7 = self.ccc[7 - 1]
        A_8 = self.ccc[8 - 1]
        A_9 = self.ccc[9 - 1]
        A_10 = self.ccc[10 - 1]
        self.ccc[1 - 1] = A_1 * SIN_CIL ** 4 + A_2 * COS_CIL ** 2 * SIN_CIL ** 2 - A_4 * COS_CIL * SIN_CIL ** 3
        self.ccc[2 - 1] = A_2 * COS_CIL ** 4 + A_1 * COS_CIL ** 2 * SIN_CIL ** 2 - A_4 * COS_CIL ** 3 * SIN_CIL
        self.ccc[3 - 1] = A_3  # Z^2
        self.ccc[
            4 - 1] = - 2 * A_1 * COS_CIL * SIN_CIL ** 3 - 2 * A_2 * COS_CIL ** 3 * SIN_CIL + 2 * A_4 * COS_CIL ** 2 * SIN_CIL ** 2  # X Y
        self.ccc[5 - 1] = A_5 * COS_CIL ** 2 - A_6 * COS_CIL * SIN_CIL  # Y Z
        self.ccc[6 - 1] = A_6 * SIN_CIL ** 2 - A_5 * COS_CIL * SIN_CIL  # X Z
        self.ccc[7 - 1] = A_7 * SIN_CIL ** 2 - A_8 * COS_CIL * SIN_CIL  # X
        self.ccc[8 - 1] = A_8 * COS_CIL ** 2 - A_7 * COS_CIL * SIN_CIL  # Y
        self.ccc[9 - 1] = A_9  # Z
        self.ccc[10 - 1] = A_10


    def switch_convexity(self):
        self.ccc[5 - 1] = - self.ccc[5 - 1]
        self.ccc[6 - 1] = - self.ccc[6 - 1]
        self.ccc[9 - 1] = - self.ccc[9 - 1]


    def set_sphere_from_focal_distances(self, ssour, simag, theta):  # ssour = p, simag = q
        self.p=ssour
        self.q=simag
        self.theta=theta

        # todo: implement also sagittal bending
        print('>>>> set_sphere_from_focal_distances: Angle with respect to the surface normal [rad]:', theta)
        rmirr = ssour * simag * 2 / np.cos(theta) / (ssour + simag)
        self.ccc[1 - 1] = 1.0  # X^2  # = 0 in cylinder case
        self.ccc[2 - 1] = 1.0  # Y^2
        self.ccc[3 - 1] = 1.0  # Z^2
        self.ccc[4 - 1] = .0  # X*Y   # = 0 in cylinder case
        self.ccc[5 - 1] = .0  # Y*Z
        self.ccc[6 - 1] = .0  # X*Z   # = 0 in cylinder case
        self.ccc[7 - 1] = .0  # X     # = 0 in cylinder case
        self.ccc[8 - 1] = .0  # Y
        self.ccc[9 - 1] = -2 * rmirr  # Z
        self.ccc[10 - 1] = .0  # G
        print(self.ccc)
        print(">>>> set_sphere_from_focal_distances: Spherical radius: %f \n" % (rmirr))


    def set_sphere_from_curvature_radius(self, rmirr):
        self.ccc[1 - 1] = 1.0  # X^2  # = 0 in cylinder case
        self.ccc[2 - 1] = 1.0  # Y^2
        self.ccc[3 - 1] = 1.0  # Z^2
        self.ccc[4 - 1] = .0  # X*Y   # = 0 in cylinder case
        self.ccc[5 - 1] = .0  # Y*Z
        self.ccc[6 - 1] = .0  # X*Z   # = 0 in cylinder case
        self.ccc[7 - 1] = .0  # X     # = 0 in cylinder case
        self.ccc[8 - 1] = .0  # Y
        self.ccc[9 - 1] = -2 * rmirr  # Z
        self.ccc[10 - 1] = .0  # G


    def set_ellipsoid_from_focal_distances(self, ssour, simag, theta):
        self.p=ssour
        self.q=simag
        self.theta=theta

        COSTHE = np.cos(theta)
        SINTHE = np.sin(theta)
        AXMAJ = (ssour + simag) / 2
        AXMIN = np.sqrt(simag * ssour) * COSTHE
        AFOCI = np.sqrt(AXMAJ ** 2 - AXMIN ** 2)
        ECCENT = AFOCI / AXMAJ
        # ;C
        # ;C The center is computed on the basis of the object and image positions
        # ;C
        YCEN = (ssour - simag) * 0.5 / ECCENT
        ZCEN = -np.sqrt(1 - YCEN ** 2 / AXMAJ ** 2) * AXMIN
        # ;C
        # ;C Computes now the normal in the mirror center.
        # ;C
        RNCEN = np.zeros(3)
        RNCEN[1 - 1] = 0.0
        RNCEN[2 - 1] = -2 * YCEN / AXMAJ ** 2
        RNCEN[3 - 1] = -2 * ZCEN / AXMIN ** 2
        # ;CALL NORM(RNCEN,RNCEN)
        RNCEN = RNCEN / np.sqrt((RNCEN ** 2).sum())
        # ;C
        # ;C Computes the tangent versor in the mirror center.
        # ;C
        RTCEN = np.zeros(3)
        RTCEN[1 - 1] = 0.0
        RTCEN[2 - 1] = RNCEN[3 - 1]
        RTCEN[3 - 1] = -RNCEN[2 - 1]
        # txt = [txt,  $
        # String('Rev Ellipsoid a: ', $
        # AXMAJ, Format='(A40,G20.15)'), $
        # String('Rev Ellipsoid b: ', $
        # AXMIN, Format='(A40,G20.15)'), $
        # String('Rev Ellipsoid c=sqrt(a^2-b^2): ', $
        # AFOCI, Format='(A40,G20.15)'), $
        # String('Rev Ellipsoid focal discance c^2: ', $
        # AFOCI^2, Format='(A40,G20.15)'), $
        # String('Rev Ellipsoid excentricity: ', $
        # ECCENT, Format='(A40,G20.15)'),$
        # 'Mirror center at: '+vect2string([0,YCEN,ZCEN]), $
        # 'Mirror normal: '+vect2string(RNCEN), $
        # 'Mirror tangent: '+vect2string(RTCEN) ]
        # ;C Computes now the quadric coefficient with the mirror center
        # ;C located at (0,0,0) and normal along (0,0,1)
        # ;C
        A = 1 / AXMIN ** 2
        B = 1 / AXMAJ ** 2
        C = A
        self.ccc[0] = A
        self.ccc[1] = B * RTCEN[2 - 1] ** 2 + C * RTCEN[3 - 1] ** 2
        self.ccc[2] = B * RNCEN[2 - 1] ** 2 + C * RNCEN[3 - 1] ** 2
        self.ccc[3] = 0.0
        self.ccc[4] = 2 * (B * RNCEN[2 - 1] * RTCEN[2 - 1] + C * RNCEN[3 - 1] * RTCEN[3 - 1])
        self.ccc[5] = 0.0
        self.ccc[6] = 0.0
        self.ccc[7] = 0.0
        self.ccc[8] = 2 * (B * YCEN * RNCEN[2 - 1] + C * ZCEN * RNCEN[3 - 1])
        self.ccc[9] = 0.0

        print("The major axis is %f \nThe minor axis is %f" %(AXMAJ,AXMIN))


    def set_paraboloid_from_focal_distances(self, SSOUR, SIMAG, theta):

        self.p=SSOUR
        self.q=SIMAG
        self.theta=theta

        # ;C
        # ;C Computes the parabola
        # ;C
        COSTHE = np.cos(theta)
        SINTHE = np.sin(theta)
        if SSOUR < SIMAG:
            PARAM = 2 * SSOUR * COSTHE ** 2
            YCEN = -SSOUR * SINTHE ** 2
            ZCEN = -2 * SSOUR * SINTHE * COSTHE
            fact = -1.0
        else:
            PARAM = 2 * SIMAG * COSTHE ** 2
            YCEN = - SIMAG * SINTHE ** 2
            ZCEN = - 2 * SIMAG * SINTHE * COSTHE
            fact = 1.0
        # txt = [txt, $
        # String('Parabolois p: ', $
        # PARAM, Format='(A40,G20.15)')]
        self.ccc[0] = 1.0
        self.ccc[1] = COSTHE ** 2
        self.ccc[2] = SINTHE ** 2
        self.ccc[3] = 0.0
        self.ccc[4] = 2 * fact * COSTHE * SINTHE
        self.ccc[5] = 0.0
        self.ccc[6] = 0.0
        self.ccc[7] = 0.0
        self.ccc[8] = 2 * ZCEN * SINTHE - 2 * PARAM * COSTHE
        self.ccc[9] = 0.0


    def set_hyperboloid_from_focal_distances(self, SSOUR, SIMAG, theta):

        self.p=SSOUR
        self.q=SIMAG
        self.theta=theta

        COSTHE = np.cos(theta)
        SINTHE = np.sin(theta)
        AXMAJ = (SSOUR - SIMAG) / 2
        # ;C
        # ;C If AXMAJ > 0, then we are on the left branch of the hyp. Else we
        # ;C are onto the right one. We have to discriminate between the two cases
        # ;C In particular, if AXMAJ.LT.0 then the hiperb. will be convex.
        # ;C
        AFOCI = 0.5 * np.sqrt(SSOUR ** 2 + SIMAG ** 2 + 2 * SSOUR * SIMAG * np.cos(2 * theta))
        # ;; why this works better?
        # ;;		AFOCI = 0.5D0*SQRT( SSOUR^2 + SIMAG^2 - 2*SSOUR*SIMAG*COS(2*THETA) )
        AXMIN = np.sqrt(AFOCI ** 2 - AXMAJ ** 2)
        ECCENT = AFOCI / np.abs(AXMAJ)
        BRANCH = -1.0  # ; branch=+1,-1
        # ;C
        # ;C Computes the center coordinates in the hiperbola RF.
        # ;C
        # ;IF AXMAJ GT 0.0D0 THEN BEGIN
        # ;  YCEN	=   ( AXMAJ - SSOUR )/ECCENT			; < 0
        # ;ENDIF ELSE BEGIN
        # ;  YCEN	=   ( SSOUR - AXMAJ )/ECCENT			; > 0
        # ;ENDELSE
        YCEN = np.abs(SSOUR - AXMAJ) / ECCENT * BRANCH
        ZCEN_ARG = np.abs(YCEN ** 2 / AXMAJ ** 2 - 1.0)
        if ZCEN_ARG > 1.0e-14:
            ZCEN = -AXMIN * np.sqrt(ZCEN_ARG)  # < 0
        else:
            ZCEN = 0.0
        # ;
        # ; THIS GIVES BETTER LOOKING HYPERBOLA BUT WORSE TRACING. WHY?
        # ;YCEN=ABS(YCEN)
        # ;ZCEN=ABS(ZCEN)
        # ;C
        # ;C Computes now the normal in the same RF. The signs are forced to
        # ;C suit our RF.
        # ;C
        RNCEN = np.zeros(3)
        RNCEN[1 - 1] = 0.0
        RNCEN[2 - 1] = -np.abs(YCEN) / AXMAJ ** 2  # < 0
        RNCEN[3 - 1] = -ZCEN / AXMIN ** 2  # > 0
        RNCEN = RNCEN / np.sqrt((RNCEN ** 2).sum())
        # ;C
        # ;C Computes the tangent in the same RF
        # ;C
        RTCEN = np.zeros(3)
        RTCEN[1 - 1] = 0.0
        RTCEN[2 - 1] = -RNCEN[3 - 1]  # > 0
        RTCEN[3 - 1] = RNCEN[2 - 1]  # > 0
        # txt = [txt,  $
        # String('Rev Hyperboloid a: ', $
        # AXMAJ, Format='(A40,G20.15)'), $
        # String('Rev Hyperboloid b: ', $
        # AXMIN, Format='(A40,G20.15)'), $
        # String('Rev Hyperboloid c: ', $
        # AFOCI, Format='(A40,G20.15)'), $
        # String('Rev Hyperboloid focal discance c^2: ', $
        # AFOCI^2, Format='(A40,G20.15)'), $
        # String('Rev Hyperboloid excentricity: ', $
        # ECCENT, Format='(A40,G20.15)'), $
        # 'Mirror BRANCH: '+String(branch), $
        # 'Mirror center at: '+vect2string([0,YCEN,ZCEN]), $
        # 'Mirror normal: '+vect2string(RNCEN), $
        # 'Mirror tangent: '+vect2string(RTCEN) ]
        # ;C
        # ;C Coefficients of the canonical form
        # ;C
        A = -1 / AXMIN ** 2
        B = 1 / AXMAJ ** 2
        C = A
        # ;C
        # ;C Rotate now in the mirror RF. The equations are the same as for the
        # ;C ellipse case.
        # ;C
        self.ccc[0] = A
        self.ccc[1] = B * RTCEN[2 - 1] ** 2 + C * RTCEN[3 - 1] ** 2
        self.ccc[2] = B * RNCEN[2 - 1] ** 2 + C * RNCEN[3 - 1] ** 2
        self.ccc[3] = 0.0
        self.ccc[4] = 2 * (B * RNCEN[2 - 1] * RTCEN[2 - 1] + C * RNCEN[3 - 1] * RTCEN[3 - 1])
        self.ccc[5] = 0.0
        self.ccc[6] = 0.0
        self.ccc[7] = 0.0
        self.ccc[8] = 2 * (B * YCEN * RNCEN[2 - 1] + C * ZCEN * RNCEN[3 - 1])
        self.ccc[9] = 0.0


    def set_spherical_mirror_radius_from_focal_distances(self,p=None,q=None,theta=None):
        if p is None:
            p = self.p
        if q is None:
            q = self.q
        if theta is None:
            theta = self.theta

        self.R = 2 * p * q / (p + q) / np.cos(theta)



    def rotation_to_the_optical_element(self, beam):                               # rotation along x axis of an angle depending on theta, and along y with an angle tetha

        position = Vector(beam.x,beam.y,beam.z)
        velocity = Vector(beam.vx,beam.vy,beam.vz)
        position.rotation(self.alpha,"y")
        position.rotation(-(np.pi/2-self.theta),"x")
        velocity.rotation(self.alpha,"y")
        velocity.rotation(-(np.pi/2-self.theta),"x")
        [beam.x,beam.y,beam.z] = [position.x,position.y,position.z]
        [beam.vx,beam.vy,beam.vz] = [velocity.x,velocity.y,velocity.z]



    def translation_to_the_optical_element(self, beam):
        vector_point=Vector(0,self.p,0)
        vector_point.rotation(self.alpha,"y")
        vector_point.rotation(-(np.pi/2-self.theta),"x")

        beam.x=beam.x-vector_point.x
        beam.y=beam.y-vector_point.y
        beam.z=beam.z-vector_point.z


    def rotation_to_the_screen(self,beam):

        position = Vector(beam.x,beam.y,beam.z)
        velocity = Vector(beam.vx,beam.vy,beam.vz)
        position.rotation(-(np.pi/2-self.theta),"x")
        velocity.rotation(-(np.pi/2-self.theta),"x")
        [beam.x,beam.y,beam.z] = [position.x,position.y,position.z]
        [beam.vx,beam.vy,beam.vz] = [velocity.x,velocity.y,velocity.z]



    def translation_to_the_screen(self,beam):
        beam.y=beam.y-self.q


    def intersection_with_plane_mirror(self,beam,bound):
        t=-beam.z/beam.vz
        beam.x = beam.x+beam.vx*t
        beam.y = beam.y+beam.vy*t
        beam.z = beam.z+beam.vz*t
        beam.flag=beam.flag+(np.sign(beam.x-bound.xmin*np.ones(beam.N))-1)/2
        beam.flag=beam.flag+(np.sign(beam.y-bound.ymin*np.ones(beam.N))-1)/2
        beam.flag=beam.flag+(np.sign(bound.xmax*np.ones(beam.N)-beam.x)-1)/2
        beam.flag=beam.flag+(np.sign(bound.xmax*np.ones(beam.N)-beam.x)-1)/2
        beam.flag=np.sign(beam.flag)


    def intersection_with_spherical_mirror(self,beam,bound):
            a=beam.vx**2+beam.vy**2+beam.vz**2
            b=beam.x*beam.vx+beam.y*beam.vy+beam.z*beam.vz-beam.vz*self.R                             #This is not b but b/2
            c=beam.x**2+beam.y**2+beam.z**2-2*beam.z*self.R
            t=(-2*b+np.sqrt(4*b**2-4*a*c))/(2*a)
            if t[0]>=0:
                t=t
            else:
                t=(-b-np.sqrt(b**2-a*c))/a

            beam.x = beam.x+beam.vx*t
            beam.y = beam.y+beam.vy*t
            beam.z = beam.z+beam.vz*t

            beam.flag=beam.flag+(np.sign(bound.R**2*np.ones(beam.N)-beam.x**2-beam.y**2)-1)/2
            beam.flag=np.sign(beam.flag)



    def intersection_with_surface_conic(self,beam):
        a=self.ccc[1-1]*beam.vx**2+self.ccc[2-1]*beam.vy**2+self.ccc[3-1]*beam.vz**2+self.ccc[4-1]*beam.vx*beam.vy+self.ccc[5-1]*beam.vy*beam.vz+self.ccc[6-1]*beam.vx*beam.vz
        b=2*self.ccc[1-1]*beam.x*beam.vx+2*self.ccc[2-1]*beam.y*beam.vy+2*self.ccc[3-1]*beam.z*beam.vz+self.ccc[4-1]*beam.x*beam.vy+self.ccc[4-1]*beam.y*beam.vx+self.ccc[5-1]*beam.y*beam.vz+self.ccc[5-1]*beam.z*beam.vy+self.ccc[6-1]*beam.x*beam.vz+self.ccc[6-1]*beam.z*beam.vx+self.ccc[7-1]*beam.vx+self.ccc[8-1]*beam.vy+self.ccc[9-1]*beam.vz
        c=self.ccc[1-1]*beam.x**2+self.ccc[2-1]*beam.y**2+self.ccc[3-1]*beam.z**2+self.ccc[4-1]*beam.x*beam.y+self.ccc[5-1]*beam.y*beam.z+self.ccc[6-1]*beam.x*beam.z+self.ccc[7-1]*beam.x+self.ccc[8-1]*beam.y+self.ccc[9-1]*beam.z+self.ccc[10-1]

        t=(-b+np.sqrt(b**2-4*a*c))/(2*a)
        if t[0]>=0:
            t=t
        else:
            t=(-b-np.sqrt(b**2-a*c))/a
        beam.x = beam.x+beam.vx*t
        beam.y = beam.y+beam.vy*t
        beam.z = beam.z+beam.vz*t



    def output_direction_from_optical_element(self, beam):

        position=Vector(beam.x,beam.y,beam.z)

        if self.type == "Plane mirror":
            normal=position.plane_normal()
        elif self.type == "Spherical mirror":
            normal=position.spherical_normal(self.R)

        normal.normalization()

        velocity=Vector(beam.vx,beam.vy,beam.vz)
        vperp=velocity.perpendicular_component(normal)
        v2=velocity.sum(vperp)
        v2=v2.sum(vperp)

        [beam.vx,beam.vy,beam.vz] = [ v2.x, v2.y, v2.z]



    def intersection_with_the_screen(self,beam):
        t=-beam.y/beam.vy
        beam.x = beam.x+beam.vx*t
        beam.y = beam.y+beam.vy*t
        beam.z = beam.z+beam.vz*t



    def intersection_with_optical_element(self, beam,bound):
        if self.type == "Plane mirror":
            self.intersection_with_plane_mirror(beam,bound)
        elif self.type == "Spherical mirror":
            self.intersection_with_spherical_mirror(beam,bound)


    def trace_optical_element(self, beam1,bound):

        beam=beam1.duplicate()
        #
        # change beam to o.e. frame
        #
        self.rotation_to_the_optical_element(beam)
        self.translation_to_the_optical_element(beam)

        self.intersection_with_optical_element(beam,bound)

        beam.plot_yx()
        plt.title("footprint")

        self.output_direction_from_optical_element(beam)
        print(np.mean(beam.x**2),np.mean(beam.y**2),np.mean(beam.z**2))
        print(np.mean(beam.vx),np.mean(beam.vy),np.mean(beam.vz))


        self.rotation_to_the_screen(beam)

        self.translation_to_the_screen(beam)
        self.intersection_with_the_screen(beam)



        return beam


    def trace_surface_conic(self,beam):

        beam=beam.duplicate()
        self.rotation_to_the_optical_element(beam)
        self.translation_to_the_optical_element(beam)


        self.intersection_with_surface_conic(beam)

        beam.plot_yx()
        plt.title("footprint")

    #####  Output direction ############################################################################################

        position=Vector(beam.x,beam.y,beam.z)
        normal=position.surface_conic_normal(self.ccc)
        normal.normalization()
        velocity = Vector(beam.vx, beam.vy, beam.vz)
        vperp = velocity.perpendicular_component(normal)
        v2 = velocity.sum(vperp)
        v2 = v2.sum(vperp)
        [beam.vx, beam.vy, beam.vz] = [v2.x, v2.y, v2.z]

    ####################################################################################################################

        self.rotation_to_the_screen(beam)
        self.translation_to_the_screen(beam)
        self.intersection_with_the_screen(beam)
        return beam

