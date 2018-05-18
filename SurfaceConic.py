
import numpy

from numpy.testing import assert_equal, assert_almost_equal

# OE surface in form of conic equation:
#      ccc[0]*X^2 + ccc[1]*Y^2 + ccc[2]*Z^2 +
#      ccc[3]*X*Y + ccc[4]*Y*Z + ccc[5]*X*Z  +
#      ccc[6]*X   + ccc[7]*Y   + ccc[8]*Z + ccc[9] = 0


class SurfaceConic(object):

    def __init__(self, ccc=numpy.zeros(10)):

        if ccc is not None:
            self.ccc = ccc.copy()
        else:
            self.ccc = numpy.zeros(10)

    @classmethod
    def initialize_from_coefficients(cls, ccc):
        if numpy.array(ccc).size != 10:
            raise Exception("Invalid coefficients (dimension must be 10)")
        return SurfaceConic(ccc=ccc)

    @classmethod
    def initialize_as_plane(cls):
        return SurfaceConic(numpy.array([0,0,0,0,0,0,0,0,-1.,0]))

    #
    # initializers from focal distances
    #
    @classmethod
    def initialize_as_sphere_from_focal_distances(cls,p, q, theta1, cylindrical=0, cylangle=0.0, switch_convexity=0):
        ccc = SurfaceConic()
        ccc.set_sphere_from_focal_distances(p,q,theta1)
        if cylindrical:
            ccc.set_cylindrical(cylangle)
        if switch_convexity:
            ccc.switch_convexity()
        return ccc


    @classmethod
    def initialize_as_ellipsoid_from_focal_distances(cls,p, q, theta1, cylindrical=0, cylangle=0.0, switch_convexity=0):
        ccc = SurfaceConic()
        ccc.set_ellipsoid_from_focal_distances(p,q,theta1)
        if cylindrical:
            ccc.set_cylindrical(cylangle)
        if switch_convexity:
            ccc.switch_convexity()
        return ccc

    @classmethod
    def initialize_as_paraboloid_from_focal_distances(cls,p, q, theta1, cylindrical=0, cylangle=0.0, switch_convexity=0):
        ccc = SurfaceConic()
        ccc.set_paraboloid_from_focal_distances(p,q,theta1)
        if cylindrical:
            ccc.set_cylindrical(cylangle)
        if switch_convexity:
            ccc.switch_convexity()
        return ccc

    @classmethod
    def initialize_as_hyperboloid_from_focal_distances(cls,p, q, theta1, cylindrical=0, cylangle=0.0, switch_convexity=0):
        ccc = SurfaceConic()
        ccc.set_hyperboloid_from_focal_distances(p,q,theta1)
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
        ccc = SurfaceConic()
        ccc.set_sphere_from_curvature_radius(radius)
        if cylindrical:
            ccc.set_cylindrical(cylangle)
        if switch_convexity:
            ccc.switch_convexity()
        return ccc

    def duplicate(self):
        return SurfaceConic.initialize_from_coefficients(self.ccc.copy())

    #
    # getters
    #

    def get_coefficients(self):
        return self.ccc.copy()


    #
    # setters
    #

    def set_coefficients(self,ccc):
        if numpy.array(ccc).size != 10:
            raise Exception("Invalid coefficients (dimension must be 10)")
        self.ccc = ccc


    def vector_reflection(self,v1,normal):
        tmp = v1 * normal
        tmp2 = tmp[0,:] + tmp[1,:] + tmp[2,:]
        tmp3 = normal.copy()

        for jj in (0,1,2):
            tmp3[jj,:] = tmp3[jj,:] * tmp2

        v2 = v1 - 2 * tmp3
        v2mod = numpy.sqrt(v2[0,:]**2 + v2[1,:]**2 + v2[2,:]**2)
        v2 /= v2mod

        return v2

    def get_normal(self,x2):
        # ;
        # ; Calculates the normal at intercept points x2 [see shadow's normal.F]
        # ;
        normal = numpy.zeros_like(x2)

        normal[0,:] = 2 * self.ccc[1-1] * x2[0,:] + self.ccc[4-1] * x2[1,:] + self.ccc[6-1] * x2[2,:] + self.ccc[7-1]
        normal[1,:] = 2 * self.ccc[2-1] * x2[1,:] + self.ccc[4-1] * x2[0,:] + self.ccc[5-1] * x2[2,:] + self.ccc[8-1]
        normal[2,:] = 2 * self.ccc[3-1] * x2[2,:] + self.ccc[5-1] * x2[1,:] + self.ccc[6-1] * x2[0,:] + self.ccc[9-1]

        normalmod =  numpy.sqrt( normal[0,:]**2 + normal[1,:]**2 + normal[2,:]**2 )
        normal[0,:] /= normalmod
        normal[1,:] /= normalmod
        normal[2,:] /= normalmod

        return normal


    def apply_specular_reflection_on_beam(self,newbeam):
        # ;
        # ; TRACING...
        # ;

        x1 =   newbeam.get_columns([1,2,3]) # numpy.array(a3.getshcol([1,2,3]))
        v1 =   newbeam.get_columns([4,5,6]) # numpy.array(a3.getshcol([4,5,6]))
        flag = newbeam.get_column(10)        # numpy.array(a3.getshonecol(10))


        t,iflag = self.calculate_intercept(x1,v1)
        x2 = x1 + v1 * t
        for i in range(flag.size):
            if iflag[i] < 0: flag[i] = -100


        # ;
        # ; Calculates the normal at each intercept [see shadow's normal.F]
        # ;

        normal = self.get_normal(x2)

        # ;
        # ; reflection
        # ;

        v2 = self.vector_reflection(v1,normal)

        # ;
        # ; writes the mirr.XX file
        # ;

        newbeam.set_column(1, x2[0])
        newbeam.set_column(2, x2[1])
        newbeam.set_column(3, x2[2])
        newbeam.set_column(4, v2[0])
        newbeam.set_column(5, v2[1])
        newbeam.set_column(6, v2[2])
        newbeam.set_column(10, flag )

        return newbeam

    def calculate_intercept(self,XIN,VIN,keep=0):
    # FUNCTION conicintercept,ccc,xIn1,vIn1,iflag,keep=keep
    #
    #
    # ;+
    # ;
    # ;       NAME:
    # ;               CONICINTERCEPT
    # ;
    # ;       PURPOSE:
    # ;               This function Calculates the intersection of a
    # ;               conic (defined by its 10 coefficients in ccc)
    # ;               with a straight line, defined by a point xIn and
    # ;               an unitary direction vector vIn
    # ;
    # ;       CATEGORY:
    # ;               SHADOW tools
    # ;
    # ;       CALLING SEQUENCE:
    # ;               t = conicIntercept(ccc,xIn,vIn,iFlag)
    # ;
    # ; 	INPUTS:
    # ;		ccc: the array with the 10 coefficients defining the
    # ;                    conic.
    # ;		xIn: a vector DblArr(3) or stack of vectors DblArr(3,nvectors)
    # ;		vIn: a vector DblArr(3) or stack of vectors DblArr(3,nvectors)
    # ;
    # ;       OUTPUTS
    # ;		t the "travelled" distance between xIn and the surface
    # ;
    # ; 	OUTPUT KEYWORD PARAMETERS
    # ;		IFLAG: A flag (negative if no intersection)
    # ;
    # ; 	KEYWORD PARAMETERS
    # ;               keep: 0 [default] keep the max t from both solutions
    # ;                     1 keep the MIN t from both solutions
    # ;                     2 keep the first solution
    # ;                     3 keep the second solution
    # ;	ALGORITHM:
    # ;		 Adapted from SHADOW/INTERCEPT
    # ;
    # ;		 Equation of the conic:
    # ;
    # ;	         c[0]*X^2 + c[1]*Y^2 + c[2]*Z^2 +
    # ;                c[3]*X*Y + c[4]*Y*Z + c[5]*X*Z  +
    # ;                c[6]*X + c[7]*Y + c[8]*Z + c[9] = 0
    # ;
    # ;       NOTE that the vectors, that are usually DblArr(3) can be
    # ;            stacks of vectors DblArr(3,nvectors). In such a case,
    # ;            the routine returns t
    # ;
    # ;
    # ;	AUTHOR:
    # ;		M. Sanchez del Rio srio@esrf.eu, Sept. 29, 2009
    # ;
    # ;	MODIFICATION HISTORY:
    # ;
    # ;-
    #
    #
    # ;CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    # ;C
    # ;C	subroutine	intercept	( xin, vin, tpar, iflag)
    # ;C
    # ;C	purpose		computes the intercepts onto the mirror surface
    # ;C
    # ;C	arguments	xin	ray starting position     mirror RF
    # ;C			vin	ray direction		  mirror RF
    # ;C			tpar	distance from start of
    # ;C				intercept
    # ;C			iflag   input		1	ordinary case
    # ;C					       -1	ripple case
    # ;C			iflag	output		0	success
    # ;C					       -1       complex sol.
    # ;C
    # ;CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    #

        CCC = self.ccc

        if XIN.shape==(3,):
            XIN.shape = (3,1)
        if VIN.shape==(3,):
            VIN.shape = (3,1)

        AA 	=       CCC[1-1]*VIN[1-1,:]**2  \
                        + CCC[2-1]*VIN[2-1,:]**2  \
                        + CCC[3-1]*VIN[3-1,:]**2  \
                        + CCC[4-1]*VIN[1-1,:]*VIN[2-1,:]  \
                        + CCC[5-1]*VIN[2-1,:]*VIN[3-1,:]  \
                        + CCC[6-1]*VIN[1-1,:]*VIN[3-1,:]


        BB 	=       CCC[1-1] *  XIN[1-1,:] * VIN[1-1,:]*2    \
                        + CCC[2-1] *  XIN[2-1,:] * VIN[2-1,:]*2    \
                        + CCC[3-1] *  XIN[3-1,:] * VIN[3-1,:]*2    \
                        + CCC[4-1] * (XIN[2-1,:] * VIN[1-1,:]    \
                        + XIN[1-1,:] * VIN[2-1,:])    \
                        + CCC[5-1]*(XIN[3-1,:]*VIN[2-1,:]    \
                        + XIN[2-1,:]*VIN[3-1,:])    \
                        + CCC[6-1]*(XIN[1-1,:]*VIN[3-1,:]    \
                        + XIN[3-1,:]*VIN[1-1,:])    \
                        + CCC[7-1] * VIN[1-1,:]    \
                        + CCC[8-1] * VIN[2-1,:]    \
                        + CCC[9-1] * VIN[3-1,:]

        CC 	=             CCC[1-1] * XIN[1-1,:]**2    \
                        + CCC[2-1] * XIN[2-1,:]**2    \
                        + CCC[3-1] * XIN[3-1,:]**2    \
                        + CCC[4-1] * XIN[2-1,:] * XIN[1-1,:]    \
                        + CCC[5-1] * XIN[2-1,:] * XIN[3-1,:]    \
                        + CCC[6-1] * XIN[1-1,:] * XIN[3-1,:]    \
                        + CCC[7-1] * XIN[1-1,:]    \
                        + CCC[8-1] * XIN[2-1,:]    \
                        + CCC[9-1] * XIN[3-1,:]    \
                        + CCC[10-1]


    # ;C
    # ;C Solve now the second deg. equation **
    # ;C


        DENOM = AA*0.0
        DETER = AA*0.0
        TPAR1 = AA*0.0
        TPAR2 = AA*0.0
        IFLAG = numpy.ones(AA.size) # int(AA*0)+1

        # itest1 = numpy.argwhere( numpy.abs(AA) > 1e-15)
        # if len(itest1) > 0:
        #
        #     DENOM[itest1] = 0.5 / AA[itest1]
        #     DETER[itest1] = BB[itest1]**2 - CC[itest1] * AA[itest1] * 4
        #
        #     TMP = DETER[itest1]
        #
        #     ibad = numpy.argwhere(TMP < 0)
        #     if len(ibad) == 0:
        #         IFLAG[itest1[ibad]] = -1
        #
        #     igood = numpy.argwhere(TMP >= 0)
        #     if len(igood) > 0:
        #         itmp = itest1[igood]
        #         TPAR1[itmp] = -(BB[itmp] + numpy.sqrt(DETER[itmp])) * DENOM[itmp]
        #         TPAR2[itmp] = -(BB[itmp] - numpy.sqrt(DETER[itmp])) * DENOM[itmp]
        #
        #         if keep == 0:
        #             TPAR = numpy.maximum(TPAR1,TPAR2)
        #         elif keep == 1:
        #             TPAR = numpy.minimum(TPAR1,TPAR2)
        #         elif keep == 2:
        #             TPAR = TPAR1
        #         elif keep == 3:
        #             TPAR = TPAR2
        #         else:
        #             TPAR = TPAR1
        #
        # else:
        #     TPAR = - CC / BB

        TPAR = numpy.zeros_like(AA)
        T_SOURCE = 10.0

        # TODO: remove loop!
        for i in range(AA.size):
            if numpy.abs(AA[i])  < 1e-15:
                TPAR1[i] = - CC[i] / BB[i]
                TPAR2[i] = TPAR1[i]
            else:

                DENOM = 0.5 / AA[i]
                DETER = BB[i] ** 2 - CC[i] * AA[i] * 4

                if DETER < 0.0:

                    TPAR[i] = 0.0
                    IFLAG[i] = -1
                else:
                    TPAR1 = -(BB[i] + numpy.sqrt(DETER)) * DENOM
                    TPAR2 = -(BB[i] - numpy.sqrt(DETER)) * DENOM
                    #if ( numpy.abs(TPAR1-T_SOURCE) <= numpy.abs(TPAR2-T_SOURCE)):
                    #    TPAR[i] = TPAR1
                    #else:
                    #    TPAR[i] = TPAR2

        print("the times are:     ")
        print(TPAR1, TPAR2)
        return TPAR1, TPAR2, IFLAG



    def set_cylindrical(self,CIL_ANG):

        COS_CIL = numpy.cos(CIL_ANG)
        SIN_CIL = numpy.sin(CIL_ANG)

        A_1	 =   self.ccc[1-1]
        A_2	 =   self.ccc[2-1]
        A_3	 =   self.ccc[3-1]
        A_4	 =   self.ccc[4-1]
        A_5	 =   self.ccc[5-1]
        A_6	 =   self.ccc[6-1]
        A_7	 =   self.ccc[7-1]
        A_8	 =   self.ccc[8-1]
        A_9	 =   self.ccc[9-1]
        A_10 =   self.ccc[10-1]


        self.ccc[1-1] =  A_1 * SIN_CIL**4 + A_2 * COS_CIL**2 * SIN_CIL**2 - A_4 * COS_CIL * SIN_CIL**3
        self.ccc[2-1] =  A_2 * COS_CIL**4 + A_1 * COS_CIL**2 * SIN_CIL**2 - A_4 * COS_CIL**3 * SIN_CIL
        self.ccc[3-1] =  A_3						 # Z^2
        self.ccc[4-1] =  - 2*A_1 * COS_CIL * SIN_CIL**3 - 2 * A_2 * COS_CIL**3 * SIN_CIL + 2 * A_4 * COS_CIL**2 *SIN_CIL**2 # X Y
        self.ccc[5-1] =  A_5 * COS_CIL**2 - A_6 * COS_CIL * SIN_CIL	 # Y Z
        self.ccc[6-1] =  A_6 * SIN_CIL**2 - A_5 * COS_CIL * SIN_CIL	 # X Z
        self.ccc[7-1] =  A_7 * SIN_CIL**2 - A_8 * COS_CIL * SIN_CIL	 # X
        self.ccc[8-1] =  A_8 * COS_CIL**2 - A_7 * COS_CIL * SIN_CIL	 # Y
        self.ccc[9-1] =  A_9						 # Z
        self.ccc[10-1]=  A_10


    def switch_convexity(self):
        self.ccc[5-1]  = - self.ccc[5-1]
        self.ccc[6-1]  = - self.ccc[6-1]
        self.ccc[9-1]  = - self.ccc[9-1]


    def set_sphere_from_focal_distances(self, ssour, simag, theta_grazing):
        # todo: implement also sagittal bending
        print("Theta grazing is: %f" %(theta_grazing))
        theta = (numpy.pi/2) - theta_grazing
        print("Theta  is: %f" %(theta))
        print('>>>> set_sphere_from_focal_distances: Angle with respect to the surface normal [rad]:',theta)
        rmirr = ssour * simag * 2 / numpy.cos(theta) / (ssour + simag)

        self.ccc[1-1] =  1.0	        # X^2  # = 0 in cylinder case
        self.ccc[2-1] =  1.0	        # Y^2
        self.ccc[3-1] =  1.0	        # Z^2
        self.ccc[4-1] =   .0	        # X*Y   # = 0 in cylinder case
        self.ccc[5-1] =   .0	        # Y*Z
        self.ccc[6-1] =   .0	        # X*Z   # = 0 in cylinder case
        self.ccc[7-1] =   .0	        # X     # = 0 in cylinder case
        self.ccc[8-1] =   .0	        # Y
        self.ccc[9-1] = -2 * rmirr	# Z
        self.ccc[10-1]  =   .0       # G
        print(">>>> set_sphere_from_focal_distances: Spherical radius: %f \n"%(rmirr))

    def set_sphere_from_curvature_radius(self,rmirr):
        self.ccc[1-1] =  1.0	        # X^2  # = 0 in cylinder case
        self.ccc[2-1] =  1.0	        # Y^2
        self.ccc[3-1] =  1.0	        # Z^2
        self.ccc[4-1] =   .0	        # X*Y   # = 0 in cylinder case
        self.ccc[5-1] =   .0	        # Y*Z
        self.ccc[6-1] =   .0	        # X*Z   # = 0 in cylinder case
        self.ccc[7-1] =   .0	        # X     # = 0 in cylinder case
        self.ccc[8-1] =   .0	        # Y
        self.ccc[9-1] = -2 * rmirr	# Z
        self.ccc[10-1]  =   .0       # G

    def set_ellipsoid_from_focal_distances(self, ssour, simag, theta_grazing):

        theta = (numpy.pi/2) - theta_grazing
        COSTHE = numpy.cos(theta)
        SINTHE = numpy.sin(theta)

        AXMAJ = ( ssour + simag )/2
        AXMIN = numpy.sqrt( simag * ssour) * COSTHE
        AFOCI = numpy.sqrt( AXMAJ**2 - AXMIN**2 )
        ECCENT = AFOCI/AXMAJ
        # ;C
        # ;C The center is computed on the basis of the object and image positions
        # ;C
        YCEN  = (ssour - simag) * 0.5 / ECCENT
        ZCEN  = -numpy.sqrt( 1 - YCEN**2 / AXMAJ**2) * AXMIN
        # ;C
        # ;C Computes now the normal in the mirror center.
        # ;C
        RNCEN = numpy.zeros(3)
        RNCEN[1-1] =  0.0
        RNCEN[2-1] = -2 * YCEN / AXMAJ**2
        RNCEN[3-1] = -2 * ZCEN / AXMIN**2
        # ;CALL NORM(RNCEN,RNCEN)
        RNCEN = RNCEN / numpy.sqrt((RNCEN**2).sum())
        # ;C
        # ;C Computes the tangent versor in the mirror center.
        # ;C
        RTCEN = numpy.zeros(3)
        RTCEN[1-1] =  0.0
        RTCEN[2-1] =  RNCEN[3-1]
        RTCEN[3-1] = -RNCEN[2-1]

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

        A = 1 / AXMIN**2
        B = 1 / AXMAJ**2
        C = A
        self.ccc[0] = A
        self.ccc[1] = B * RTCEN[2-1]**2 + C * RTCEN[3-1]**2
        self.ccc[2] = B * RNCEN[2-1]**2 + C * RNCEN[3-1]**2
        self.ccc[3] = 0.0
        self.ccc[4] = 2 * (B * RNCEN[2-1] * RTCEN[2-1] + C * RNCEN[3-1] * RTCEN[3-1])
        self.ccc[5] = 0.0
        self.ccc[6] = 0.0
        self.ccc[7] = 0.0
        self.ccc[8] = 2 * (B * YCEN * RNCEN[2-1] + C * ZCEN * RNCEN[3-1])
        self.ccc[9] = 0.0


    def set_paraboloid_from_focal_distance(self, SSOUR, SIMAG, theta_grazing, infinity_location):
        # ;C
        # ;C Computes the parabola
        # ;C
        theta = (numpy.pi/2) - theta_grazing
        COSTHE = numpy.cos(theta)
        SINTHE = numpy.sin(theta)

        if infinity_location=="q":
            PARAM = 2 * SSOUR * COSTHE**2
            YCEN = -SSOUR * SINTHE**2
            ZCEN = -2 * SSOUR * SINTHE * COSTHE
            fact = -1.0
        elif infinity_location == "p":
            PARAM =   2 * SIMAG * COSTHE**2
            YCEN = - SIMAG * SINTHE**2
            ZCEN = - 2 * SIMAG * SINTHE * COSTHE
            fact = 1.0

        # txt = [txt, $
        # String('Parabolois p: ', $
        # PARAM, Format='(A40,G20.15)')]
        self.ccc[0] = 1.0
        self.ccc[1] = COSTHE**2
        self.ccc[2] = SINTHE**2
        self.ccc[3] = 0.0
        self.ccc[4] = 2 * fact * COSTHE * SINTHE
        self.ccc[5] = 0.0
        self.ccc[6] = 0.0
        self.ccc[7] = 0.0
        self.ccc[8] = 2 * ZCEN * SINTHE - 2 * PARAM * COSTHE
        self.ccc[9] = 0.0


    def set_hyperboloid_from_focal_distances(self, SSOUR, SIMAG, theta_grazing):

        theta = (numpy.pi/2) - theta_grazing
        COSTHE = numpy.cos(theta)
        SINTHE = numpy.sin(theta)

        AXMAJ = (SSOUR - SIMAG)/2
        # ;C
        # ;C If AXMAJ > 0, then we are on the left branch of the hyp. Else we
        # ;C are onto the right one. We have to discriminate between the two cases
        # ;C In particular, if AXMAJ.LT.0 then the hiperb. will be convex.
        # ;C
        AFOCI = 0.5 * numpy.sqrt( SSOUR**2 + SIMAG**2 + 2 * SSOUR * SIMAG * numpy.cos(2 * theta) )
        # ;; why this works better?
        # ;;		AFOCI = 0.5D0*SQRT( SSOUR^2 + SIMAG^2 - 2*SSOUR*SIMAG*COS(2*THETA) )
        AXMIN = numpy.sqrt( AFOCI**2 - AXMAJ**2 )

        ECCENT = AFOCI / numpy.abs( AXMAJ )

        BRANCH = -1.0   #; branch=+1,-1
        # ;C
        # ;C Computes the center coordinates in the hiperbola RF.
        # ;C
        # ;IF AXMAJ GT 0.0D0 THEN BEGIN
        # ;  YCEN	=   ( AXMAJ - SSOUR )/ECCENT			; < 0
        # ;ENDIF ELSE BEGIN
        # ;  YCEN	=   ( SSOUR - AXMAJ )/ECCENT			; > 0
        # ;ENDELSE

        if AXMAJ>0:
            YCEN = (SSOUR - AXMAJ) / ECCENT
        else:
            YCEN = (SSOUR - AXMAJ) / ECCENT


        #YCEN =   numpy.abs( SSOUR - AXMAJ ) / ECCENT * BRANCH

        ZCEN_ARG = numpy.abs( YCEN**2 / AXMAJ**2 - 1.0)

        if ZCEN_ARG > 1.0e-14:
            ZCEN = -AXMIN * numpy.sqrt(ZCEN_ARG)  # < 0
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

        RNCEN = numpy.zeros(3)
        RNCEN[1-1] = 0.0
        RNCEN[2-1] = -numpy.abs( YCEN ) / AXMAJ**2 # < 0
        RNCEN[3-1] = -ZCEN / AXMIN**2              # > 0

        RNCEN = RNCEN / numpy.sqrt((RNCEN**2).sum())
        # ;C
        # ;C Computes the tangent in the same RF
        # ;C
        RTCEN = numpy.zeros(3)
        RTCEN[1-1] =  0.0
        RTCEN[2-1] = -RNCEN[3-1]  # > 0
        RTCEN[3-1] =  RNCEN[2-1]  # > 0

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
        A = -1 / AXMIN**2
        B =  1 / AXMAJ**2
        C =  A
        # ;C
        # ;C Rotate now in the mirror RF. The equations are the same as for the
        # ;C ellipse case.
        # ;C
        self.ccc[0] = A
        self.ccc[1] = B * RTCEN[2-1]**2 + C * RTCEN[3-1]**2
        self.ccc[2] = B * RNCEN[2-1]**2 + C * RNCEN[3-1]**2
        self.ccc[3] = 0.0
        self.ccc[4] =  2 * (B *RNCEN[2-1] * RTCEN[2-1] + C * RNCEN[3-1] * RTCEN[3-1])
        self.ccc[5] = 0.0
        self.ccc[6] = 0.0
        self.ccc[7] = 0.0
        self.ccc[8] = 2 * (B * YCEN * RNCEN[2-1] + C * ZCEN * RNCEN[3-1])
        self.ccc[9] = 0.0


        #self.ccc[0] = -0.0111673
        #self.ccc[1] = -0.005
        #self.ccc[2] = 0.0338327
        #self.ccc[3] = 0.0
        #self.ccc[4] = 0.0333184
        #self.ccc[5] = 0.0
        #self.ccc[6] = 0.0
        #self.ccc[7] = 0.0
        #self.ccc[8] = -0.453685
        #self.ccc[9] = 0.0
        print("Axmax is:   %f" %(AXMAJ))
        print("Axmin is:   %f" %(AXMIN))

    #
    # info
    #
    def info(self):
        """

        :return:
        """
        txt = ""

        txt += "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
        txt += "OE surface in form of conic equation: \n"
        txt += "  ccc[0]*X^2 + ccc[1]*Y^2 + ccc[2]*Z^2  \n"
        txt += "  ccc[3]*X*Y + ccc[4]*Y*Z + ccc[5]*X*Z  \n"
        txt += "  ccc[6]*X   + ccc[7]*Y   + ccc[8]*Z + ccc[9] = 0 \n"
        txt += " with \n"
        txt += " c[0] = %f \n "%self.ccc[0]
        txt += " c[1] = %f \n "%self.ccc[1]
        txt += " c[2] = %f \n "%self.ccc[2]
        txt += " c[3] = %f \n "%self.ccc[3]
        txt += " c[4] = %f \n "%self.ccc[4]
        txt += " c[5] = %f \n "%self.ccc[5]
        txt += " c[6] = %f \n "%self.ccc[6]
        txt += " c[7] = %f \n "%self.ccc[7]
        txt += " c[8] = %f \n "%self.ccc[8]
        txt += " c[9] = %f \n "%self.ccc[9]
        txt += "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'n"

        return txt

