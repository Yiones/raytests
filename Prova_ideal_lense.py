from Beam import Beam
from OpticalElement import Optical_element
from Shape import BoundaryRectangle
import numpy as np
import matplotlib.pyplot as plt
from numpy.testing import assert_almost_equal
from Vector import Vector

fx=0.5
fz=0.5


beam=Beam(5000)
#beam.set_divergences_collimated()
#beam.set_rectangular_spot(1.,-1.,1.,-1.)
beam.set_flat_divergence(0.05,0.05)

beam.plot_xz()
beam.plot_xpzp()


lens=Optical_element()
lens.set_parameters(1.,1.)

beam=lens.trace_ideal_lens(beam)

beam.plot_xz()
beam.histogram()

plt.show()