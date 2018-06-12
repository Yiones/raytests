import numpy as np

class BoundaryShape(object):
    "this is an abstract class"


class BoundaryRectangle(BoundaryShape):
    def __init__(self ,xmax ,xmin , ymax, ymin):
        self.xmax = xmax
        self.xmin = xmin
        self.ymax = ymax
        self.ymin = ymin


    def info(self):
        return ("xmax=%f, xmin=%f, ymax=%f, ymin=%f"  %(self.xmax, self.xmin, self.ymax, self.ymin))

class BoundaryCircle(BoundaryShape):
    def __init__(self,R):
        self.R=R




bound=BoundaryCircle(1)

x=np.array([0.3, 0.5])
y=np.array([0.3, 1.5])
t=np.zeros(2)

t = t + (np.sign(bound.R ** 2 * np.ones(2) - x ** 2 - y ** 2) - 1) / 2
print (t)

t = np.sign(t)
print(t)