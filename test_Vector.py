from Vector import Vector
import numpy as np
from numpy.testing import assert_almost_equal





def test_rotation_1():
    theta=0
    v = Vector(0,1,0)
    v.rotation(-(90-theta),"x")
    print(v.info())
    assert_almost_equal ( v.x , 0.0 , 15)
    assert_almost_equal ( v.y , 0.0 , 15)
    assert_almost_equal ( v.z , -1.0 , 15)


def test_rotation_2():
    theta=45
    v = Vector(0,1,0)
    v.rotation(-(90-theta),"x")
    print(v.info())
    assert_almost_equal ( v.x , 0.0 , 15)
    assert_almost_equal ( v.y , 1/np.sqrt(2) , 15)
    assert_almost_equal ( v.z , -1/np.sqrt(2) , 15)


def test_rotation_3():
    alpha=90
    v = Vector(0,1,0)
    v.rotation(alpha,"y")
    print(v.info())
    assert_almost_equal ( v.x , 0.0 , 15)
    assert_almost_equal ( v.y , 1.0 , 15)
    assert_almost_equal ( v.z , 0.0 , 15)

def test_rotation_4():
    theta = 45
    alpha = 90
    v = Vector(0, 1, 0)
    v.rotation(alpha,"y")
    v.rotation(-(90 - theta), "x")
    print(v.info())
    assert_almost_equal(v.x, 0.0, 15)
    assert_almost_equal(v.y, 1/np.sqrt(2) , 15)
    assert_almost_equal(v.z, -1/np.sqrt(2), 15)


def test_rodrigues():
    theta=90
    axis=Vector(0,0,1)
    v=Vector(0,1,0)
    vrot=v.rodrigues_formula(axis,theta)
    print(v.info())
    assert_almost_equal(vrot.x, -1.0, 15)
    assert_almost_equal(vrot.y, 0, 15)
    assert_almost_equal(vrot.z, 0, 15)

def test_rotation_10_array():
    theta = 45
    alpha = 90
    x=np.zeros(10)
    y=np.ones(10)
    z=np.zeros(10)
    v=Vector(x,y,z)
    v.rotation(alpha,"y")
    v.rotation(-(90 - theta), "x")

    assert_almost_equal(v.x,np.zeros(10))
    assert_almost_equal(v.y,1/np.sqrt(2) *np.ones(10))
    assert_almost_equal(v.z,-1/np.sqrt(2)*np.ones(10))


def test_sum_vector():
    v=Vector(0.,1.,0.)
    v2=v.sum(v)

    print(v2.info())
    assert_almost_equal(v2.x, 0.0, 15)
    assert_almost_equal(v2.y, 2., 15)
    assert_almost_equal(v2.z, 0., 15)

def test_sum_list_vector():

    x=np.zeros(10)
    y=np.ones(10)
    z=np.zeros(10)
    v=Vector(x,y,z)
    v2=v.sum(v)
    assert_almost_equal(v2.x, np.zeros(10), 15)
    assert_almost_equal(v2.y, 2*np.ones(10), 15)
    assert_almost_equal(v2.z, np.zeros(10), 15)


def test_dot_product_vector():
    v=Vector(0.,1.,0.)
    dot_product=v.dot(v)

    print(dot_product)
    assert_almost_equal(dot_product, 1., 15)


def test_normalizationvector():

    x=1
    y=1
    z=1
    v=Vector(x,y,z)
    v.normalization()
    print(v.info())
    assert_almost_equal(v.x, 1/np.sqrt(3), 15)
    assert_almost_equal(v.y, 1/np.sqrt(3), 15)
    assert_almost_equal(v.z, 1/np.sqrt(3), 15)


def test_normalization_list_vector():
    x = np.ones(10)
    y = np.ones(10)
    z = np.ones(10)
    v = Vector(x, y, z)
    v.normalization()
    assert_almost_equal(v.x, 1 / np.sqrt(3) * np.ones(10), 15)
    assert_almost_equal(v.y, 1 / np.sqrt(3) * np.ones(10), 15)
    assert_almost_equal(v.z, 1 / np.sqrt(3) * np.ones(10), 15)






