import pytest
from math import sqrt

from athanor import Point, PlanePoint, SpacePoint, ConstantPoint, CeroPoint

def test_point_creation():
    p = Point((1, 2, 3), name="A")
    assert p.coords == (1, 2, 3)
    assert p.name == "A"
    assert len(p) == 3
    assert p.x == 1
    assert p.y == 2
    assert p.z == 3

def test_point_creation_rn():
    coords = tuple(range(10))  # 10D point
    p = Point(coords, name="Rn")
    assert p.coords == coords
    assert len(p) == 10
    assert p.x == 0
    assert p.y == 1
    assert p.z == 2  # z exists because len >= 3

def test_point_invalid_coordinates():
    with pytest.raises(AssertionError):
        Point((1, "a", 3))

# ---------- Distance tests ----------

def test_distance_2d():
    p1 = Point((0,0))
    p2 = Point((3,4))
    assert p1.distance_to(p2) == 5

def test_distance_3d():
    p1 = Point((1,2,2))
    p2 = Point((4,6,6))
    expected = sqrt((4-1)**2 + (6-2)**2 + (6-2)**2)
    assert p1.distance_to(p2) == expected

def test_distance_rn():
    coords1 = tuple(range(5))        # (0,1,2,3,4)
    coords2 = tuple(range(5,10))     # (5,6,7,8,9)
    p1 = Point(coords1)
    p2 = Point(coords2)
    expected = sqrt(sum((a-b)**2 for a,b in zip(coords1, coords2)))
    assert p1.distance_to(p2) == expected

def test_distance_dimension_mismatch():
    p1 = Point((1,2))
    p2 = Point((1,2,3))
    with pytest.raises(AssertionError):
        p1.distance_to(p2)

def test_distance_invalid_argument():
    p = Point((0,0))
    with pytest.raises(AssertionError):
        p.distance_to((0,0))

# ---------- __repr__ ----------

def test_repr():
    p = Point((1,2), name="R")
    assert repr(p) == "Point (R) (1, 2)"

# ---------- PlanePoint (2D) ----------

def test_plane_point_creation():
    p = PlanePoint(4,5, name="B")
    assert p.coords == (4,5)
    assert p.x == 4
    assert p.y == 5
    assert p.z is None

# ---------- SpacePoint (3D) ----------

def test_space_point_creation():
    p = SpacePoint(7,8,9, name="C")
    assert p.coords == (7,8,9)
    assert p.x == 7
    assert p.y == 8
    assert p.z == 9

# ---------- ConstantPoint ----------

def test_constant_point():
    c = ConstantPoint(5, dimension=4, name="Const")
    assert c.coords == (5,5,5,5)
    assert c.name == "Const"
    assert len(c) == 4

# ---------- CeroPoint ----------

def test_cero_point():
    o = CeroPoint(dimension=3, name="Zero")
    assert o.coords == (0,0,0)
    assert o.name == "Zero"
    assert len(o) == 3

# ---------- Additional tests for edge cases in R^n ----------

def test_point_1d():
    p = Point((42,), name="OneD")
    assert len(p) == 1
    assert p.x == 42
    assert p.y is None
    assert p.z is None

def test_point_empty():
    p = Point((), name="Empty")
    assert len(p) == 0
    assert p.x is None
    assert p.y is None
    assert p.z is None

def test_constant_point_rn():
    c = ConstantPoint(3, dimension=7)
    assert c.coords == (3,3,3,3,3,3,3)
    assert len(c) == 7

def test_cero_point_rn():
    z = CeroPoint(dimension=5)
    assert z.coords == (0,0,0,0,0)
    assert len(z) == 5