class Point:
     def __init__(self, coords:tuple[float], name:str="P"):
          assert all(isinstance(c, (int, float)) for c in coords), "All coordinates must be real numbers to define a point."
          self.coords:tuple[float] = coords
          self.name = name

     def __repr__(self):
          return f"Point ({self.name}) {self.coords}"
     
     def __len__(self):
          return len(self.coords)
     
     def distance_to(self, other:'Point') -> float:
          """Calculates the Euclidean distance between this point and another point."""
          assert len(self) == len(other), "Points must have the same dimension to calculate distance."
          assert isinstance(other, Point), "Can only calculate distance to another Point."
          return sum((a - b) ** 2 for a, b in zip(self.coords, other.coords)) ** 0.5
     
     @property
     def x(self) -> float:
          return self.coords[0] if len(self) >= 1 else None
     
     @property
     def y(self) -> float:
          return self.coords[1] if len(self) >= 2 else None
     
     @property
     def z(self) -> float:
          return self.coords[2] if len(self) >= 3 else None   

class PlanePoint(Point):
     """Point of R2 (2D Point)."""
     def __init__(self, x, y, name:str="P"):
          super().__init__((x, y), name = name) # Initialize base Point class with 2D coordinates

class SpacePoint(Point):
     """Point of R3 (3D Point)."""
     def __init__(self, x, y, z, name:str="P"):
          super().__init__((x, y, z), name=name) # Initialize base Point class with 3D coordinates

class ConstantPoint(Point):
     """Point with all coordinates equal to a constant value C."""
     def __init__(self, value:float, dimension:int=3, name:str="C"):
          coords = tuple(value for _ in range(dimension))
          super().__init__(coords, name=name)

class CeroPoint(ConstantPoint):
     """Point with all coordinates equal to zero."""
     def __init__(self, dimension:int=3, name:str="O"):
          super().__init__(0, dimension=dimension, name=name)