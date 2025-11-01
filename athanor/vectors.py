from .points import Point, PlanePoint, SpacePoint, CeroPoint

class Vector:
     def __init__(self, start:Point, end:Point, name:str = None):
          self.start = start
          self.end = end
          # By using zip, we can handle points in any dimension
          coords = tuple(end_coord - start_coord for start_coord, end_coord in zip(self.start.coords, self.end.coords))
          self.coords = tuple(end_coord - start_coord for start_coord, end_coord in zip(self.start.coords, self.end.coords))
          self.name = name if name else f"{self.start.name}{self.end.name}" 

     def __str__(self) -> str:
          return f"Vector ({self.name}):({self.start} -> {self.end})"
     
     def __len__(self) -> int:
          return len(self.coords)
          
     def __mul__(self, scalar:float) -> 'Vector':
          """Scalar multiplication of the vector by a real number."""
          assert isinstance(scalar, (int, float)), "Can only multiply a SpaceVector by a scalar (real number)."
    
          scaled_coords = tuple(coord * scalar for coord in self.coords)
          # P'_end = P_start + P_scaled
          new_end_coords = tuple(
               s_coord + v_coord 
               for s_coord, v_coord in zip(self.start.coords, scaled_coords)
          )
    
          return Vector(
               start=self.start,
               end=Point(new_end_coords, name=f"{scalar} * {self.end.name}")
          )

     def __add__(self, other:'Vector') -> 'Vector':
          assert isinstance(other, Vector), "Can only add a vector with another vector."
    
          if other:
               added_coords = tuple(a + b for a, b in zip(self.coords, other.coords))
               # P'_end = P_start + (v_self + v_other)
               new_end_coords = tuple(
                    s_coord + v_coord 
                    for s_coord, v_coord in zip(self.start.coords, added_coords)
               )

               return Vector(
                    start=self.start, # Mantenemos el punto de inicio
                    end=Point(new_end_coords, name=f"{self.end.name} + {other.end.name}")
               )
          return self
     
     def __iadd__(self, other:'Vector') -> 'Vector':
          return self + other
     
     def __sub__(self, other:'Vector') -> 'Vector':
          return self + (other*-1)
     
     def __isub__(self, other:'Vector') -> 'Vector':
          return self - other
     
     def __neg__(self) -> 'Vector':
          return self * -1
     
     def __eq__(self, other:'Vector') -> bool:
          assert isinstance(other, Vector), "Can only compare a vector with another vector."
          return self.coords == other.coords
     
     def __ne__(self, other:'Vector') -> bool:
          return not self == other     
     
     def scale(self, scalar:float) -> 'Vector':
          return self * scalar
     
     def add(self, other:'Vector') -> 'Vector':
          return self + other
     
     def subtract(self, other:'Vector') -> 'Vector':
          return self - other
     
     def norm(self, p:float=2) -> float:
          """Returns the p-norm of the vector."""
          return sum(abs(coord) ** p for coord in self.coords) ** (1 / p)
     
     def magnitude(self):
          """Returns the magnitude (length) of the vector. Magnitude is the same as the 2-norm."""
          return self.norm()
     
     
     def from_origin(self) -> 'Vector':
          """Returns a vector from the origin to the coordinates."""
          origin_point = CeroPoint(dimension=len(self.coords))
          return Vector(
               start=origin_point,
               end=Point(self.coords, name=f"{self.end.name} - {self.start.name}")
          )
     
     def unitary(self) -> 'Vector':
          """Returns the unitary vector in the same direction as the original vector. The initial point remains the same."""
          magnitude:float = self.magnitude()
          if magnitude == 0:
               return self
          return Vector(
               start = self.start,
               end = Point(tuple(coord / magnitude for coord in self.coords), name=f"u{self.end.name}")
          )
     
     def normalize(self) -> 'Vector':
          """Returns the unitary vector from the origin to the coordinates of the original vector."""
          return self.from_origin().unitary()
     
class PlaneVector(Vector):
     def __init__(self, start:PlanePoint, end:PlanePoint, name:str = None):
          assert len(start) == 2 and len(end) == 2, "Both points must be 2D to define a PlaneVector."
          super().__init__(start, end, name)

class SpaceVector(Vector):
     def __init__(self, start:SpacePoint, end:SpacePoint, name:str = None):
          assert len(start) == 3 and len(end) == 3, "Both points must be 3D to define a SpaceVector."
          super().__init__(start, end, name)


class FreeVector(Vector):
     """A Free Vector is defined only by its coordinates, not by specific start and end points."""
     def __init__(self, coords:tuple[float], name:str = None):
          origin_point = CeroPoint(dimension=len(coords))
          end_point = Point(coords, name=f"P{coords}")
          super().__init__(origin_point, end_point, name=name)

class FreePlaneVector(PlaneVector):
     def __init__(self, coords:tuple[float], name:str = None):
          assert len(coords) == 2, "Coordinates must be 2D to define a FreePlaneVector."
          super().__init__(CeroPoint(dimension=2), Point(coords, name=f"P{coords}"), name=name)

class FreeSpaceVector(SpaceVector):
     def __init__(self, coords:tuple[float], name:str = None):
          assert len(coords) == 3, "Coordinates must be 3D to define a FreeSpaceVector."
          super().__init__(CeroPoint(dimension=3), Point(coords, name=f"P{coords}"), name=name)
