import pytest
import math

# Ajusta las importaciones para que Pytest pueda acceder a tus clases.
# Asumo que Point y CeroPoint están disponibles de alguna forma.
# EJEMPLO: Si están en 'points.py' y 'vectors.py' en el mismo directorio.
from athanor import Point, CeroPoint, Vector, FreeVector, FreePlaneVector, FreeSpaceVector, PlaneVector, SpaceVector, PlanePoint, SpacePoint

# --- SETUP (Fixtures para Pytest) ---

# Puntos base para la creación de vectores
@pytest.fixture
def p_start():
    """Punto de inicio A: (1.0, 2.0)"""
    return Point((1.0, 2.0), name="A")

@pytest.fixture
def p_end():
    """Punto final B: (4.0, 6.0)"""
    return Point((4.0, 6.0), name="B")

@pytest.fixture
def p_cero_3d():
    """Punto Origen 3D: (0.0, 0.0, 0.0)"""
    return CeroPoint(dimension=3, name="O")

@pytest.fixture
def p_final_3d():
    """Punto final 3D: (4.0, 1.0, 4.0)"""
    return Point((4.0, 1.0, 4.0), name="E")

# Vectores base
@pytest.fixture
def v_2d(p_start, p_end):
    """Vector 2D: (3.0, 4.0). Magnitud: 5.0"""
    return Vector(p_start, p_end)

@pytest.fixture
def v_3d(p_cero_3d, p_final_3d):
    """Vector 3D: (4.0, 1.0, 4.0). Magnitud: sqrt(33)"""
    return Vector(p_cero_3d, p_final_3d)

@pytest.fixture
def v_another_2d(p_start, p_end):
    """Otro Vector 2D, con coordenadas diferentes: (2.0, 1.0)"""
    p_diff_start = Point((0.0, 0.0))
    p_diff_end = Point((2.0, 1.0))
    return Vector(p_diff_start, p_diff_end)

# Tolerancia para comparaciones de punto flotante
TOLERANCE = 1e-9

# --- TESTS PARA LA CLASE BASE VECTOR ---

class TestBaseVector:

    ## 1. Inicialización y Propiedades Básicas
    
    def test_initialization_and_coords(self, v_2d):
        """Prueba la creación del vector y el cálculo correcto de coordenadas y longitud."""
        assert v_2d.coords == (3.0, 4.0)
        assert len(v_2d) == 2

    def test_default_name(self, v_2d):
        """Prueba que el nombre por defecto se genera correctamente."""
        assert v_2d.name == "AB"

    def test_custom_name(self, p_start, p_end):
        """Prueba la asignación de un nombre personalizado."""
        v = Vector(p_start, p_end, name="V1")
        assert v.name == "V1"

    ## 2. Operaciones Aritméticas (Sobrecarga de Operadores)

    def test_scalar_multiplication(self, v_2d):
        """Prueba la multiplicación escalar (*) con positivos, negativos y cero."""
        v_scaled = v_2d * 2.5
        assert v_scaled.coords == (7.5, 10.0)
        
        v_neg_scaled = v_2d * -1
        assert v_neg_scaled.coords == (-3.0, -4.0)
        
        v_zero = v_2d * 0
        assert v_zero.coords == (0.0, 0.0)
        
    def test_negation_operator(self, v_3d):
        """Prueba el operador unario de negación (-)."""
        v_negated = -v_3d # (4, 1, 4) -> (-4, -1, -4)
        assert v_negated.coords == (-4.0, -1.0, -4.0)
        
    def test_vector_addition(self, v_2d, v_another_2d):
        """Prueba la suma de vectores (+)."""
        v_sum = v_2d + v_another_2d # (3, 4) + (2, 1) = (5, 5)
        assert v_sum.coords == (5.0, 5.0)

    def test_vector_subtraction(self, v_2d, v_another_2d):
        """Prueba la resta de vectores (-)."""
        v_diff = v_2d - v_another_2d # (3, 4) - (2, 1) = (1, 3)
        assert v_diff.coords == (1.0, 3.0)
        
    def test_in_place_addition(self, v_2d, v_another_2d):
        """Prueba el operador de suma in-place (+=)."""
        v_test = v_2d
        v_test += v_another_2d
        # Como tu implementación crea un nuevo Vector, probamos que la reasignación funciona
        assert v_test.coords == (5.0, 5.0)

    def test_operator_assertions(self, v_2d):
        """Prueba las validaciones de tipos para operaciones."""
        # Multiplicación escalar
        with pytest.raises(AssertionError, match="Can only multiply"):
            v_2d * "string"
        # Suma de vectores
        with pytest.raises(AssertionError, match="Can only add a vector"):
            v_2d + (1, 2)

    ## 3. Geometría y Magnitud

    def test_magnitude(self, v_2d, v_3d):
        """Prueba la magnitud (norma-2)."""
        # (3, 4) -> 5.0
        assert math.isclose(v_2d.magnitude(), 5.0, abs_tol=TOLERANCE)
        # (4, 1, 4) -> sqrt(16 + 1 + 16) = sqrt(33)
        assert math.isclose(v_3d.magnitude(), math.sqrt(33), abs_tol=TOLERANCE)

    def test_norm_p(self, v_2d):
        """Prueba la p-norma (L1-norm)."""
        # Norma-1: |3| + |4| = 7.0
        assert math.isclose(v_2d.norm(p=1), 7.0, abs_tol=TOLERANCE)
        
    def test_from_origin(self, v_2d):
        """Prueba la creación del vector equivalente partiendo del origen."""
        v_origin = v_2d.from_origin()
        assert v_origin.start.coords == (0.0, 0.0)
        assert v_origin.end.coords == v_2d.coords # (3.0, 4.0)

    def test_unitary(self, v_2d):
        # vectors.py (Tu código)
        def unitary(self) -> 'Vector':
            """Returns the unitary vector... The initial point remains the same."""
            magnitude:float = self.magnitude()
            if magnitude == 0:
                return self
                
            # 1. Coordenadas unitarias calculadas (e.g., (0.6, 0.8)) <-- Esto es correcto.
            unit_coords = tuple(coord / magnitude for coord in self.coords) 
            
            return Vector(
                start = self.start, # <-- Punto inicial A(1, 2)
                end = Point(unit_coords, name=f"u{self.end.name}") # <-- Nuevo punto final P'(0.6, 0.8)
            )

    def test_normalize(self, v_2d):
        """Prueba la normalización (unitario desde el origen)."""
        v_norm = v_2d.normalize()
        assert math.isclose(v_norm.magnitude(), 1.0, abs_tol=TOLERANCE)
        assert v_norm.start.coords == (0.0, 0.0)
        
    def test_unitary_zero_vector(self, v_3d):
        """Prueba el vector unitario de un vector nulo (debe devolver el mismo vector)."""
        v_zero = Vector(v_3d.start, v_3d.start)
        assert v_zero.unitary() is v_zero
        
    ## 4. Comparación

    def test_equality(self, v_2d):
        """Prueba la igualdad (==) basada únicamente en coordenadas."""
        # Vector con los mismos componentes, pero puntos diferentes
        v_equivalent = Vector(Point((0.0, 0.0)), Point((3.0, 4.0))) 
        
        assert v_2d == v_equivalent
        assert v_2d != (v_2d * 0.9)


# PLANE AND SPACE VECTOR TESTS
@pytest.fixture
def pp1():
    """PlanePoint P1: (1.0, 2.0)"""
    return PlanePoint(1.0, 2.0, name="P1")

@pytest.fixture
def pp2():
    """PlanePoint P2: (5.0, 5.0)"""
    return PlanePoint(5.0, 5.0, name="P2")

@pytest.fixture
def sp3():
    """SpacePoint S3: (1.0, 2.0, 3.0)"""
    return SpacePoint(1.0, 2.0, 3.0, name="S3")

@pytest.fixture
def sp4():
    """SpacePoint S4: (4.0, 0.0, 5.0)"""
    return SpacePoint(4.0, 0.0, 5.0, name="S4")

# Tolerance for floating point comparisons
TOLERANCE = 1e-9

# --- TESTS FOR DIMENSIONAL VALIDATION ---

@pytest.mark.dimensional
class TestDimensionalVectors:

    ### 1. PlaneVector (2D Validation)

    def test_planevector_initialization_valid(self, pp1, pp2):
        """Tests successful creation of a PlaneVector using two PlanePoints (2D)."""
        pv = PlaneVector(pp1, pp2, name="PV_OK")
        
        # Coords: (5-1, 5-2) = (4.0, 3.0)
        assert pv.coords == (4.0, 3.0)
        assert len(pv) == 2
        
        # Check that Vector functionality is inherited (magnitude 5.0)
        assert math.isclose(pv.magnitude(), 5.0, abs_tol=TOLERANCE)

    def test_planevector_initialization_invalid_end(self, pp1, sp4):
        """Tests that PlaneVector fails if the end point is not 2D (e.g., using a SpacePoint)."""
        
        # PlanePoint (2D) -> SpacePoint (3D)
        with pytest.raises(AssertionError, match="Both points must be 2D"):
            PlaneVector(pp1, sp4)

    def test_planevector_initialization_invalid_start(self, sp4, pp2):
        """Tests that PlaneVector fails if the starting point is not 2D (e.g., using a SpacePoint)."""
        
        # SpacePoint (3D) -> PlanePoint (2D)
        with pytest.raises(AssertionError, match="Both points must be 2D"):
            PlaneVector(sp4, pp2)


    ### 2. SpaceVector (3D Validation)

    def test_spacevector_initialization_valid(self, sp3, sp4):
        """Tests successful creation of a SpaceVector using two SpacePoints (3D)."""
        sv = SpaceVector(sp3, sp4, name="SV_OK")
        
        # Coords: (4-1, 0-2, 5-3) = (3.0, -2.0, 2.0)
        assert sv.coords == (3.0, -2.0, 2.0)
        assert len(sv) == 3
        
        # Magnitude: sqrt(9 + 4 + 4) = sqrt(17)
        assert math.isclose(sv.magnitude(), math.sqrt(17), abs_tol=TOLERANCE)

    def test_spacevector_initialization_invalid_end(self, sp3, pp2):
        """Tests that SpaceVector fails if the end point is not 3D (e.g., using a PlanePoint)."""
        
        # SpacePoint (3D) -> PlanePoint (2D)
        with pytest.raises(AssertionError, match="Both points must be 3D"):
            SpaceVector(sp3, pp2)

    def test_spacevector_initialization_invalid_start(self, pp2, sp4):
        """Tests that SpaceVector fails if the starting point is not 3D (e.g., using a PlanePoint)."""
        
        # PlanePoint (2D) -> SpacePoint (3D)
        with pytest.raises(AssertionError, match="Both points must be 3D"):
            SpaceVector(pp2, sp4)

# FREE VECTOR TESTS

@pytest.fixture
def free_v_2d():
    """FreeVector: (3.0, 4.0)"""
    return FreeVector((3.0, 4.0), name="FV2D")

@pytest.fixture
def free_pv():
    """FreePlaneVector: (10.0, -5.0)"""
    return FreePlaneVector((10.0, -5.0), name="FPV")

@pytest.fixture
def free_sv():
    """FreeSpaceVector: (1.0, 2.0, 3.0)"""
    return FreeSpaceVector((1.0, 2.0, 3.0), name="FSV")

# Tolerancia para comparaciones de punto flotante
TOLERANCE = 1e-9


# --- TESTS PARA LA CLASE BASE FREEVECTOR ---

@pytest.mark.freevector
class TestFreeVector:

    def test_freevector_initialization(self, free_v_2d):
        """Prueba que FreeVector se inicializa correctamente y parte del origen."""
        
        # 1. Las coordenadas deben ser las suministradas
        assert free_v_2d.coords == (3.0, 4.0)
        
        # 2. El punto de inicio debe ser el Origen (CeroPoint)
        assert free_v_2d.start.coords == (0.0, 0.0)
        
        # 3. El punto final debe tener las coordenadas del vector
        assert free_v_2d.end.coords == (3.0, 4.0)
        
        # 4. Prueba la herencia (debe tener magnitud)
        assert math.isclose(free_v_2d.magnitude(), 5.0, abs_tol=TOLERANCE)
        
    def test_freevector_3d(self):
        """Prueba un FreeVector 3D para asegurar que la dimensión es genérica."""
        fv_3d = FreeVector((1.0, 0.0, -2.0))
        assert fv_3d.coords == (1.0, 0.0, -2.0)
        assert len(fv_3d) == 3
        
        # Magnitud: sqrt(1^2 + 0^2 + (-2)^2) = sqrt(5)
        assert math.isclose(fv_3d.magnitude(), math.sqrt(5), abs_tol=TOLERANCE)

    def test_freevector_arithmetic_inheritance(self, free_v_2d, free_sv):
        """Prueba que las operaciones aritméticas de Vector base funcionan por herencia."""
        
        # Prueba Suma (debe funcionar si __add__ está corregido)
        v_sum = free_v_2d + FreeVector((1.0, 1.0))
        assert v_sum.coords == (4.0, 5.0)
        
        # Prueba Negación (debe funcionar)
        v_neg = -free_v_2d
        assert v_neg.coords == (-3.0, -4.0)
        
        # Prueba Multiplicación (debe funcionar si __mul__ está corregido)
        v_scaled = free_sv * 2
        assert v_scaled.coords == (2.0, 4.0, 6.0)


# --- TESTS PARA CLASES HIJAS ESPECÍFICAS (Validaciones dimensionales) ---

@pytest.mark.freevector_subclasses
class TestFreeVectorSubclasses:

    def test_freeplanevector_initialization_valid(self, free_pv):
        """Prueba FreePlaneVector con coordenadas 2D válidas."""
        assert free_pv.coords == (10.0, -5.0)
        assert len(free_pv) == 2
        
        # El punto de inicio debe ser CeroPoint(2D)
        assert free_pv.start.coords == (0.0, 0.0)

    def test_freeplanevector_initialization_invalid(self):
        """Prueba la validación de dimensión (assert) en FreePlaneVector (solo 2D)."""
        # Intenta usar 3D
        with pytest.raises(AssertionError, match="Coordinates must be 2D"):
            FreePlaneVector((1.0, 2.0, 3.0))

    def test_freespacevector_initialization_valid(self, free_sv):
        """Prueba FreeSpaceVector con coordenadas 3D válidas."""
        assert free_sv.coords == (1.0, 2.0, 3.0)
        assert len(free_sv) == 3
        
        # El punto de inicio debe ser CeroPoint(3D)
        assert free_sv.start.coords == (0.0, 0.0, 0.0)

    def test_freespacevector_initialization_invalid(self):
        """Prueba la validación de dimensión (assert) en FreeSpaceVector (solo 3D)."""
        # Intenta usar 2D
        with pytest.raises(AssertionError, match="Coordinates must be 3D"):
            FreeSpaceVector((1.0, 2.0))
        # Intenta usar 4D
        with pytest.raises(AssertionError, match="Coordinates must be 3D"):
            FreeSpaceVector((1.0, 2.0, 3.0, 4.0))

