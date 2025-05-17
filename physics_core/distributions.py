# physics_core/distributions.py
from abc import ABC, abstractmethod
import numpy as np
from .constants import K_E, EPSILON_0
from scipy.integrate import quad # New import for integration

class ChargeDistribution(ABC):
    """Abstract base class for all charge distributions."""

    @abstractmethod
    def get_electric_field(self, x: float, y: float, z: float) -> np.ndarray:
        """
        Calculate the electric field vector (Ex, Ey, Ez) at a given point.
        Returns a numpy array [Ex, Ey, Ez].
        """
        pass

    @abstractmethod
    def get_potential(self, x: float, y: float, z: float) -> float:
        """
        Calculate the electric potential at a given point.
        """
        pass

    @abstractmethod
    def to_dict(self) -> dict:
        """Serialize the distribution to a dictionary for API responses or storage."""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict) -> 'ChargeDistribution':
        """Deserialize a distribution from a dictionary."""
        pass


class PointCharge(ChargeDistribution):
    def __init__(self, charge_q: float, position: np.ndarray):
        """
        Args:
            charge_q (float): Charge in Coulombs.
            position (np.ndarray): 1D array/list of 3 elements [x, y, z].
        """
        self.q = charge_q
        self.position = np.array(position, dtype=float)
        if self.position.shape != (3,):
            raise ValueError("Position must be a 3-element array.")

    def get_electric_field(self, x: float, y: float, z: float) -> np.ndarray:
        point_r = np.array([x, y, z])
        r_vec = point_r - self.position
        r_mag_sq = np.sum(r_vec**2)

        if r_mag_sq < 1e-12:  # Avoid division by zero near the charge's location
            return np.array([0.0, 0.0, 0.0]) # Field is infinite at the point charge location

        r_mag = np.sqrt(r_mag_sq)
        r_hat = r_vec / r_mag

        E_mag = K_E * self.q / r_mag_sq
        return E_mag * r_hat

    def get_potential(self, x: float, y: float, z: float) -> float:
        point_r = np.array([x, y, z])
        r_vec = point_r - self.position
        r_mag = np.linalg.norm(r_vec)

        if r_mag < 1e-12:
            return np.inf if self.q > 0 else -np.inf # Potential diverges

        return K_E * self.q / r_mag

    def to_dict(self) -> dict:
        return {
            "type": "point_charge",
            "charge_q": self.q,
            "position": self.position.tolist()
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'PointCharge':
        return cls(charge_q=data['charge_q'], position=np.array(data['position']))


class InfinitePlaneCharge(ChargeDistribution):
    def __init__(self, surface_charge_density_sigma: float, normal_vector: np.ndarray, point_on_plane: np.ndarray):
        """
        Args:
            surface_charge_density_sigma (float): Sigma in C/m^2.
            normal_vector (np.ndarray): Normal vector to the plane (e.g., [0,0,1] for xy-plane).
            point_on_plane (np.ndarray): Any point [x0, y0, z0] on the plane.
        """
        self.sigma = surface_charge_density_sigma
        normal_vector_arr = np.array(normal_vector, dtype=float)
        norm = np.linalg.norm(normal_vector_arr)
        if norm < 1e-9:
             raise ValueError("Normal vector magnitude is too small.")
        self.normal = normal_vector_arr / norm

        self.point_on_plane = np.array(point_on_plane, dtype=float)

        if self.normal.shape != (3,) or self.point_on_plane.shape != (3,):
            raise ValueError("Normal vector and point on plane must be 3-element arrays.")

    def get_electric_field(self, x: float, y: float, z: float) -> np.ndarray:
        point_r = np.array([x, y, z])
        # Vector from a point on the plane to the observation point
        vec_to_point = point_r - self.point_on_plane

        # Distance to plane along the normal (signed)
        distance_signed = np.dot(vec_to_point, self.normal)

        # Magnitude of E-field for an infinite plane
        E_mag = self.sigma / (2 * EPSILON_0)

        # Direction depends on which side of the plane the point is
        if abs(distance_signed) < 1e-9: # Point is very close to or on the plane
            return np.array([0.0, 0.0, 0.0]) # Field is undefined or half on the surface

        direction = np.sign(distance_signed) * self.normal
        return E_mag * direction

    def get_potential(self, x: float, y: float, z: float) -> float:
        # Potential for an infinite plane depends on a reference point.
        # We'll define potential V=0 on the plane itself. V = -E * d
        point_r = np.array([x, y, z])
        vec_to_point = point_r - self.point_on_plane
        distance_signed = np.dot(vec_to_point, self.normal)

        E_mag_component = self.sigma / (2 * EPSILON_0) # Magnitude of E component along normal
        return -E_mag_component * distance_signed

    def to_dict(self) -> dict:
        return {
            "type": "infinite_plane_charge",
            "sigma": self.sigma,
            "normal_vector": self.normal.tolist(),
            "point_on_plane": self.point_on_plane.tolist()
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'InfinitePlaneCharge':
        return cls(
            surface_charge_density_sigma=data['sigma'],
            normal_vector=np.array(data['normal_vector']),
            point_on_plane=np.array(data['point_on_plane'])
        )

class ChargedSphereShell(ChargeDistribution):
    def __init__(self, total_charge_q: float, radius: float, center: np.ndarray):
        self.q = total_charge_q
        self.radius = radius
        self.center = np.array(center, dtype=float)
        if self.radius < 0:
            raise ValueError("Radius cannot be negative.")
        if self.center.shape != (3,):
            raise ValueError("Center must be a 3-element array.")

    def get_electric_field(self, x: float, y: float, z: float) -> np.ndarray:
        point_r = np.array([x, y, z])
        r_vec = point_r - self.center
        r_mag_sq = np.sum(r_vec**2)
        r_mag = np.sqrt(r_mag_sq)

        if r_mag < self.radius: # Inside the shell
            # For r=0 and radius=0, it's a point charge, handled by PointCharge class.
            # For r=0 and radius > 0, inside a shell, E=0.
            # For r > 0 and r < radius, inside a shell, E=0.
            return np.array([0.0, 0.0, 0.0])
        elif abs(r_mag - self.radius) < 1e-9: # On the surface
             # Field is often considered discontinuous here. Return average or limit from outside.
             # For visualization purposes, using the limit from outside is common.
             r_mag = self.radius # Use radius to avoid near-singularity issues

        # Outside the shell (r > radius) or effectively on the surface
        r_hat = r_vec / r_mag
        E_mag = K_E * self.q / r_mag_sq
        return E_mag * r_hat


    def get_potential(self, x: float, y: float, z: float) -> float:
        point_r = np.array([x, y, z])
        r_vec = point_r - self.center
        r_mag = np.linalg.norm(r_vec)

        if r_mag < self.radius: # Inside
            # Handle r=0 and radius=0 (point charge at origin) separately if necessary,
            # but the PointCharge class handles this case already.
            if self.radius < 1e-9: # Effectively a point charge, potential diverges
                 return np.inf if self.q > 0 else -np.inf
            return K_E * self.q / self.radius # Potential is constant inside

        else: # Outside or on the surface
            # Handle r=0, radius=0 (point charge at origin)
            if r_mag < 1e-12 and self.radius < 1e-9:
                 return np.inf if self.q > 0 else -np.inf
            return K_E * self.q / r_mag

    def to_dict(self) -> dict:
        return {
            "type": "charged_sphere_shell",
            "total_charge_q": self.q,
            "radius": self.radius,
            "center": self.center.tolist()
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ChargedSphereShell':
        return cls(
            total_charge_q=data['total_charge_q'],
            radius=data['radius'],
            center=np.array(data['center'])
        )

class FiniteLineCharge(ChargeDistribution):
    def __init__(self, total_charge_q: float, start_point: np.ndarray, end_point: np.ndarray):
        self.q = total_charge_q
        self.p1 = np.array(start_point, dtype=float)
        self.p2 = np.array(end_point, dtype=float)

        if self.p1.shape != (3,) or self.p2.shape != (3,):
            raise ValueError("Start and end points must be 3-element arrays.")

        self.line_vector = self.p2 - self.p1
        self.length = np.linalg.norm(self.line_vector)

        if self.length < 1e-9: # Effectively a point charge if length is negligible
            self.is_point_like = True
            self.lambda_val = 0 # Not used in point-like case
            # Potential singularity handled by point charge logic
        else:
            self.is_point_like = False
            self.lambda_val = self.q / self.length

    def _distance_point_to_segment(self, point: np.ndarray) -> float:
        if self.length < 1e-9:
            return np.linalg.norm(point - self.p1)

        # Vector from p1 to point
        ap = point - self.p1
        # Vector from p1 to p2 (line_vector)
        ab = self.line_vector

        # Project ap onto ab (t is parameter along line segment)
        # t = dot(ap, ab) / |ab|^2
        t = np.dot(ap, ab) / (self.length**2)

        if t < 0.0:
            closest_point = self.p1 # Closest point is start_point
        elif t > 1.0:
            closest_point = self.p2 # Closest point is end_point
        else:
            closest_point = self.p1 + t * ab # Closest point is on the segment

        return np.linalg.norm(point - closest_point)

    def get_electric_field(self, x: float, y: float, z: float) -> np.ndarray:
        point_r = np.array([x, y, z])

        if self.is_point_like: # Treat as point charge at p1
            r_vec = point_r - self.p1
            r_mag_sq = np.sum(r_vec**2)
            if r_mag_sq < 1e-12: return np.array([0.0, 0.0, 0.0])
            r_mag = np.sqrt(r_mag_sq)
            r_hat = r_vec / r_mag
            E_mag = K_E * self.q / r_mag_sq
            return E_mag * r_hat

        # Check if observation point is on or very close to the line segment
        if self._distance_point_to_segment(point_r) < 1e-9:
            # Field is technically infinite/undefined on an ideal line charge.
            # Return zero for visualization stability, or NaN/Inf if strict.
            return np.array([0.0, 0.0, 0.0])

        # Calculate E-field using integration: E = Integral[ (K_E * dq / r^2) * r_hat ]
        # dq = lambda * dl = (q / Length) * Length * dt_param = q * dt_param for t_param in [0,1]
        # r_vec = point_r - (p1 + t_param * line_vector)
        # r_mag = ||r_vec||
        # r_hat = r_vec / r_mag
        # E = Integral[ K_E * q * (r_vec / r_mag^3) dt_param ] from 0 to 1

        def integrand_Ex(t_param):
            pos_on_line = self.p1 + t_param * self.line_vector
            r_vec = point_r - pos_on_line
            r_mag_cubed = np.linalg.norm(r_vec)**3
            if r_mag_cubed < 1e-18: return 0.0 # Avoid numerical issues near singularity
            return r_vec[0] / r_mag_cubed

        def integrand_Ey(t_param):
            pos_on_line = self.p1 + t_param * self.line_vector
            r_vec = point_r - pos_on_line
            r_mag_cubed = np.linalg.norm(r_vec)**3
            if r_mag_cubed < 1e-18: return 0.0
            return r_vec[1] / r_mag_cubed

        def integrand_Ez(t_param):
            pos_on_line = self.p1 + t_param * self.line_vector
            r_vec = point_r - pos_on_line
            r_mag_cubed = np.linalg.norm(r_vec)**3
            if r_mag_cubed < 1e-18: return 0.0
            return r_vec[2] / r_mag_cubed

        # Common factor K_E * q
        common_factor = K_E * self.q

        # Use scipy.integrate.quad for numerical integration
        # Integration limits are 0 to 1 for the normalized parameter t_param
        Ex, err_Ex = quad(integrand_Ex, 0, 1, epsabs=1e-8, epsrel=1e-8)
        Ey, err_Ey = quad(integrand_Ey, 0, 1, epsabs=1e-8, epsrel=1e-8)
        Ez, err_Ez = quad(integrand_Ez, 0, 1, epsabs=1e-8, epsrel=1e-8)

        # You might want to check err_Ex, err_Ey, err_Ez for potential integration issues
        # and return a warning or error, especially if the point is very close to the line.

        return common_factor * np.array([Ex, Ey, Ez])

    def get_potential(self, x: float, y: float, z: float) -> float:
        point_r = np.array([x, y, z])

        if self.is_point_like: # Treat as point charge at p1
            r_vec = point_r - self.p1
            r_mag = np.linalg.norm(r_vec)
            if r_mag < 1e-12: return np.inf if self.q > 0 else -np.inf
            return K_E * self.q / r_mag

        # Check if observation point is on or very close to the line segment
        if self._distance_point_to_segment(point_r) < 1e-9:
            return np.inf if self.q > 0 else -np.inf # Potential diverges

        # Calculate Potential using integration: V = Integral[ K_E * dq / r ]
        # V = Integral[ K_E * q / r * dt_param ] from 0 to 1

        def integrand_V(t_param):
            pos_on_line = self.p1 + t_param * self.line_vector
            r_vec = point_r - pos_on_line
            r_mag = np.linalg.norm(r_vec)
            if r_mag < 1e-12: return 0.0 # Should be handled by distance check, but defensive
            return 1.0 / r_mag

        common_factor = K_E * self.q
        V, err_V = quad(integrand_V, 0, 1, epsabs=1e-8, epsrel=1e-8)

        return common_factor * V

    def to_dict(self) -> dict:
        return {
            "type": "finite_line_charge",
            "total_charge_q": self.q,
            "start_point": self.p1.tolist(),
            "end_point": self.p2.tolist()
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'FiniteLineCharge':
        return cls(
            total_charge_q=data.get('total_charge_q', 0.0), # Provide default
            start_point=np.array(data['start_point']),
            end_point=np.array(data['end_point'])
        )


# Factory to create distribution objects from dictionary data
DISTRIBUTION_CLASSES = {
    "point_charge": PointCharge,
    "infinite_plane_charge": InfinitePlaneCharge,
    "charged_sphere_shell": ChargedSphereShell,
    "finite_line_charge": FiniteLineCharge, # Added new type
}

def create_distribution_from_dict(data: dict) -> ChargeDistribution:
    """Creates a ChargeDistribution object from a dictionary representation."""
    dist_type = data.get("type")
    if not dist_type or dist_type not in DISTRIBUTION_CLASSES:
        raise ValueError(f"Unknown or missing distribution type: '{dist_type}'. Available types: {list(DISTRIBUTION_CLASSES.keys())}")

    klass = DISTRIBUTION_CLASSES[dist_type]
    try:
        return klass.from_dict(data)
    except Exception as e:
        # Wrap exceptions from from_dict for better error reporting
        raise ValueError(f"Error creating distribution of type '{dist_type}' from data {data}: {e}")