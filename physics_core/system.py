# physics_core/system.py
import numpy as np
from typing import List, Tuple
from .distributions import ChargeDistribution, create_distribution_from_dict

class ChargeSystem:
    """Represents a collection of charge distributions."""
    def __init__(self, distributions: List[ChargeDistribution] = None):
        self.distributions = distributions if distributions is not None else []

    def add_distribution(self, dist: ChargeDistribution):
        """Adds a single distribution to the system."""
        if not isinstance(dist, ChargeDistribution):
             raise TypeError("Only objects inheriting from ChargeDistribution can be added.")
        self.distributions.append(dist)

    def clear_distributions(self):
        """Removes all distributions from the system."""
        self.distributions = []

    def calculate_total_electric_field(self, x: float, y: float, z: float) -> np.ndarray:
        """Calculates the total electric field vector at a given point using superposition."""
        total_E_field = np.array([0.0, 0.0, 0.0])
        for dist in self.distributions:
            # Note: Handle potential errors/NaNs from individual distributions if needed
            # For now, we assume distributions return valid numpy arrays.
            try:
                 e_field = dist.get_electric_field(x, y, z)
                 total_E_field += e_field
            except Exception as e:
                 print(f"Warning: Error calculating field from {type(dist).__name__} at ({x},{y},{z}): {e}")
                 # Optionally, return NaN or raise error depending on desired behavior
                 pass # Continue with other distributions
        return total_E_field

    def calculate_total_potential(self, x: float, y: float, z: float) -> float:
        """Calculates the total electric potential at a given point."""
        total_potential = 0.0
        for dist in self.distributions:
             try:
                potential = dist.get_potential(x, y, z)
                # Check for infinity/NaN before adding
                if np.isinf(potential) or np.isnan(potential):
                     print(f"Warning: Potential is infinite/NaN from {type(dist).__name__} at ({x},{y},{z})")
                     return float(potential) # Return Inf/NaN immediately if any component is
                total_potential += potential
             except Exception as e:
                print(f"Warning: Error calculating potential from {type(dist).__name__} at ({x},{y},{z}): {e}")
                pass # Continue with other distributions

        return total_potential

    def calculate_field_on_grid(self,
                                x_coords: np.ndarray,
                                y_coords: np.ndarray,
                                z_coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates E-field on a grid of points formed by the Cartesian product
        of x_coords, y_coords, and z_coords.

        Returns:
            points (np.ndarray): N_points x 3 array of grid point coordinates.
            vectors (np.ndarray): N_points x 3 array of electric field vectors at these points.
        """
        # Create the grid points
        grid_x, grid_y, grid_z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
        points = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T # N_points x 3 array

        # Calculate vectors at each point
        vectors = np.zeros_like(points) # N_points x 3 array
        # This loop is the most computationally intensive part for large grids
        for i, p in enumerate(points):
            vectors[i] = self.calculate_total_electric_field(p[0], p[1], p[2])

        return points, vectors

    @classmethod
    def from_dict_list(cls, dict_list: List[dict]) -> 'ChargeSystem':
        """Creates a ChargeSystem from a list of distribution dictionaries."""
        distributions = [create_distribution_from_dict(d) for d in dict_list]
        return cls(distributions)