import numpy as np
from typing import Tuple, Callable
import dataclasses

def initialize_bond_distribution_fun_numpy(bdf_params):
    """
    NumPy version of initialize_bond_distribution_fun using numpy.histogram.
    
    This function creates a bond distribution function that computes the 
    instantaneous bond correlation function using Gaussian smoothing.
    
    Args:
        bdf_params: Parameters containing bin_centers, bin_boundaries, sigma, and bond_top
        
    Returns:
        A function that computes bond distribution from system positions
    """
    _, bdf_bin_centers, bdf_bin_boundaries, sigma, bond_top = dataclasses.astuple(bdf_params)
    bin_size = np.diff(bdf_bin_boundaries)
    
    def bond_corr_fun_numpy(system, **dynamic_kwargs):
        """
        NumPy implementation of bond correlation function.
        
        Args:
            system: Object with attributes R (positions) and cell (simulation box)
            
        Returns:
            mean_bond_corr: Normalized bond distribution function
        """
        positions = system.R  # Assuming this is a numpy array
        
        # Extract bonded atom positions
        Ra = positions[bond_top[:, 0], :]  # First atoms in bonds
        Rb = positions[bond_top[:, 1], :]  # Second atoms in bonds
        
        # Calculate displacement vectors with periodic boundary conditions
        if hasattr(system, 'cell') and system.cell is not None:
            edges = Rb - Ra
            # Apply minimum image convention
            edges = edges - system.cell * np.round(edges / system.cell)
        else:
            edges = Rb - Ra
            
        # Calculate distances
        dr = np.linalg.norm(edges, axis=1)
        
        # Method 1: Using numpy histogram with post-smoothing
        bond_corr_hist = numpy_histogram_with_gaussian_smoothing(
            dr, bdf_bin_centers, bdf_bin_boundaries, sigma
        )
        
        # Method 2: Direct Gaussian smoothing (equivalent to original JAX)
        bond_corr_gaussian = numpy_direct_gaussian_smoothing(
            dr, bdf_bin_centers, bin_size, sigma
        )
        
        return bond_corr_gaussian  # Use the direct method for closest equivalence
    
    def bdf_compute_fun_numpy(system, **unused_kwargs):
        """Compute bond distribution function."""
        bdf = bond_corr_fun_numpy(system)
        return bdf
        
    return bond_corr_fun_numpy, bdf_compute_fun_numpy


def numpy_histogram_with_gaussian_smoothing(distances: np.ndarray,
                                           bin_centers: np.ndarray,
                                           bin_boundaries: np.ndarray,
                                           sigma: float) -> np.ndarray:
    """
    Approach 1: Use numpy.histogram then apply Gaussian smoothing.
    
    Args:
        distances: Bond distances
        bin_centers: Centers of bins
        bin_boundaries: Boundaries of bins  
        sigma: Gaussian smoothing parameter
        
    Returns:
        Smoothed and normalized bond correlation function
    """
    # Create histogram using numpy
    hist, _ = np.histogram(distances, bins=bin_boundaries)
    
    # Convert to float and apply Gaussian smoothing
    hist = hist.astype(float)
    smoothed = np.zeros_like(bin_centers, dtype=float)
    
    # Apply Gaussian kernel to smooth the histogram
    for i, center_i in enumerate(bin_centers):
        gaussian_kernel = np.exp(-0.5 * (bin_centers - center_i)**2 / sigma**2)
        gaussian_kernel /= np.sqrt(2 * np.pi * sigma**2)
        smoothed[i] = np.sum(hist * gaussian_kernel)
    
    # Normalize using trapezoidal integration
    integral = np.trapz(smoothed, bin_centers)
    if integral > 0:
        smoothed /= integral
        
    return smoothed


def numpy_direct_gaussian_smoothing(distances: np.ndarray,
                                   bin_centers: np.ndarray,
                                   bin_size: np.ndarray,
                                   sigma: float) -> np.ndarray:
    """
    Approach 2: Direct Gaussian smoothing (closest to original JAX implementation).
    
    This directly translates the JAX code:
    exp = jnp.exp(-0.5 * (dr[:, jnp.newaxis] - bdf_bin_centers) ** 2 / sigma ** 2)
    gaussian_distances = exp * bin_size / jnp.sqrt(2 * jnp.pi * sigma ** 2)
    bond_corr = high_precision_sum(gaussian_distances, axis=0)
    mean_bond_corr = bond_corr / jnp.trapz(bond_corr, bdf_bin_centers)
    
    Args:
        distances: Bond distances, shape (n_bonds,)
        bin_centers: Bin centers, shape (n_bins,)
        bin_size: Size of each bin, shape (n_bins,)
        sigma: Gaussian smoothing parameter
        
    Returns:
        Normalized bond correlation function
    """
    # Expand dimensions for broadcasting: distances (n_bonds, 1), bin_centers (1, n_bins)
    dr_expanded = distances[:, np.newaxis]  # Shape: (n_bonds, 1)
    bin_centers_expanded = bin_centers[np.newaxis, :]  # Shape: (1, n_bins)
    
    # Gaussian exponent: exp(-0.5 * (dr - bin_center)^2 / sigma^2)
    exp = np.exp(-0.5 * (dr_expanded - bin_centers_expanded)**2 / sigma**2)
    
    # Gaussian probability density function
    gaussian_distances = exp * bin_size / np.sqrt(2 * np.pi * sigma**2)
    
    # Sum over all bonds (axis=0) - equivalent to high_precision_sum
    bond_corr = np.sum(gaussian_distances, axis=0)
    
    # Normalize using trapezoidal integration
    integral = np.trapz(bond_corr, bin_centers)
    if integral > 0:
        mean_bond_corr = bond_corr / integral
    else:
        mean_bond_corr = bond_corr
        
    return mean_bond_corr


def numpy_weighted_histogram_approach(distances: np.ndarray,
                                     bin_boundaries: np.ndarray,
                                     sigma: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Approach 3: Use numpy.histogram with custom weights for Gaussian smoothing.
    
    This approach uses the weights parameter of numpy.histogram to implement
    Gaussian smoothing directly during the histogram creation.
    
    Args:
        distances: Bond distances
        bin_boundaries: Histogram bin boundaries
        sigma: Gaussian smoothing parameter
        
    Returns:
        bin_centers: Centers of bins
        weighted_hist: Gaussian-weighted histogram
    """
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    n_bins = len(bin_centers)
    
    # For each distance, create Gaussian weights for all bins
    weighted_hist = np.zeros(n_bins)
    
    for distance in distances:
        # Calculate Gaussian weights for this distance across all bins
        weights = np.exp(-0.5 * (bin_centers - distance)**2 / sigma**2)
        weights /= np.sqrt(2 * np.pi * sigma**2)  # Normalize Gaussian
        
        # Add weighted contribution to histogram
        # We can use histogram with weights, but it's simpler to just add directly
        weighted_hist += weights
    
    # Normalize
    integral = np.trapz(weighted_hist, bin_centers)
    if integral > 0:
        weighted_hist /= integral
        
    return bin_centers, weighted_hist


# Example usage showing how to replace the JAX function
if __name__ == "__main__":
    # Mock system class for testing
    class MockSystem:
        def __init__(self, positions, cell=None):
            self.R = positions
            self.cell = cell
    
    # Mock BDF parameters
    @dataclasses.dataclass
    class BDFParams:
        unused: None
        bin_centers: np.ndarray
        bin_boundaries: np.ndarray
        sigma: float
        bond_top: np.ndarray
    
    # Create test data
    np.random.seed(42)
    n_particles = 50
    positions = np.random.rand(n_particles, 3) * 10
    
    # Create bond topology
    n_bonds = 25
    bond_topology = np.column_stack([
        np.random.choice(n_particles, n_bonds, replace=False),
        np.random.choice(n_particles, n_bonds, replace=False)
    ])
    
    # Create BDF parameters
    bin_boundaries = np.linspace(0, 5, 26)  # 25 bins
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    sigma = 0.1
    
    bdf_params = BDFParams(
        unused=None,
        bin_centers=bin_centers,
        bin_boundaries=bin_boundaries,
        sigma=sigma,
        bond_top=bond_topology
    )
    
    # Initialize the numpy version
    bond_corr_fun, bdf_compute_fun = initialize_bond_distribution_fun_numpy(bdf_params)
    
    # Create mock system
    system = MockSystem(positions, cell=np.array([10.0, 10.0, 10.0]))
    
    # Compute bond distribution
    result = bond_corr_fun(system)
    
    print(f"Bond distribution computed successfully!")
    print(f"Result shape: {result.shape}")
    print(f"Result sum: {np.sum(result):.6f}")
    print(f"Result integral: {np.trapz(result, bin_centers):.6f}")
    
    # Compare different methods
    distances = np.linalg.norm(
        positions[bond_topology[:, 1]] - positions[bond_topology[:, 0]], 
        axis=1
    )
    
    method1 = numpy_histogram_with_gaussian_smoothing(distances, bin_centers, bin_boundaries, sigma)
    method2 = numpy_direct_gaussian_smoothing(distances, bin_centers, np.diff(bin_boundaries), sigma)
    bin_centers_w, method3 = numpy_weighted_histogram_approach(distances, bin_boundaries, sigma)
    
    print(f"\nMethod comparison:")
    print(f"Histogram + smoothing integral: {np.trapz(method1, bin_centers):.6f}")
    print(f"Direct Gaussian integral: {np.trapz(method2, bin_centers):.6f}")
    print(f"Weighted histogram integral: {np.trapz(method3, bin_centers_w):.6f}")
    print(f"Function result integral: {np.trapz(result, bin_centers):.6f}")
