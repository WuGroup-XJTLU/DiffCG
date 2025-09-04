import numpy as np
from typing import Tuple, Optional
import dataclasses

def numpy_bond_distribution_function(positions: np.ndarray, 
                                    bond_topology: np.ndarray,
                                    bin_boundaries: np.ndarray,
                                    sigma: float,
                                    cell: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute bond distribution function using NumPy histogram with Gaussian smoothing.
    
    This is a NumPy translation of the JAX-based bond distribution function that uses
    Gaussian smoothing instead of simple histogram binning.
    
    Args:
        positions: Array of shape (N, 3) containing particle positions
        bond_topology: Array of shape (M, 2) containing bond pairs (indices)
        bin_boundaries: Array of bin boundaries for the histogram
        sigma: Gaussian smoothing parameter
        cell: Optional simulation cell for periodic boundary conditions
        
    Returns:
        bin_centers: Centers of histogram bins
        bond_distribution: Normalized bond distribution function
    """
    
    # Calculate bin centers and bin sizes
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    bin_size = np.diff(bin_boundaries)
    
    # Extract bonded atom positions
    Ra = positions[bond_topology[:, 0]]  # First atom in each bond
    Rb = positions[bond_topology[:, 1]]  # Second atom in each bond
    
    # Calculate bond vectors (displacement)
    if cell is not None:
        # Apply periodic boundary conditions
        edges = Rb - Ra
        # Minimum image convention for periodic boundaries
        edges = edges - cell * np.round(edges / cell)
    else:
        edges = Rb - Ra
    
    # Calculate bond distances
    bond_distances = np.linalg.norm(edges, axis=1)
    
    # Method 1: Using numpy.histogram with Gaussian smoothing
    # This is the most direct translation using histogram
    bond_distribution_hist = numpy_gaussian_histogram(bond_distances, bin_boundaries, sigma)
    
    # Method 2: Direct Gaussian smoothing (equivalent to original JAX implementation)
    bond_distribution_gaussian = numpy_gaussian_smoothing(bond_distances, bin_centers, bin_size, sigma)
    
    return bin_centers, bond_distribution_hist, bond_distribution_gaussian


def numpy_gaussian_histogram(distances: np.ndarray, 
                           bin_boundaries: np.ndarray, 
                           sigma: float) -> np.ndarray:
    """
    Create a Gaussian-smoothed histogram using numpy.histogram as the base.
    
    Args:
        distances: Array of bond distances
        bin_boundaries: Histogram bin boundaries
        sigma: Gaussian smoothing parameter
        
    Returns:
        Normalized Gaussian-smoothed distribution
    """
    # Create initial histogram
    hist, _ = np.histogram(distances, bins=bin_boundaries)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    bin_width = np.diff(bin_boundaries)[0]  # Assuming uniform bins
    
    # Apply Gaussian smoothing to the histogram
    smoothed_hist = np.zeros_like(hist, dtype=float)
    
    for i, center in enumerate(bin_centers):
        # For each bin, add contributions from nearby bins weighted by Gaussian
        gaussian_weights = np.exp(-0.5 * (bin_centers - center)**2 / sigma**2)
        gaussian_weights /= np.sqrt(2 * np.pi * sigma**2)  # Normalize Gaussian
        smoothed_hist[i] = np.sum(hist * gaussian_weights * bin_width)
    
    # Normalize so integral equals 1
    integral = np.trapz(smoothed_hist, bin_centers)
    if integral > 0:
        smoothed_hist /= integral
        
    return smoothed_hist


def numpy_gaussian_smoothing(distances: np.ndarray,
                           bin_centers: np.ndarray,
                           bin_size: np.ndarray,
                           sigma: float) -> np.ndarray:
    """
    Direct Gaussian smoothing approach (closest to original JAX implementation).
    
    Args:
        distances: Array of bond distances
        bin_centers: Centers of histogram bins
        bin_size: Width of each bin
        sigma: Gaussian smoothing parameter
        
    Returns:
        Normalized bond distribution function
    """
    # Create Gaussian contributions for each distance at each bin center
    # distances shape: (n_bonds,)
    # bin_centers shape: (n_bins,)
    # Result shape: (n_bonds, n_bins)
    
    distances_expanded = distances[:, np.newaxis]  # Shape: (n_bonds, 1)
    bin_centers_expanded = bin_centers[np.newaxis, :]  # Shape: (1, n_bins)
    
    # Gaussian exponent: exp(-0.5 * (d - r)^2 / sigma^2)
    gaussian_exp = np.exp(-0.5 * (distances_expanded - bin_centers_expanded)**2 / sigma**2)
    
    # Gaussian probability density: exp * bin_size / sqrt(2*pi*sigma^2)
    gaussian_pdf = gaussian_exp * bin_size / np.sqrt(2 * np.pi * sigma**2)
    
    # Sum over all bonds (axis=0)
    bond_correlation = np.sum(gaussian_pdf, axis=0)
    
    # Normalize using trapezoidal integration
    integral = np.trapz(bond_correlation, bin_centers)
    if integral > 0:
        normalized_distribution = bond_correlation / integral
    else:
        normalized_distribution = bond_correlation
        
    return normalized_distribution


def simple_numpy_histogram_approach(distances: np.ndarray,
                                   bin_boundaries: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple approach using just numpy.histogram without Gaussian smoothing.
    
    Args:
        distances: Array of bond distances
        bin_boundaries: Histogram bin boundaries
        
    Returns:
        bin_centers: Centers of histogram bins
        normalized_hist: Normalized histogram (probability density)
    """
    hist, _ = np.histogram(distances, bins=bin_boundaries, density=True)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    
    return bin_centers, hist


# Example usage and comparison
if __name__ == "__main__":
    # Example data
    np.random.seed(42)
    n_particles = 100
    positions = np.random.rand(n_particles, 3) * 10  # Random positions in 10x10x10 box
    
    # Create some bond topology (e.g., nearest neighbors)
    n_bonds = 50
    bond_topology = np.random.choice(n_particles, size=(n_bonds, 2), replace=False)
    
    # Histogram parameters
    bin_boundaries = np.linspace(0, 5, 51)  # 50 bins from 0 to 5
    sigma = 0.1  # Gaussian smoothing parameter
    
    # Compute bond distribution
    bin_centers, hist_smooth, gaussian_smooth = numpy_bond_distribution_function(
        positions, bond_topology, bin_boundaries, sigma
    )
    
    # Simple histogram for comparison
    bin_centers_simple, hist_simple = simple_numpy_histogram_approach(
        np.linalg.norm(positions[bond_topology[:, 1]] - positions[bond_topology[:, 0]], axis=1),
        bin_boundaries
    )
    
    print("Bond distribution computed successfully!")
    print(f"Histogram approach - sum: {np.sum(hist_smooth):.6f}")
    print(f"Gaussian approach - sum: {np.sum(gaussian_smooth):.6f}")
    print(f"Simple histogram - integral: {np.trapz(hist_simple, bin_centers_simple):.6f}")
    
    # Optional: Plot results if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.plot(bin_centers, hist_smooth, 'b-', label='Histogram + Gaussian')
        plt.xlabel('Distance')
        plt.ylabel('Probability Density')
        plt.title('Gaussian Histogram Method')
        plt.legend()
        
        plt.subplot(1, 3, 2)
        plt.plot(bin_centers, gaussian_smooth, 'r-', label='Direct Gaussian')
        plt.xlabel('Distance')
        plt.ylabel('Probability Density')
        plt.title('Direct Gaussian Method')
        plt.legend()
        
        plt.subplot(1, 3, 3)
        plt.plot(bin_centers_simple, hist_simple, 'g-', label='Simple Histogram')
        plt.xlabel('Distance')
        plt.ylabel('Probability Density')
        plt.title('Simple Histogram')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('bond_distribution_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for plotting")
