#!/usr/bin/env python
"""Generate O-O RDF from LAMMPS trajectory file for coarse-grained water model.

This script reads a LAMMPS dump file containing all-atom SPC/E water trajectory,
extracts oxygen atoms, and computes the oxygen-oxygen radial distribution function.
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from diffcg.io.lammps import read_lammps_dump
from diffcg.observable.structure import (
    initialize_inter_radial_distribution_fun,
    InterRDFParams,
    rdf_discretization,
)
from diffcg.system import AtomicSystem


def main():
    # Parameters for RDF computation
    R_CUT = 1.0  # nm - RDF cutoff
    RDF_START = 0.0  # nm - RDF start
    NBINS = 300  # Number of RDF bins
    OXYGEN_TYPE = 2  # Type 2 in LAMMPS dump file

    # Files
    dump_file = "water.dump"
    output_file = "water_oo_rdf.dat"

    print(f"Reading LAMMPS dump file: {dump_file}")

    # Read LAMMPS dump file
    dump_data = read_lammps_dump(dump_file)

    # Extract data
    all_coords = dump_data["coords"]  # (n_frames, n_atoms, 3) in Angstroms
    all_types = dump_data["atom_types"]  # (n_atoms,)
    all_cells = dump_data["cells"]  # (n_frames, 3, 3) in Angstroms

    # Convert coordinates from Angstroms to nanometers
    all_coords = all_coords / 10.0
    all_cells = all_cells / 10.0

    # Find oxygen atoms (type 2 in original LAMMPS, becomes type 1 after 0-indexing)
    # In the dump file, type 2 = oxygen, type 1 = hydrogen
    # After read_lammps_dump converts to 0-indexed: type 1 = oxygen, type 0 = hydrogen
    oxygen_mask = all_types == 1  # 0-indexed oxygen type

    print(f"Total atoms: {all_types.shape[0]}")
    print(f"Oxygen atoms: {oxygen_mask.sum()}")

    # Extract oxygen coordinates for all frames
    oxygen_coords = all_coords[:, oxygen_mask, :]  # (n_frames, n_oxygen, 3)

    # Get number of oxygen atoms (should be 300)
    n_oxygen = oxygen_mask.sum()
    print(f"Number of oxygen atoms: {n_oxygen}")

    # Setup RDF discretization
    rdf_bin_centers, rdf_bin_boundaries, sigma_RDF = rdf_discretization(
        RDF_cut=R_CUT, nbins=NBINS, RDF_start=RDF_START
    )

    # No exclusions needed for single-site water - all O-O pairs contribute
    exclude_pairs = jnp.empty((0, 2), dtype=jnp.int32)

    # Initialize RDF parameters
    rdf_params = InterRDFParams(
        reference_rdf=None,  # No target, computing from trajectory
        rdf_bin_centers=rdf_bin_centers,
        rdf_bin_boundaries=rdf_bin_boundaries,
        sigma_RDF=sigma_RDF,
        exclude_pairs=exclude_pairs,
    )

    # Create RDF compute function
    rdf_compute_fn = initialize_inter_radial_distribution_fun(rdf_params)

    # Compute RDF for each frame and average
    print(f"Computing O-O RDF from {len(oxygen_coords)} frames...")

    rdf_sum = None
    for i, coords in enumerate(oxygen_coords):
        # Create AtomicSystem for this frame
        cell = all_cells[i]
        system = AtomicSystem(
            R=jnp.array(coords, dtype=jnp.float32),
            Z=jnp.zeros(n_oxygen, dtype=jnp.int32),  # All same type for water
            cell=jnp.array(cell, dtype=jnp.float32),
            masses=jnp.array([15.9994] * n_oxygen, dtype=jnp.float32),
            pbc=True,
        )

        # Compute RDF for this frame
        frame_rdf = rdf_compute_fn(system)

        # Accumulate
        if rdf_sum is None:
            rdf_sum = frame_rdf
        else:
            rdf_sum = rdf_sum + frame_rdf

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1} frames...")

    # Average over all frames
    avg_rdf = rdf_sum / len(oxygen_coords)

    # Save to file (two-column format: distance, g(r))
    print(f"Saving O-O RDF to: {output_file}")
    np.savetxt(output_file, np.column_stack([rdf_bin_centers, avg_rdf]),
               header=f"distance(nm) g(r)  # {len(oxygen_coords)} frames averaged")

    print(f"RDF computation complete!")
    print(f"  - Peak g(r): {jnp.max(avg_rdf):.3f} at {rdf_bin_centers[jnp.argmax(avg_rdf)]:.3f} nm")
    print(f"  - First peak should be around 0.28 nm for water")


if __name__ == "__main__":
    main()
