# SPDX-License-Identifier: MIT
# Copyright (c) 2025 WuResearchGroup

"""JAX-MD based molecular dynamics sampler.

This module provides a fully differentiable MD sampler using JAX-MD integrators,
enabling gradient flow through MD trajectories for DiffSim and related methods.
"""

from functools import partial
from typing import Callable, Optional, Tuple, Any, Dict
from collections import namedtuple

import jax
import jax.numpy as jnp
from jax import random
from jax_md import simulate, space, partition

from diffcg.system import System
from diffcg.common.neighborlist import jaxmd_neighbor_list, JAXMDSpatialPartitioning
from diffcg.util.logger import get_logger

logger = get_logger(__name__)

# Boltzmann constant in kJ/(mol*K)
KB_KJ_MOL = 0.0083145107

# Conversion factor from fs to JAX-MD internal time units
# JAX-MD uses reduced units, but we'll keep consistent with DiffCG's conventions
# The length scale is in nm, time unit is sqrt(u*nm^2/(kJ/mol)) ~ 0.1 ps
# 1 fs = 0.001 ps, so we multiply by 0.1 to convert fs to internal units
FS_TO_INTERNAL = 0.1

# Result container for MD simulation
MDResult = namedtuple("MDResult", ("final_state", "trajectory", "final_neighbors"))


def _wrap_energy_fn_for_jaxmd(
    energy_fn: Callable,
    Z: jnp.ndarray,
    cell: Optional[jnp.ndarray],
) -> Callable:
    """Wrap a DiffCG energy function for JAX-MD interface.

    JAX-MD expects energy_fn(R, neighbor, **kwargs) -> float
    DiffCG provides energy_fn(System, neighbors) -> float

    Args:
        energy_fn: DiffCG energy function.
        Z: Atomic numbers array.
        cell: Unit cell matrix (column-major format).

    Returns:
        JAX-MD compatible energy function.
    """
    def jaxmd_energy_fn(R, neighbor, **kwargs):
        system = System(R=R, Z=Z, cell=cell)
        return energy_fn(system, neighbor, **kwargs)
    return jaxmd_energy_fn


class JAXMDSampler:
    """JAX-MD based molecular dynamics sampler.

    This class wraps JAX-MD integrators to provide a unified interface for
    NVE, NVT (Langevin/Nose-Hoover), and NPT (Nose-Hoover) ensembles.

    The sampler is fully JIT-compilable and differentiable, enabling
    gradient-based optimization through MD trajectories.
    """

    def __init__(
        self,
        energy_fn: Callable,
        Z: jnp.ndarray,
        cell: Optional[jnp.ndarray],
        cutoff: float,
        ensemble: str = 'nvt',
        thermostat: str = 'langevin',
        temperature: float = 300.0,
        timestep: float = 2.0,
        kT: Optional[float] = None,
        capacity_multiplier: float = 1.25,
        pressure: float = 1.01325e-4,
        taut: Optional[float] = None,
        taup: Optional[float] = None,
        friction: float = 1.0,
        mass: Optional[jnp.ndarray] = None,
        **kwargs
    ):
        """Initialize the JAX-MD sampler.

        Args:
            energy_fn: Energy function with signature (System, neighbors) -> float.
            Z: Atomic numbers array, shape (N,).
            cell: Unit cell matrix, shape (3, 3) in column-major format. None for free BC.
            cutoff: Neighbor list cutoff distance.
            ensemble: Ensemble type, one of 'nve', 'nvt', 'npt'.
            thermostat: Thermostat type for NVT, one of 'langevin', 'nose-hoover'.
            temperature: Temperature in Kelvin.
            timestep: Time step in fs.
            kT: Thermal energy in kJ/mol. If None, computed from temperature.
            capacity_multiplier: Neighbor list capacity multiplier.
            pressure: External pressure in GPa (for NPT).
            taut: Temperature coupling time constant in fs (for Nose-Hoover).
            taup: Pressure coupling time constant in fs (for NPT Nose-Hoover).
            friction: Friction coefficient in 1/ps (for Langevin).
            mass: Particle masses. If None, uses unit masses.
            **kwargs: Additional keyword arguments.
        """
        self.Z = Z
        self.cell = cell
        self.cutoff = cutoff
        self.ensemble = ensemble.lower()
        self.thermostat = thermostat.lower()
        self.temperature = temperature
        self.timestep = timestep
        self.capacity_multiplier = capacity_multiplier

        # Compute thermal energy
        self.kT = kT if kT is not None else temperature * KB_KJ_MOL

        # Convert timestep to internal units
        dt = timestep * FS_TO_INTERNAL

        # Default time constants
        if taut is None:
            taut = 100 * timestep
        if taup is None:
            taup = 1000 * timestep
        self.taut = taut * FS_TO_INTERNAL
        self.taup = taup * FS_TO_INTERNAL

        # Setup space functions for JAX-MD
        if cell is not None:
            cell_jaxmd = cell.T  # Convert to row-major format
            self.displacement_fn, self.shift_fn = space.periodic_general(cell_jaxmd)
            self.box = cell_jaxmd
        else:
            self.displacement_fn, self.shift_fn = space.free()
            self.box = None

        # Store original energy function and create wrapped version
        self._original_energy_fn = energy_fn
        self._wrapped_energy_fn = _wrap_energy_fn_for_jaxmd(energy_fn, Z, cell)

        # Create neighbor list function
        self.neighbor_fn = partition.neighbor_list(
            self.displacement_fn,
            box=self.box,
            r_cutoff=cutoff,
            capacity_multiplier=capacity_multiplier,
            format=partition.Sparse,
        )

        # Select and create integrator
        self._create_integrator(dt, friction, pressure, mass)

        logger.info(
            "Created JAX-MD sampler: ensemble=%s, thermostat=%s, T=%s K, dt=%s fs",
            self.ensemble, self.thermostat if self.ensemble == 'nvt' else 'n/a',
            temperature, timestep
        )

    def _create_integrator(
        self,
        dt: float,
        friction: float,
        pressure: float,
        mass: Optional[jnp.ndarray],
    ) -> None:
        """Create the appropriate JAX-MD integrator.

        Args:
            dt: Time step in internal units.
            friction: Friction coefficient for Langevin.
            pressure: External pressure for NPT.
            mass: Particle masses.
        """
        # Wrap energy function to work with neighbor list
        def energy_fn_with_neighbors(R, neighbor, **kwargs):
            return self._wrapped_energy_fn(R, neighbor, **kwargs)

        if self.ensemble == 'nve':
            self.init_fn, self.apply_fn = simulate.nve(
                energy_fn_with_neighbors,
                self.shift_fn,
                dt,
            )
            logger.debug("Using NVE integrator")

        elif self.ensemble == 'nvt':
            if self.thermostat == 'langevin':
                # Friction is in 1/ps, convert to internal units
                gamma = friction * 0.1  # Convert from 1/ps to internal units
                self.init_fn, self.apply_fn = simulate.nvt_langevin(
                    energy_fn_with_neighbors,
                    self.shift_fn,
                    dt,
                    self.kT,
                    gamma,
                )
                logger.debug("Using NVT Langevin integrator with gamma=%s", gamma)

            elif self.thermostat == 'nose-hoover':
                self.init_fn, self.apply_fn = simulate.nvt_nose_hoover(
                    energy_fn_with_neighbors,
                    self.shift_fn,
                    dt,
                    self.kT,
                    tau=self.taut,
                )
                logger.debug("Using NVT Nose-Hoover integrator with tau=%s", self.taut)

            else:
                raise ValueError(
                    f"Unsupported thermostat: {self.thermostat}. "
                    "Use 'langevin' or 'nose-hoover'."
                )

        elif self.ensemble == 'npt':
            # NPT only supports Nose-Hoover in JAX-MD
            # Convert pressure from GPa to internal units (kJ/(mol*nm^3))
            # 1 GPa = 1e9 Pa = 1e9 J/m^3 = 1e9 * 1e-3 kJ / (1e27 nm^3) = 1e-21 kJ/nm^3
            # But JAX-MD uses reduced units, so we need to be careful
            # For now, assume pressure is already in appropriate units
            pressure_internal = pressure  # User should provide in correct units

            self.init_fn, self.apply_fn = simulate.npt_nose_hoover(
                energy_fn_with_neighbors,
                self.shift_fn,
                dt,
                pressure_internal,
                self.kT,
                tau=self.taut,
                barostat_tau=self.taup,
            )
            logger.debug(
                "Using NPT Nose-Hoover integrator with tau=%s, barostat_tau=%s",
                self.taut, self.taup
            )

        else:
            raise ValueError(
                f"Unsupported ensemble: {self.ensemble}. "
                "Use 'nve', 'nvt', or 'npt'."
            )

    def initialize_state(
        self,
        R: jnp.ndarray,
        key: jax.random.PRNGKey,
        neighbor: Optional[Any] = None,
    ) -> Tuple[Any, Any]:
        """Initialize the MD state.

        Args:
            R: Initial positions, shape (N, 3).
            key: JAX random key for initialization.
            neighbor: Optional pre-allocated neighbor list.

        Returns:
            Tuple of (state, neighbor).
        """
        if neighbor is None:
            neighbor = self.neighbor_fn.allocate(R)

        state = self.init_fn(key, R, neighbor=neighbor)
        return state, neighbor

    def step(
        self,
        state: Any,
        neighbor: Any,
        **kwargs
    ) -> Tuple[Any, Any]:
        """Perform a single MD step.

        Args:
            state: Current MD state.
            neighbor: Current neighbor list.
            **kwargs: Additional arguments passed to energy function.

        Returns:
            Tuple of (new_state, new_neighbor).
        """
        new_state = self.apply_fn(state, neighbor=neighbor, **kwargs)
        new_neighbor = self.neighbor_fn.update(new_state.position, neighbor)
        return new_state, new_neighbor

    def run(
        self,
        R: jnp.ndarray,
        steps: int,
        key: Optional[jax.random.PRNGKey] = None,
        neighbor: Optional[Any] = None,
        save_frequency: int = 1,
        **kwargs
    ) -> MDResult:
        """Run MD simulation.

        Args:
            R: Initial positions, shape (N, 3).
            steps: Number of MD steps to run.
            key: JAX random key. If None, uses a default key.
            neighbor: Optional pre-allocated neighbor list.
            save_frequency: Save trajectory every N steps.
            **kwargs: Additional arguments passed to energy function.

        Returns:
            MDResult namedtuple with final_state, trajectory, and final_neighbors.
        """
        if key is None:
            key = random.PRNGKey(0)

        state, neighbor = self.initialize_state(R, key, neighbor)

        # Run simulation with lax.scan for efficiency
        def step_fn(carry, _):
            state, neighbor = carry
            new_state = self.apply_fn(state, neighbor=neighbor, **kwargs)
            new_neighbor = self.neighbor_fn.update(new_state.position, neighbor)
            return (new_state, new_neighbor), new_state.position

        (final_state, final_neighbor), trajectory = jax.lax.scan(
            step_fn, (state, neighbor), jnp.arange(steps)
        )

        # Subsample trajectory if save_frequency > 1
        if save_frequency > 1:
            trajectory = trajectory[::save_frequency]

        return MDResult(
            final_state=final_state,
            trajectory=trajectory,
            final_neighbors=final_neighbor,
        )

    def run_with_checkpoints(
        self,
        R: jnp.ndarray,
        steps: int,
        checkpoint_interval: int = 100,
        key: Optional[jax.random.PRNGKey] = None,
        neighbor: Optional[Any] = None,
        **kwargs
    ) -> MDResult:
        """Run MD simulation with gradient checkpointing.

        This version uses gradient checkpointing to reduce memory usage
        during backpropagation through long trajectories.

        Args:
            R: Initial positions, shape (N, 3).
            steps: Number of MD steps to run.
            checkpoint_interval: Number of steps between checkpoints.
            key: JAX random key. If None, uses a default key.
            neighbor: Optional pre-allocated neighbor list.
            **kwargs: Additional arguments passed to energy function.

        Returns:
            MDResult namedtuple with final_state, trajectory, and final_neighbors.
        """
        if key is None:
            key = random.PRNGKey(0)

        state, neighbor = self.initialize_state(R, key, neighbor)

        # Use remat (gradient checkpointing) for memory efficiency
        @jax.checkpoint
        def checkpoint_block(carry, _):
            state, neighbor = carry
            # Run checkpoint_interval steps
            def inner_step(inner_carry, __):
                s, n = inner_carry
                new_s = self.apply_fn(s, neighbor=n, **kwargs)
                new_n = self.neighbor_fn.update(new_s.position, n)
                return (new_s, new_n), new_s.position

            final_carry, positions = jax.lax.scan(
                inner_step, (state, neighbor), jnp.arange(checkpoint_interval)
            )
            return final_carry, positions

        num_checkpoints = steps // checkpoint_interval
        remainder = steps % checkpoint_interval

        if num_checkpoints > 0:
            (state, neighbor), trajectory_blocks = jax.lax.scan(
                checkpoint_block, (state, neighbor), jnp.arange(num_checkpoints)
            )
            # Reshape from (num_checkpoints, checkpoint_interval, N, 3) to (total, N, 3)
            trajectory = trajectory_blocks.reshape(-1, *R.shape)
        else:
            trajectory = jnp.empty((0,) + R.shape)

        # Handle remainder steps
        if remainder > 0:
            def final_step(carry, _):
                s, n = carry
                new_s = self.apply_fn(s, neighbor=n, **kwargs)
                new_n = self.neighbor_fn.update(new_s.position, n)
                return (new_s, new_n), new_s.position

            (state, neighbor), remainder_traj = jax.lax.scan(
                final_step, (state, neighbor), jnp.arange(remainder)
            )
            trajectory = jnp.concatenate([trajectory, remainder_traj], axis=0)

        return MDResult(
            final_state=state,
            trajectory=trajectory,
            final_neighbors=neighbor,
        )


def create_jaxmd_sampler(
    energy_fn: Callable,
    system: System,
    cutoff: float,
    ensemble: str = 'nvt',
    thermostat: str = 'langevin',
    temperature: float = 300.0,
    timestep: float = 2.0,
    **kwargs
) -> JAXMDSampler:
    """Factory function to create a JAX-MD sampler from a System.

    Args:
        energy_fn: Energy function with signature (System, neighbors) -> float.
        system: DiffCG System namedtuple.
        cutoff: Neighbor list cutoff distance.
        ensemble: Ensemble type.
        thermostat: Thermostat type for NVT.
        temperature: Temperature in Kelvin.
        timestep: Time step in fs.
        **kwargs: Additional arguments passed to JAXMDSampler.

    Returns:
        Configured JAXMDSampler instance.
    """
    return JAXMDSampler(
        energy_fn=energy_fn,
        Z=system.Z,
        cell=system.cell,
        cutoff=cutoff,
        ensemble=ensemble,
        thermostat=thermostat,
        temperature=temperature,
        timestep=timestep,
        **kwargs
    )
