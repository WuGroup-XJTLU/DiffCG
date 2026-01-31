# SPDX-License-Identifier: MIT
# Copyright (c) 2025 WuResearchGroup

"""
Visualization utilities for DiffCG training iterations.

Provides functions to create output folders, plot observable comparisons,
and save loss curves during optimization.
"""

import os
import numpy as np
from diffcg._core.logger import get_logger

logger = get_logger(__name__)

# Try to import matplotlib, but handle gracefully if not installed
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not installed. Figure generation will be disabled.")


# Observable metadata for plotting
OBSERVABLE_METADATA = {
    'rdf': {
        'title': 'Radial Distribution Function',
        'xlabel': 'r (nm)',
        'ylabel': 'g(r)',
    },
    'bdf': {
        'title': 'Bond Distribution Function',
        'xlabel': 'Bond Length (nm)',
        'ylabel': 'P(r)',
    },
    'adf': {
        'title': 'Angle Distribution Function',
        'xlabel': 'Angle (rad)',
        'ylabel': 'P(θ)',
    },
    'ddf': {
        'title': 'Dihedral Distribution Function',
        'xlabel': 'Dihedral Angle (rad)',
        'ylabel': 'P(φ)',
    },
}


def create_iteration_folder(output_dir, step):
    """
    Create output folder for a training iteration.

    Args:
        output_dir: Base output directory.
        step: Iteration step number.

    Returns:
        Path to the created iteration folder.
    """
    iteration_folder = os.path.join(output_dir, f"iteration_{step}")
    os.makedirs(iteration_folder, exist_ok=True)
    return iteration_folder


def plot_observable_comparison(predicted, target, bin_centers, observable_name, save_path):
    """
    Plot predicted vs target observable comparison.

    Args:
        predicted: Predicted observable values (array).
        target: Target observable values (array).
        bin_centers: Bin centers for x-axis (array).
        observable_name: Name of the observable (e.g., 'rdf', 'bdf').
        save_path: Path to save the figure.
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning(f"Cannot plot {observable_name}: matplotlib not installed.")
        return

    # Convert JAX arrays to numpy if needed
    predicted = np.asarray(predicted)
    target = np.asarray(target)
    bin_centers = np.asarray(bin_centers)

    metadata = OBSERVABLE_METADATA.get(observable_name, {
        'title': observable_name.upper(),
        'xlabel': 'x',
        'ylabel': 'y',
    })

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(bin_centers, target, 'k-', linewidth=2, label='Target')
    ax.plot(bin_centers, predicted, 'r--', linewidth=2, label='Predicted')
    ax.set_xlabel(metadata['xlabel'], fontsize=12)
    ax.set_ylabel(metadata['ylabel'], fontsize=12)
    ax.set_title(metadata['title'], fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.debug(f"Saved observable comparison plot: {save_path}")


def plot_loss_curve(loss_history, save_path):
    """
    Plot training loss curve.

    Args:
        loss_history: List of loss values per iteration.
        save_path: Path to save the figure.
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Cannot plot loss curve: matplotlib not installed.")
        return

    # Convert to numpy array
    loss_values = np.asarray([float(l) for l in loss_history])
    steps = np.arange(len(loss_values))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(steps, loss_values, 'b-', linewidth=2, marker='o', markersize=4)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Loss Curve', fontsize=14)
    ax.grid(True, alpha=0.3)

    # Use log scale if loss range is large
    if len(loss_values) > 1:
        loss_range = np.max(loss_values) / (np.min(loss_values) + 1e-10)
        if loss_range > 100:
            ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.debug(f"Saved loss curve plot: {save_path}")


def plot_per_state_loss_curve(per_state_loss_history, save_path):
    """
    Plot per-state loss curves for multistate optimization.

    Args:
        per_state_loss_history: List of dicts mapping state_id -> loss per iteration.
        save_path: Path to save the figure.
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Cannot plot per-state loss curve: matplotlib not installed.")
        return

    if not per_state_loss_history:
        return

    # Extract state IDs from first entry
    state_ids = list(per_state_loss_history[0].keys())
    steps = np.arange(len(per_state_loss_history))

    fig, ax = plt.subplots(figsize=(10, 6))

    for sid in state_ids:
        losses = [float(entry[sid]) for entry in per_state_loss_history]
        ax.plot(steps, losses, linewidth=2, marker='o', markersize=4, label=f'State {sid}')

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Per-State Loss Curves', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Use log scale if any state has large loss range
    all_losses = []
    for entry in per_state_loss_history:
        all_losses.extend([float(v) for v in entry.values()])
    if len(all_losses) > 1:
        loss_range = np.max(all_losses) / (np.min(all_losses) + 1e-10)
        if loss_range > 100:
            ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.debug(f"Saved per-state loss curve plot: {save_path}")


def save_observable_data(predicted, target, bin_centers, observable_name, save_dir):
    """
    Save observable data to CSV and NPY files.

    Args:
        predicted: Predicted observable values (array).
        target: Target observable values (array).
        bin_centers: Bin centers (array).
        observable_name: Name of the observable.
        save_dir: Directory to save files.
    """
    # Convert to numpy
    predicted = np.asarray(predicted)
    target = np.asarray(target)
    bin_centers = np.asarray(bin_centers)

    # Save as CSV (human-readable)
    csv_path = os.path.join(save_dir, f"{observable_name}_data.csv")
    header = "bin_centers,predicted,target"
    data = np.column_stack([bin_centers, predicted, target])
    np.savetxt(csv_path, data, delimiter=',', header=header, comments='')
    logger.debug(f"Saved observable data CSV: {csv_path}")

    # Save predicted as NPY (for further analysis)
    npy_path = os.path.join(save_dir, f"{observable_name}_predicted.npy")
    np.save(npy_path, predicted)
    logger.debug(f"Saved predicted array: {npy_path}")


def save_loss_history(loss_history, save_path):
    """
    Save loss history to CSV file.

    Args:
        loss_history: List of loss values.
        save_path: Path to save CSV file.
    """
    loss_values = np.asarray([float(l) for l in loss_history])
    steps = np.arange(len(loss_values))
    data = np.column_stack([steps, loss_values])
    header = "iteration,loss"
    np.savetxt(save_path, data, delimiter=',', header=header, comments='')
    logger.debug(f"Saved loss history: {save_path}")


def save_per_state_loss_history(per_state_loss_history, save_path):
    """
    Save per-state loss history to CSV file.

    Args:
        per_state_loss_history: List of dicts mapping state_id -> loss.
        save_path: Path to save CSV file.
    """
    if not per_state_loss_history:
        return

    state_ids = list(per_state_loss_history[0].keys())
    steps = np.arange(len(per_state_loss_history))

    # Build data array
    data = [steps]
    for sid in state_ids:
        losses = [float(entry[sid]) for entry in per_state_loss_history]
        data.append(losses)
    data = np.column_stack(data)

    header = "iteration," + ",".join([f"state_{sid}" for sid in state_ids])
    np.savetxt(save_path, data, delimiter=',', header=header, comments='')
    logger.debug(f"Saved per-state loss history: {save_path}")


def save_iteration_figures(step, predictions, quantity_dict, loss_history, output_dir):
    """
    Save all figures and data for a single-state iteration.

    Args:
        step: Current iteration step.
        predictions: Dict mapping observable_name -> predicted values.
        quantity_dict: Dict with observable info including 'target' and 'bin_centers'.
        loss_history: List of loss values up to current step.
        output_dir: Base output directory.
    """
    iteration_folder = create_iteration_folder(output_dir, step)

    # Save observable comparisons
    for obs_name, pred_values in predictions.items():
        if obs_name not in quantity_dict:
            continue

        obs_info = quantity_dict[obs_name]
        target = obs_info.get('target')
        bin_centers = obs_info.get('bin_centers')

        if target is None or bin_centers is None:
            logger.warning(f"Missing target or bin_centers for {obs_name}, skipping.")
            continue

        # Plot comparison
        plot_path = os.path.join(iteration_folder, f"{obs_name}_comparison.png")
        plot_observable_comparison(pred_values, target, bin_centers, obs_name, plot_path)

        # Save data
        save_observable_data(pred_values, target, bin_centers, obs_name, iteration_folder)

    # Save loss curve
    if loss_history:
        loss_plot_path = os.path.join(iteration_folder, "loss_curve.png")
        plot_loss_curve(loss_history, loss_plot_path)

        loss_csv_path = os.path.join(iteration_folder, "loss_history.csv")
        save_loss_history(loss_history, loss_csv_path)

    logger.info(f"Saved iteration {step} figures to {iteration_folder}")


def save_multistate_iteration_figures(step, predictions_by_state, states, quantity_dicts,
                                       loss_history, per_state_loss_history, output_dir):
    """
    Save all figures and data for a multistate iteration.

    Args:
        step: Current iteration step.
        predictions_by_state: Dict mapping state_id -> {observable_name -> predicted values}.
        states: Dict mapping state_id -> state config (for reference).
        quantity_dicts: Dict mapping state_id -> quantity_dict with 'target' and 'bin_centers'.
        loss_history: List of total loss values up to current step.
        per_state_loss_history: List of dicts mapping state_id -> loss.
        output_dir: Base output directory.
    """
    iteration_folder = create_iteration_folder(output_dir, step)

    # Save per-state observable comparisons
    for sid, predictions in predictions_by_state.items():
        state_folder = os.path.join(iteration_folder, f"state_{sid}")
        os.makedirs(state_folder, exist_ok=True)

        quantity_dict = quantity_dicts.get(sid, {})

        for obs_name, pred_values in predictions.items():
            if obs_name not in quantity_dict:
                continue

            obs_info = quantity_dict[obs_name]
            target = obs_info.get('target')
            bin_centers = obs_info.get('bin_centers')

            if target is None or bin_centers is None:
                logger.warning(f"Missing target or bin_centers for state {sid} {obs_name}, skipping.")
                continue

            # Plot comparison
            plot_path = os.path.join(state_folder, f"{obs_name}_comparison.png")
            plot_observable_comparison(pred_values, target, bin_centers, obs_name, plot_path)

            # Save data
            save_observable_data(pred_values, target, bin_centers, obs_name, state_folder)

    # Save total loss curve
    if loss_history:
        loss_plot_path = os.path.join(iteration_folder, "loss_curve.png")
        plot_loss_curve(loss_history, loss_plot_path)

        loss_csv_path = os.path.join(iteration_folder, "loss_history.csv")
        save_loss_history(loss_history, loss_csv_path)

    # Save per-state loss curves
    if per_state_loss_history:
        per_state_plot_path = os.path.join(iteration_folder, "per_state_loss_curve.png")
        plot_per_state_loss_curve(per_state_loss_history, per_state_plot_path)

        per_state_csv_path = os.path.join(iteration_folder, "per_state_loss_history.csv")
        save_per_state_loss_history(per_state_loss_history, per_state_csv_path)

    logger.info(f"Saved multistate iteration {step} figures to {iteration_folder}")


def plot_potentials(potentials_data, save_path):
    """Plot tabulated, prior, and total potentials for bond/angle/dihedral/pair.

    Args:
        potentials_data: list of dicts, each with keys:
            - 'name': str (e.g. 'bond')
            - 'x': array of x-grid values
            - 'tabulated': array or list of arrays of tabulated potential values
            - 'prior': array or list of arrays of prior potential values
            - 'xlabel': str
            - 'ylabel': str
            - 'labels': optional list of str labels (for multi-type, e.g. pair types)
        save_path: path to save the figure
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Cannot plot potentials: matplotlib not installed.")
        return

    n = len(potentials_data)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, data in zip(axes, potentials_data):
        x = np.asarray(data['x'])
        tabulated = data['tabulated']
        prior = data['prior']
        labels = data.get('labels', None)

        # Handle multi-type (list of arrays) vs single array
        if isinstance(tabulated, list):
            for i, (tab, pri) in enumerate(zip(tabulated, prior)):
                tab = np.asarray(tab)
                pri = np.asarray(pri)
                total = tab + pri
                lbl = labels[i] if labels else f'type {i}'
                ax.plot(x, tab, '--', alpha=0.6, label=f'{lbl} tab')
                ax.plot(x, pri, ':', alpha=0.6, label=f'{lbl} prior')
                ax.plot(x, total, '-', linewidth=1.5, label=f'{lbl} total')
        else:
            tab = np.asarray(tabulated)
            pri = np.asarray(prior)
            total = tab + pri
            ax.plot(x, tab, 'b--', linewidth=1.5, label='Tabulated')
            ax.plot(x, pri, 'g:', linewidth=1.5, label='Prior')
            ax.plot(x, total, 'r-', linewidth=1.5, label='Total')

        ax.set_xlabel(data.get('xlabel', 'x'), fontsize=11)
        ax.set_ylabel(data.get('ylabel', 'Energy (kJ/mol)'), fontsize=11)
        ax.set_title(data['name'], fontsize=13)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.debug(f"Saved potentials plot: {save_path}")


def save_potentials_data(potentials_data, save_dir):
    """Save tabulated, prior, and total potential data as CSV files.

    Args:
        potentials_data: list of dicts (same format as plot_potentials),
            each with keys 'name', 'x', 'tabulated', 'prior', and
            optionally 'labels' for multi-type potentials (e.g. pair).
        save_dir: directory to save CSV files into.
    """
    os.makedirs(save_dir, exist_ok=True)

    for data in potentials_data:
        name = data['name'].lower()
        x = np.asarray(data['x'])
        tabulated = data['tabulated']
        prior = data['prior']

        if isinstance(tabulated, list):
            # Multi-type (e.g. pair): one CSV per sub-type
            for i, (tab, pri) in enumerate(zip(tabulated, prior)):
                tab = np.asarray(tab)
                pri = np.asarray(pri)
                total = tab + pri
                csv_data = np.column_stack([x, tab, pri, total])
                csv_path = os.path.join(save_dir, f"{name}_potential_{i}.csv")
                np.savetxt(csv_path, csv_data, delimiter=',',
                           header='x,tabulated,prior,total', comments='')
                logger.debug(f"Saved potential data: {csv_path}")
        else:
            tab = np.asarray(tabulated)
            pri = np.asarray(prior)
            total = tab + pri
            csv_data = np.column_stack([x, tab, pri, total])
            csv_path = os.path.join(save_dir, f"{name}_potential.csv")
            np.savetxt(csv_path, csv_data, delimiter=',',
                       header='x,tabulated,prior,total', comments='')
            logger.debug(f"Saved potential data: {csv_path}")
