#!/usr/bin/env python
"""Create pretrained potential parameters via Boltzmann inversion.

Reads a YAML config specifying target distributions and prior potentials,
performs Boltzmann inversion to get target PMFs, then optimizes tabulated
spline corrections so that (spline + prior) matches the BI target within
the valid distribution region.

Usage:
    python scripts/create_pretrained_params.py config.yaml
    python scripts/create_pretrained_params.py config.yaml --temperature 500 --num-steps 20000 --no-plot
"""

import argparse
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import yaml
from scipy.interpolate import interp1d

# Ensure the package is importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diffcg._core.boltzmann import boltzmann_inversion
from diffcg._core.constants import BOLTZMANN_KJMOLK
from diffcg._core.interpolate import MonotonicInterpolate
from diffcg.energy import generic_repulsion, harmonic_angle, harmonic_dihedral, simple_spring

# ---------------------------------------------------------------------------
# Prior function dispatch
# ---------------------------------------------------------------------------

PRIOR_REGISTRY = {
    "generic_repulsion": lambda x, p: generic_repulsion(x, sigma=p["sigma"], epsilon=p["epsilon"], exp=p["exp"]),
    "harmonic_spring": lambda x, p: simple_spring(x, length=p["length"], epsilon=p["epsilon"]),
    "harmonic_angle": lambda x, p: harmonic_angle(x, angle_0=p["angle_0"], epsilon=p["epsilon"]),
    "harmonic_dihedral": lambda x, p: harmonic_dihedral(x, angle_0=p["angle_0"], epsilon=p["epsilon"]),
}

# ---------------------------------------------------------------------------
# Distribution loading and valid-region detection
# ---------------------------------------------------------------------------


def load_distribution(path, disc, zero_eps):
    """Load a two-column .dist.tgt file and interpolate onto a uniform grid.

    Returns (x_grid, dist_grid) where dist_grid has values < zero_eps set to 0.
    """
    raw = np.loadtxt(path, usecols=(0, 1))
    x_raw, y_raw = raw[:, 0], raw[:, 1]

    x_grid = np.linspace(disc["start"], disc["end"], disc["nbins"])
    f = interp1d(x_raw, y_raw, kind="cubic", bounds_error=False, fill_value=0.0)
    dist_grid = f(x_grid)
    dist_grid = np.clip(dist_grid, 0.0, None)
    dist_grid[dist_grid < zero_eps] = 0.0
    return x_grid, dist_grid


def detect_valid_region(x, dist, edge_quantile=0.001):
    """Find robust valid region using cumulative distribution thresholds.

    Handles noisy edges where isolated small values appear before
    the real distribution onset.
    """
    total = np.sum(dist)
    if total < 1e-12:
        return x, dist, 0, len(x) - 1

    cumsum = np.cumsum(dist) / total
    i_start = int(np.searchsorted(cumsum, edge_quantile))
    i_end = int(np.searchsorted(cumsum, 1.0 - edge_quantile))
    i_end = min(i_end, len(x) - 1)
    return x[i_start : i_end + 1], dist[i_start : i_end + 1], i_start, i_end


# ---------------------------------------------------------------------------
# Per-type processing
# ---------------------------------------------------------------------------


def process_potential(name, cfg, temperature, zero_eps, edge_quantile, config_dir):
    """Process one potential type: load dist, BI, build spline target, masks."""
    kbT = BOLTZMANN_KJMOLK * temperature
    dist_path = os.path.join(config_dir, cfg["distribution"])

    x_grid, dist_grid = load_distribution(dist_path, cfg["discretization"], zero_eps)

    # Detect valid region
    x_valid, dist_valid, i_start, i_end = detect_valid_region(x_grid, dist_grid, edge_quantile)

    # Apply pair r_min filter
    if "r_min" in cfg:
        r_min = cfg["r_min"]
        mask_rmin = x_valid >= r_min
        if np.any(mask_rmin):
            first = int(np.argmax(mask_rmin))
            x_valid = x_valid[first:]
            dist_valid = dist_valid[first:]

    print(f"  {name}: valid region [{x_valid[0]:.4f}, {x_valid[-1]:.4f}] ({len(x_valid)} points)")

    # Boltzmann inversion within valid region
    U_bi = boltzmann_inversion(kbT, dist_valid)

    # Shift
    if name == "pair":
        U_bi = U_bi - U_bi[-1]  # zero at tail
    else:
        U_bi = U_bi - np.min(U_bi)  # zero at minimum

    # Filter inf/nan
    finite_mask = np.isfinite(U_bi)
    x_bi = x_valid[finite_mask]
    U_bi = U_bi[finite_mask]

    if len(x_bi) < 4:
        raise ValueError(f"{name}: fewer than 4 valid BI points — check distribution file")

    # Create scipy spline for evaluating BI target at arbitrary points
    bi_spline = interp1d(x_bi, U_bi, kind="cubic", bounds_error=False, fill_value="extrapolate")

    # Build spline grid
    sg = cfg["spline_grid"]
    spline_grid = np.linspace(sg[0], sg[1], int(sg[2]))
    n_grid = len(spline_grid)

    # Evaluate BI target at spline grid points
    bi_at_grid = bi_spline(spline_grid)

    # Valid-region mask over spline grid
    grid_mask = (spline_grid >= x_bi[0]) & (spline_grid <= x_bi[-1])

    # Build prior function
    prior_cfg = dict(cfg["prior"])
    prior_type = prior_cfg.pop("type")
    prior_fn = PRIOR_REGISTRY[prior_type]
    prior_params = prior_cfg  # remaining keys are prior parameters

    # Evaluate prior at spline grid points
    prior_at_grid = np.array(prior_fn(jnp.array(spline_grid), prior_params))

    # Target for the tabulated correction = BI - prior (within valid region)
    target_at_grid = bi_at_grid - prior_at_grid

    return {
        "name": name,
        "spline_grid": spline_grid,
        "n_grid": n_grid,
        "grid_mask": grid_mask,
        "bi_at_grid": bi_at_grid,
        "prior_at_grid": prior_at_grid,
        "target_at_grid": target_at_grid,
        "x_bi": x_bi,
        "U_bi": U_bi,
        "prior_fn": prior_fn,
        "prior_params": prior_params,
    }


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------


def build_loss_fn(pot_data_list):
    """Build a JAX loss function over all potential types."""
    # Pre-convert to JAX arrays
    jax_data = []
    for pd in pot_data_list:
        jax_data.append(
            {
                "grid": jnp.array(pd["spline_grid"]),
                "mask": jnp.array(pd["grid_mask"], dtype=jnp.float32),
                "bi_target": jnp.array(pd["bi_at_grid"]),
                "prior": jnp.array(pd["prior_at_grid"]),
                "n_grid": pd["n_grid"],
                "name": pd["name"],
            }
        )

    def loss_fn(params):
        total_loss = 0.0
        for i, jd in enumerate(jax_data):
            spline = MonotonicInterpolate(jd["grid"], params[i])
            tabulated = spline(jd["grid"])
            energy = tabulated + jd["prior"]
            diff = jd["mask"] * (energy - jd["bi_target"]) ** 2
            n_active = jnp.maximum(jnp.sum(jd["mask"]), 1.0)
            total_loss = total_loss + jnp.sum(diff) / n_active
        return total_loss

    return loss_fn, jax_data


def run_optimization(pot_data_list, opt_cfg):
    """Run Adam optimization to fit tabulated corrections."""
    loss_fn, jax_data = build_loss_fn(pot_data_list)

    key = jax.random.PRNGKey(opt_cfg.get("seed", 3))
    init_scale = opt_cfg.get("init_scale", 0.001)
    params = []
    for pd in pot_data_list:
        key, subkey = jax.random.split(key)
        p = init_scale * jax.random.normal(subkey, (pd["n_grid"],))
        params.append(p)

    # Optimizer: Adam with exponential decay
    num_steps = opt_cfg.get("num_steps", 10000)
    schedule = optax.exponential_decay(
        init_value=opt_cfg.get("initial_lr", 0.1),
        transition_steps=opt_cfg.get("lr_decay_steps", 100000),
        decay_rate=opt_cfg.get("lr_min_scale", 0.001),
    )
    optimizer = optax.adam(
        learning_rate=schedule,
        b1=opt_cfg.get("adam_b1", 0.9),
        b2=opt_cfg.get("adam_b2", 0.99),
    )
    opt_state = optimizer.init(params)

    loss_and_grad = jax.value_and_grad(loss_fn)

    # JIT the step function
    @jax.jit
    def step(params, opt_state):
        loss, grads = loss_and_grad(params)
        updates, opt_state_new = optimizer.update(grads, opt_state, params)
        params_new = optax.apply_updates(params, updates)
        return params_new, opt_state_new, loss

    losses = []
    print(f"\nOptimizing {num_steps} steps...")
    for i in range(num_steps):
        params, opt_state, loss = step(params, opt_state)
        losses.append(float(loss))
        if i % 1000 == 0 or i == num_steps - 1:
            print(f"  step {i:6d}: loss = {losses[-1]:.6f}")

    return params, losses, jax_data


# ---------------------------------------------------------------------------
# Diagnostics and plotting
# ---------------------------------------------------------------------------


def print_summary(pot_data_list, params, jax_data):
    """Print verification summary table and per-type diagnostics."""
    print("\n=== Pretrained Potential Summary ===")
    print(f"{'Type':<12} {'Valid Region':<18} {'Grid Points':<13} {'Loss (MSE)':<12}")

    for i, pd in enumerate(pot_data_list):
        jd = jax_data[i]
        spline = MonotonicInterpolate(jd["grid"], params[i])
        tabulated = spline(jd["grid"])
        energy = np.array(tabulated + jd["prior"])
        bi = np.array(jd["bi_target"])
        mask = np.array(jd["mask"])
        mse = float(np.sum(mask * (energy - bi) ** 2) / max(np.sum(mask), 1))

        x_bi = pd["x_bi"]
        region_str = f"[{x_bi[0]:.2f}, {x_bi[-1]:.2f}]"
        print(f"{pd['name']:<12} {region_str:<18} {pd['n_grid']:<13} {mse:<12.5f}")

    # Per-type diagnostics
    print()
    for i, pd in enumerate(pot_data_list):
        name = pd["name"]
        jd = jax_data[i]
        grid = np.array(jd["grid"])
        spline = MonotonicInterpolate(jd["grid"], params[i])
        tabulated_vals = np.array(spline(jd["grid"]))
        energy = tabulated_vals + np.array(jd["prior"])
        mask = np.array(jd["mask"]).astype(bool)

        if not np.any(mask):
            continue

        energy_valid = energy[mask]
        grid_valid = grid[mask]

        i_min = int(np.argmin(energy_valid))
        x_min = grid_valid[i_min]
        u_min = energy_valid[i_min]

        if name in ("bond", "angle"):
            well_depth = max(energy_valid[0], energy_valid[-1]) - u_min
            print(f"  {name}: minimum at x={x_min:.4f}, well depth={well_depth:.2f} kJ/mol")
            if i_min == 0 or i_min == len(energy_valid) - 1:
                print(f"    WARNING: minimum at grid boundary — valid region may not cover the well")
        elif name == "pair":
            if i_min < len(energy_valid) - 1:
                print(f"  {name}: minimum at r={x_min:.4f}, U_min={u_min:.2f} kJ/mol")
            else:
                print(f"  {name}: purely repulsive (no minimum found)")
            print(f"  {name}: value at cutoff edge = {energy_valid[-1]:.4f} kJ/mol")
        elif name == "dihedral":
            print(f"  {name}: minimum at x={x_min:.4f}, U_min={u_min:.2f} kJ/mol")

        # Warn on large values
        max_val = np.max(np.abs(energy_valid))
        if max_val > 500:
            print(f"    WARNING: {name} has values up to {max_val:.1f} kJ/mol within valid region")


def make_plots(pot_data_list, params, jax_data, losses, plot_dir):
    """Generate per-type potential plots and loss curve."""
    os.makedirs(plot_dir, exist_ok=True)

    # Loss curve
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(losses)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title("Optimization Loss Curve")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "loss_curve.png"), dpi=150)
    plt.close(fig)

    # Per-type plots
    for i, pd in enumerate(pot_data_list):
        name = pd["name"]
        jd = jax_data[i]
        grid = np.array(jd["grid"])
        mask = np.array(jd["mask"]).astype(bool)
        bi_target = np.array(jd["bi_target"])
        prior = np.array(jd["prior"])

        spline = MonotonicInterpolate(jd["grid"], params[i])
        tabulated_vals = np.array(spline(jd["grid"]))
        total = tabulated_vals + prior

        fig, ax = plt.subplots(figsize=(10, 6))

        # Shaded valid region
        if np.any(mask):
            valid_min = grid[mask][0]
            valid_max = grid[mask][-1]
            ax.axvspan(valid_min, valid_max, alpha=0.1, color="green", label="Valid region")

        # Plot components
        ax.plot(pd["x_bi"], pd["U_bi"], "r.", markersize=3, alpha=0.6, label="BI target (data)")
        ax.plot(grid, total, "k-", linewidth=1.5, label="Fitted total")
        ax.plot(grid, prior, "--", color="gray", linewidth=1, label="Prior")
        ax.plot(grid, tabulated_vals, "-", color="blue", linewidth=1, alpha=0.7, label="Tabulated correction")

        # Minimum marker
        if np.any(mask):
            energy_valid = total[mask]
            grid_valid = grid[mask]
            i_min = int(np.argmin(energy_valid))
            ax.axvline(grid_valid[i_min], color="orange", linestyle=":", alpha=0.7, label=f"Min @ {grid_valid[i_min]:.3f}")

            # Annotate
            u_min = energy_valid[i_min]
            well_depth = max(energy_valid[0], energy_valid[-1]) - u_min
            ax.annotate(
                f"min={grid_valid[i_min]:.3f}\ndepth={well_depth:.1f}",
                xy=(grid_valid[i_min], u_min),
                xytext=(10, 20),
                textcoords="offset points",
                fontsize=8,
                arrowprops=dict(arrowstyle="->", color="orange"),
            )

        ax.set_xlabel("x")
        ax.set_ylabel("U (kJ/mol)")
        ax.set_title(f"{name} potential")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f"{name}.png"), dpi=150)
        plt.close(fig)

    print(f"\nPlots saved to {plot_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Create pretrained potential parameters via Boltzmann inversion")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    parser.add_argument("--temperature", type=float, default=None, help="Override temperature (K)")
    parser.add_argument("--num-steps", type=int, default=None, help="Override optimization steps")
    parser.add_argument("--no-plot", action="store_true", help="Skip plot generation")
    args = parser.parse_args()

    # Load config
    config_path = os.path.abspath(args.config)
    config_dir = os.path.dirname(config_path)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Apply CLI overrides
    temperature = args.temperature if args.temperature is not None else config["temperature"]
    zero_eps = config.get("zero_eps", 1e-7)
    edge_quantile = config.get("edge_quantile", 0.001)

    opt_cfg = config.get("optimization", {})
    if args.num_steps is not None:
        opt_cfg["num_steps"] = args.num_steps

    out_cfg = config.get("output", {})
    params_file = os.path.join(config_dir, out_cfg.get("params_file", "pretrained_params.npy"))
    plot_dir = os.path.join(config_dir, out_cfg.get("plot_dir", "pretrain_plots"))

    # Process each potential type
    potentials = config.get("potentials", {})
    if not potentials:
        print("No potentials specified in config. Nothing to do.")
        return

    print(f"Temperature: {temperature} K, kbT = {BOLTZMANN_KJMOLK * temperature:.4f} kJ/mol")
    print(f"Processing {len(potentials)} potential type(s): {list(potentials.keys())}")

    pot_data_list = []
    for name in potentials:
        pot_data_list.append(
            process_potential(name, potentials[name], temperature, zero_eps, edge_quantile, config_dir)
        )

    # Optimize
    params, losses, jax_data = run_optimization(pot_data_list, opt_cfg)

    # Summary
    print_summary(pot_data_list, params, jax_data)

    # Save
    save_dict = {}
    for i, pd in enumerate(pot_data_list):
        save_dict[pd["name"]] = np.array(params[i])
    np.save(params_file, save_dict)
    print(f"\nParameters saved to {params_file}")

    # Plot
    if not args.no_plot:
        make_plots(pot_data_list, params, jax_data, losses, plot_dir)


if __name__ == "__main__":
    main()
