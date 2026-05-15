"""Read/write GPUMD NEP potential files (nep.txt)."""

import jax.numpy as jnp
import numpy as np

_ELEMENTS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu",
]


def _element_index(symbol: str) -> int:
    return _ELEMENTS.index(symbol)


def read_nep(filepath: str) -> dict:
    """Parse a GPUMD nep.txt file into a dict of parameters."""
    with open(filepath, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    # Line 1: nep<version> <num_types> <elem1> ...
    tokens = lines[0].split()
    version_str = tokens[0]
    if version_str.startswith("nep"):
        version = int(version_str[3:])
    else:
        raise ValueError(f"Unknown NEP version: {version_str}")
    num_types = int(tokens[1])
    elements = tokens[2:2 + num_types]

    # Line 2: cutoff <rc_radial> <rc_angular> <MN_radial> <MN_angular>
    tokens = lines[1].split()
    n_extra = len(tokens) - 1
    if n_extra == 4:
        rc_radial = [float(tokens[1])] * num_types
        rc_angular = [float(tokens[2])] * num_types
        MN_radial = int(tokens[3])
        MN_angular = int(tokens[4])
    else:
        rc_radial = [float(tokens[1 + i * 2]) for i in range(num_types)]
        rc_angular = [float(tokens[2 + i * 2]) for i in range(num_types)]
        MN_radial = int(tokens[1 + num_types * 2])
        MN_angular = int(tokens[2 + num_types * 2])

    # Line 3: n_max <n_max_radial> <n_max_angular>
    tokens = lines[2].split()
    n_max_radial = int(tokens[1])
    n_max_angular = int(tokens[2])

    # Line 4: basis_size <basis_size_radial> <basis_size_angular>
    tokens = lines[3].split()
    basis_size_radial = int(tokens[1])
    basis_size_angular = int(tokens[2])

    # Line 5: l_max <L_max> <has_q_222> <has_q_1111> [has_q_112] [has_q_1122]
    tokens = lines[4].split()
    L_max = int(tokens[1])
    has_q_222 = int(tokens[2])
    has_q_1111 = int(tokens[3])
    has_q_112 = int(tokens[4]) if len(tokens) >= 5 else 0
    has_q_1122 = int(tokens[5]) if len(tokens) >= 6 else 0

    # Line 6: ANN <num_neurons> 0
    tokens = lines[5].split()
    num_neurons = int(tokens[1])

    # Remaining lines are float values
    float_lines_start = 6
    params = []
    for line in lines[float_lines_start:]:
        params.append(float(line.split()[0]))

    params = jnp.array(params, dtype=jnp.float32)

    num_L = L_max
    if has_q_222:
        num_L += 1
    if has_q_1111:
        num_L += 1
    if has_q_112:
        num_L += 1
    if has_q_1122:
        num_L += 1
    dim = (n_max_radial + 1) + (n_max_angular + 1) * num_L

    num_types_sq = num_types * num_types
    num_descriptor = num_types_sq * (
        (n_max_radial + 1) * (basis_size_radial + 1)
        + (n_max_angular + 1) * (basis_size_angular + 1)
    )
    num_ann = (dim + 2) * num_neurons * num_types + 1
    num_q_scaler = dim

    descriptor_params = params[:num_descriptor]
    ann_flat = params[num_descriptor:num_descriptor + num_ann]
    q_scaler = params[num_descriptor + num_ann:num_descriptor + num_ann + num_q_scaler]

    ann_params = {}
    offset = 0
    for t in range(num_types):
        w0 = ann_flat[offset:offset + num_neurons * dim].reshape(num_neurons, dim)
        offset += num_neurons * dim
        b0 = ann_flat[offset:offset + num_neurons]
        offset += num_neurons
        w1 = ann_flat[offset:offset + num_neurons]
        offset += num_neurons
        ann_params[t] = {"w0": w0, "b0": b0, "w1": w1}
    b1 = ann_flat[offset]

    return {
        "version": version,
        "num_types": num_types,
        "elements": elements,
        "rc_radial": rc_radial,
        "rc_angular": rc_angular,
        "MN_radial": MN_radial,
        "MN_angular": MN_angular,
        "n_max_radial": n_max_radial,
        "n_max_angular": n_max_angular,
        "basis_size_radial": basis_size_radial,
        "basis_size_angular": basis_size_angular,
        "L_max": L_max,
        "has_q_222": has_q_222,
        "has_q_1111": has_q_1111,
        "has_q_112": has_q_112,
        "has_q_1122": has_q_1122,
        "num_neurons": num_neurons,
        "num_L": num_L,
        "dim": dim,
        "descriptor_params": descriptor_params,
        "ann_params": ann_params,
        "b1": b1,
        "q_scaler": q_scaler,
    }


def write_nep(filepath: str, nep_dict: dict) -> None:
    """Write a nep.txt file from a dict (inverse of read_nep)."""
    version = nep_dict["version"]
    num_types = nep_dict["num_types"]
    elements = nep_dict["elements"]
    rc_radial = nep_dict["rc_radial"]
    rc_angular = nep_dict["rc_angular"]
    MN_radial = nep_dict["MN_radial"]
    MN_angular = nep_dict["MN_angular"]
    n_max_radial = nep_dict["n_max_radial"]
    n_max_angular = nep_dict["n_max_angular"]
    basis_size_radial = nep_dict["basis_size_radial"]
    basis_size_angular = nep_dict["basis_size_angular"]
    L_max = nep_dict["L_max"]
    has_q_222 = nep_dict["has_q_222"]
    has_q_1111 = nep_dict["has_q_1111"]
    has_q_112 = nep_dict.get("has_q_112", 0)
    has_q_1122 = nep_dict.get("has_q_1122", 0)
    num_neurons = nep_dict["num_neurons"]

    desc = nep_dict["descriptor_params"]
    ann_params = nep_dict["ann_params"]
    b1 = nep_dict["b1"]
    q_scaler = nep_dict["q_scaler"]
    dim = nep_dict["dim"]

    with open(filepath, "w") as f:
        # Line 1
        f.write(f"nep{version} {num_types} {' '.join(elements)}\n")
        # Line 2: cutoff
        f.write(f"cutoff {rc_radial[0]} {rc_angular[0]} {MN_radial} {MN_angular}\n")
        # Line 3: n_max
        f.write(f"n_max {n_max_radial} {n_max_angular}\n")
        # Line 4: basis_size
        f.write(f"basis_size {basis_size_radial} {basis_size_angular}\n")
        # Line 5: l_max
        f.write(f"l_max {L_max} {has_q_222} {has_q_1111}")
        if has_q_112 or has_q_1122:
            f.write(f" {has_q_112}")
        if has_q_1122:
            f.write(f" {has_q_1122}")
        f.write("\n")
        # Line 6: ANN
        f.write(f"ANN {num_neurons} 0\n")

        # Descriptor params
        d = np.asarray(desc)
        for v in d.ravel():
            f.write(f" {v:.7e}\n")

        # ANN params per type
        for t in range(num_types):
            ap = ann_params[t]
            w0 = np.asarray(ap["w0"])
            for v in w0.ravel():
                f.write(f" {v:.7e}\n")
            b0 = np.asarray(ap["b0"])
            for v in b0.ravel():
                f.write(f" {v:.7e}\n")
            w1 = np.asarray(ap["w1"])
            for v in w1.ravel():
                f.write(f" {v:.7e}\n")

        # b1
        f.write(f" {float(b1):.7e}\n")

        # q_scaler
        qs = np.asarray(q_scaler)
        for v in qs.ravel():
            f.write(f" {v:.7e}\n")
