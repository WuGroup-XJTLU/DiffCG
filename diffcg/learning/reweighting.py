

class ReweightEstimator():
    def __init__(
        self,
        ref_energies,
        base_energies=None,
        volume=None,
        temperature=300.0,
        pressure=1.0,
    ):
        self.beta = 1.0 / temperature / 8.314 * 1000.0
        self.ref_energies = jnp.array(ref_energies)
        if base_energies is None:
            self.base_energies = jnp.zeros(ref_energies.shape)
        else:
            self.base_energies = jnp.array(base_energies)
        if volume is not None:
            self.pv = jnp.array(volume * pressure * 0.06023)
        else:
            self.pv = jnp.zeros(ref_energies.shape)

    def estimate_weight(self, uinit):
        unew = (uinit + self.base_energies + self.pv) * self.beta
        uref = (self.ref_energies + self.pv) * self.beta
        deltaU = unew - uref
        deltaU = deltaU - deltaU.max()
        weight = jnp.exp(-deltaU)
        weight = weight / weight.mean()
        return weight