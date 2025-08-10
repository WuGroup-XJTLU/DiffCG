import jax.numpy as jnp
from diffcg.util.math import high_precision_sum
from diffcg.common.error import MSE

def init_independent_mse_loss_fn(quantities):
    """
    Initializes the default loss function, where MSE errors of destinct quantities are added.

    First, observables are computed via the reweighting scheme. These observables can be ndarray
    valued, e.g. vectors for RDF / ADF or matrices for stress. For each observable, the element-wise
    MSE error is computed wrt. the target provided in "quantities[quantity_key]['target']".
    This per-quantity loss is multiplied by gamma in "quantities[quantity_key]['gamma']". The final loss is
    then the sum over all of these weighted per-quantity MSE losses.
    A pre-requisite for using this function is that observables are simply ensemble averages of
    instantaneously fluctuating quantities. If this is not the case, a custom loss_fn needs to be defined.
    The custom loss_fn needs to have the same input-output signuture as the loss_fn implemented here.


    Args:
        quantities: The quantity dict with 'compute_fn', 'gamma' and 'target' for each observable

    Returns:
        The loss_fn taking trajectories of fluctuating properties, computing ensemble averages via the
        reweighting scheme and outputs the loss and predicted observables.

    """
    def loss_fn(quantity_trajs, weights):
        loss = 0.
        predictions = {}
        for quantity_key in quantities:
            quantity_snapshots = quantity_trajs[quantity_key]
            weighted_snapshots = (quantity_snapshots.T * weights).T
            ensemble_average = high_precision_sum(weighted_snapshots, axis=0)  # weights account for "averaging"
            predictions[quantity_key] = ensemble_average
            loss += quantities[quantity_key]['gamma'] * MSE(ensemble_average, quantities[quantity_key]['target'])
        return loss, predictions
    return loss_fn

class ReweightEstimator():
    def __init__(
        self,
        ref_energies,
        base_energies=None,
        volume=None,
        kBT=1.0,
        pressure=1.0,
    ):
        self.beta = 1.0 / kBT
        self.ref_energies = jnp.array(ref_energies)
        if base_energies is None:
            self.base_energies = jnp.zeros(ref_energies.shape)
        else:
            self.base_energies = jnp.array(base_energies)
        if volume is not None:
            self.pv = jnp.array(volume * pressure * 0.06023)
        else:
            self.pv = jnp.zeros(ref_energies.shape)

    def estimate_effective_samples(self,weights):
        weights = jnp.where(weights > 1.e-10, weights, 1.e-10)  # mask to avoid NaN from log(0) if a few weights are 0.
        exponent = - jnp.sum(weights * jnp.log(weights))
        return jnp.exp(exponent)

    def estimate_weight(self, uinit):
        unew = uinit + self.base_energies + self.pv
        uref = self.ref_energies + self.pv
        exponent = (unew - uref) * self.beta
        exponent = exponent - exponent.max()
        prob_ratios = jnp.exp(-exponent)
        weight = prob_ratios / high_precision_sum(prob_ratios)
        n_eff = self.estimate_effective_samples(weight)
        return weight, n_eff