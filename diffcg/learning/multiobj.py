# SPDX-License-Identifier: MIT
# Copyright (c) 2025 WuResearchGroup


from collections import namedtuple
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

coweighting_stats = namedtuple("coweighting_stats", 
                                ("current_iter", "num_losses", "mean_decay", "running_mean_L", "running_mean_l", "running_std_l", "running_S_l", "alphas"))

def init_coweighting_stats(num_losses):
    current_iter=-1
    mean_decay=False
    running_mean_L=jnp.zeros(num_losses)
    running_mean_l=jnp.zeros(num_losses)
    running_std_l=jnp.zeros(num_losses)
    running_S_l=jnp.zeros(num_losses)
    alphas=jnp.ones(num_losses)
    
    return coweighting_stats(current_iter,num_losses,mean_decay,running_mean_L,running_mean_l,running_std_l,running_S_l,alphas)

def coweightingloss_init():

    def coweightingloss(loss_dict,coweighting_stats):
        L, unravel = ravel_pytree(loss_dict)
        
        current_iter,num_losses,mean_decay,running_mean_L,running_mean_l,running_std_l,running_S_l,alphas = coweighting_stats

        # Increase the current iteration parameter.
        current_iter += 1

        L0=jnp.where(current_iter == 0, L, running_mean_L)

        l = L/L0 #L / L0  

        alphas=jnp.where(current_iter <=1, alphas / num_losses, (running_std_l / running_mean_L)/jnp.sum((running_std_l / running_mean_L)))

        mean_param=jnp.where(current_iter==0,0.0,(1. - 1 / (current_iter + 1)))

        x_l = l
        new_mean_l = mean_param * running_mean_L + (1 - mean_param) * x_l
        running_S_l += (x_l - running_mean_L) * (x_l - new_mean_l)
        running_mean_L = new_mean_l

        running_variance_l = running_S_l / (current_iter + 1)
        running_std_l = jnp.sqrt(running_variance_l + 1e-8)

        x_L = L
        running_mean_L = mean_param * running_mean_L + (1 - mean_param) * x_L

        weighted_losses = jnp.sum(L*alphas)

        return weighted_losses,coweighting_stats._replace(
                                                        current_iter=current_iter,
                                                        num_losses=num_losses,
                                                        mean_decay=mean_decay,
                                                        running_mean_L=running_mean_L,
                                                        running_mean_l=running_mean_l,
                                                        running_std_l=running_std_l,
                                                        running_S_l=running_S_l,
                                                        alphas=alphas)

    return coweightingloss