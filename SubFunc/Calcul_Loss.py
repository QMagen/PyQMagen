import torch
import numpy as np

def calcul_loss(obs_real, obs_iter, target_obs, T_cut, use_T_cut=True):
    loss = 0
    for obs_name in target_obs:
        loss = loss + sum(torch.pow(
            # (getattr(obs_real, obs_name).detach() - getattr(obs_iter, obs_name))/getattr(obs_real, obs_name).detach()
            (getattr(obs_real, obs_name).detach() - getattr(obs_iter, obs_name))
            , 2)
                                [-np.arange(1, T_cut[obs_name], 1)])/(getattr(obs_real, obs_name).detach().max())

    return loss