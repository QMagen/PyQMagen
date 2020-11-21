import torch
import numpy as np

def calcul_loss(obs_real, obs_iter, target_obs, T_cut, use_T_cut=True):
    loss = 0
    for obs_name in target_obs:
        # print(obs_name, T_cut[obs_name])
        # print(obs_name, getattr(obs_real, obs_name))
        max_value = getattr(obs_real, obs_name)[-np.arange(1, T_cut[obs_name], 1)].max().detach()
        max_value = max(max_value, getattr(obs_iter, obs_name)[-np.arange(1, T_cut[obs_name], 1)].max())
        # print(obs_name, sum(torch.pow((getattr(obs_real, obs_name).detach() - getattr(obs_iter, obs_name))/max_value, 2)
        #                   [-np.arange(1, T_cut[obs_name], 1)]))

        # print(len(obs_real.T), len(obs_iter.T))
        if use_T_cut==True:
            loss = loss + sum(torch.pow((getattr(obs_real, obs_name).detach() - getattr(obs_iter, obs_name))/getattr(obs_real, obs_name).detach(), 2)
                                [-np.arange(1, T_cut[obs_name], 1)])
        else:
            loss = loss + sum(torch.pow((getattr(obs_real, obs_name).detach() - getattr(obs_iter, obs_name))/getattr(obs_real, obs_name).detach(), 2))

    return loss