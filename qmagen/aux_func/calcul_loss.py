import torch
import numpy as np

def calcul_loss(obs_real, obs_iter, target_obs, T_cut, use_T_cut=True, relative=False, losstype=1):
    if losstype == 1:
        loss = 0
        for obs_name in target_obs:
            if not relative:
                loss_temp = (getattr(obs_real, obs_name).detach() -
                             getattr(obs_iter, obs_name))[-np.arange(1, T_cut[obs_name], 1)]
                loss_temp = torch.pow(loss_temp, 2)
                loss_temp /= len(loss_temp)
                loss_temp = sum(loss_temp)/(getattr(obs_real, obs_name).detach().max())
                loss = loss + loss_temp
            else:
                loss_temp = (getattr(obs_real, obs_name).detach() -
                             getattr(obs_iter, obs_name))[-np.arange(1, T_cut[obs_name], 1)]
                loss_temp /= getattr(obs_real, obs_name)[-np.arange(1, T_cut[obs_name], 1)].detach()
                loss_temp = torch.pow(loss_temp, 2)
                loss_temp /= len(loss_temp)
                loss_temp = sum(loss_temp)
                loss = loss + loss_temp
        return loss