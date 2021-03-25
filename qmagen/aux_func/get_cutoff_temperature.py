def get_cutoff_t(target_obs, obs_real, shift=0):
    T_cut = dict()
    N_beta = len(obs_real.T)
    for obs_name in target_obs:
        maxi = getattr(obs_real, obs_name).argmax()
        if maxi == len(getattr(obs_real, obs_name)) - 1 or maxi == 0:
            T_cut[obs_name] = T_cut['C']
        else:
            T_cut[obs_name] = N_beta - maxi - shift
    return T_cut
