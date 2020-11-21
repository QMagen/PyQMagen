def get_T_cut(myresult, shift = 20):
    T_cut = dict()
    N_beta = len(myresult.obs_real.beta)
    for obs_name in myresult.target_obs:
        maxi = getattr(myresult.obs_real, obs_name).argmax()
        if maxi == len(getattr(myresult.obs_real, obs_name)) - 1 or maxi == 0:
            T_cut[obs_name] = T_cut['C']
        else:
            T_cut[obs_name] = N_beta - maxi - shift
    myresult.T_cut = T_cut
