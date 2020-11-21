from solver_ED import solver_ED
from copy import deepcopy

def calcul_chi_num(obs, params, solver_params, deltah, direction='Paral'):
    params_aux = deepcopy(params)
    if direction == 'Vert':
        params_aux.Hamiltonian_params['hz'] += deltah
    elif direction == 'Paral':
        params_aux.Hamiltonian_params['hx'] += deltah

    # solver_ED_aux = solver_ED(L=solver_params['L'])
    solver_ED_aux = solver_ED(**solver_params)
    Fe_aux = solver_ED_aux.forward(params=params_aux).Fe[:,1]
    M_aux = solver_ED_aux.forward(params=params_aux).M_paral


    # Chi_num = - (Fe_aux.detach() - obs.Fe[:,1])/(deltah ** 2/2)
    Chi_num = (M_aux - obs.M_paral)/deltah
    # Chi_num = Chi_num/obs.params.L
    obs.Chi_paral = Chi_num