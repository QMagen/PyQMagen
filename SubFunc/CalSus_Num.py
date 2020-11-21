from Codes.ExactDiagonalization import ExactDiag
from copy import deepcopy

def calcul_chi_num(obs, deltah, direction='Paral'):
    params_aux = deepcopy(obs.params)
    if direction == 'Vert':
        params_aux.hz += deltah
    elif direction == 'Paral':
        params_aux.hx += deltah
    ED_aux = ExactDiag(params=params_aux)
    Fe_aux = ED_aux.forward().Fe[:,1]
    M_aux = ED_aux.forward().M_paral


    # Chi_num = - (Fe_aux.detach() - obs.Fe[:,1])/(deltah ** 2/2)
    Chi_num = (M_aux - obs.M_paral)/deltah
    # Chi_num = Chi_num/obs.params.L
    # print(Chi_num)
    obs.Chi_paral = Chi_num