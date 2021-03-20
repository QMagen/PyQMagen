from magen.adlib import ThObs
from magen.adlib import SpinOperators
import warnings
import torch


class EDSolver:

    def __init__(self, size=None, cal_chi_para=False):
        if not size:
            raise ValueError('Please input system size')
        elif size >= 12:
            warnings.warn('Running ED with more than 12 points is not recommended', RuntimeWarning, stacklevel=2)
        self.size = size
        self.cal_chi_para = cal_chi_para

    def get_hamiltonian_matrix(self, interactions, mag_field=None):
        SpinOps = SpinOperators(d=2, l=self.size)
        hamiltonian_matrix = torch.zeros_like(SpinOps.SxP[0],
                                              requires_grad=False,
                                              dtype=torch.float64)

        def ADD(hamiltonian_temp, SpinOps_temp, direction, i, j, J):

            if (direction[0], direction[1]) == ('x', 'x'):
                spin_operator_left = SpinOps_temp.SxP[i]
                spin_operator_right = SpinOps_temp.SxP[j]
            elif (direction[0], direction[1]) == ('y', 'y'):
                spin_operator_left = SpinOps_temp.SyP[i]
                spin_operator_right = SpinOps_temp.SyP[j]
            elif (direction[0], direction[1]) == ('z', 'z'):
                spin_operator_left = SpinOps_temp.SzP[i]
                spin_operator_right = SpinOps_temp.SzP[j]
            else:
                raise ValueError('Only "xx", "yy", "zz" interactions are implemented')

            if (direction[0], direction[1]) == ('y', 'y'):
                hamiltonian_temp = hamiltonian_temp - J.mul(spin_operator_left.mm(spin_operator_right))
            else:
                hamiltonian_temp = hamiltonian_temp + J.mul(spin_operator_left.mm(spin_operator_right))
            return hamiltonian_temp

        for interaction in interactions:
            if type(interaction) is not tuple and type(interaction) is not list:
                raise TypeError('Interactions must be defined as a tuple or list: ("z", "z", 1, 2, 1.5)')
            elif interaction[0] not in ['z', 'x', 'y'] or interaction[1] not in ['z', 'x', 'y']:
                raise TypeError('First two elements must indicate be "z" "x" or "y"')
            elif type(interaction[2]) is not int or type(interaction[3]) is not int:
                raise TypeError('3rd and 4th elements must indicate be int')
            elif interaction[2] >= self.size or interaction[3] >= self.size:
                raise ValueError('3rd and 4th elements must not exceed system size')

            hamiltonian_matrix = ADD(hamiltonian_matrix, SpinOps, (interaction[0], interaction[1]),
                                     interaction[2], interaction[3], interaction[4])

        if mag_field:
            hamiltonian_matrix = hamiltonian_matrix - mag_field['hz'] * SpinOps.Sztot
            hamiltonian_matrix = hamiltonian_matrix - mag_field['hx'] * SpinOps.Sxtot

        return hamiltonian_matrix, SpinOps

    def _solve_from_iteraction(self, interactions, T, mag_field=None):
        h_matrix, SpinOps = self.get_hamiltonian_matrix(interactions=interactions, mag_field=mag_field)
        sim_therm_data = ThObs(T=T)

        [ESpec, U] = torch.symeig(h_matrix, eigenvectors=True)
        Sztot = sum(SpinOps.SzP).double()
        Sxtot = sum(SpinOps.SxP).double()
        Mnn = torch.diag(U.transpose(1, 0).mm(Sztot).mm(U))
        Mnn_x = torch.diag(U.transpose(1, 0).mm(Sxtot).mm(U))
        MSqnn = torch.diag(U.transpose(1, 0).mm(Sztot).mm(Sztot).mm(U))
        for i in range(sim_therm_data.N):
            sim_therm_data.Z[i] = (torch.exp(-sim_therm_data.beta[i] * ESpec)).sum()

        for i in range(sim_therm_data.N):
            sim_therm_data.Fe[i, 0] = sim_therm_data.beta[i]
            sim_therm_data.Fe[i, 1] = -1 / sim_therm_data.beta[i] * \
                                              torch.log((torch.exp(-sim_therm_data.beta[i] * ESpec)).sum())
            sim_therm_data.M[i] = (Mnn * torch.exp(-sim_therm_data.beta[i] * ESpec)).sum() / sim_therm_data.Z[i]
            sim_therm_data.M_paral[i] = (Mnn_x * torch.exp(-sim_therm_data.beta[i] * ESpec)).sum() / sim_therm_data.Z[i]
            sim_therm_data.E[i] = (ESpec * torch.exp(-sim_therm_data.beta[i] * ESpec)).sum() / sim_therm_data.Z[i]

        for i in range(sim_therm_data.N):
            sim_therm_data.C[i] = ((ESpec ** 2 * torch.exp(-sim_therm_data.beta[i] * ESpec)).sum() / sim_therm_data.Z[i]
                                   - sim_therm_data.E[i] ** 2) * sim_therm_data.beta[i] ** 2
            sim_therm_data.Chiz[i] = ((MSqnn * torch.exp(-sim_therm_data.beta[i] * ESpec)).sum() / sim_therm_data.Z[i]
                                     - sim_therm_data.M[i] ** 2) * sim_therm_data.beta[i]

        sim_therm_data.C /= self.size
        sim_therm_data.Chiz /= self.size
        sim_therm_data.M /= self.size
        sim_therm_data.M_paral /= self.size

        return sim_therm_data

    def forward(self, interactions, T, mag_field=None, with_chi_para=False):

        sim_therm_data = self._solve_from_iteraction(interactions, T, mag_field)
        mag_field_aux = {}

        if with_chi_para or self.cal_chi_para:
            if mag_field:
                mag_field_aux['hx'] = mag_field['hx'] + 0.05
                sim_therm_data_aux = self._solve_from_iteraction(interactions, T, mag_field_aux)
                M_aux = sim_therm_data_aux.M
                Chi_num = (M_aux - sim_therm_data.M)/0.05
            else:
                mag_field_aux = {'hx': 0.05, 'hz': 0.0}
                sim_therm_data_aux = self._solve_from_iteraction(interactions, T, mag_field_aux)
                M_aux = sim_therm_data_aux.M_paral
                Chi_num = (M_aux - sim_therm_data.M_paral) / 0.05
            sim_therm_data.Chixy = Chi_num
        return sim_therm_data

