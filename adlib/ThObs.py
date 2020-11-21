import torch
from .eigh import EigenSolver
import math
# from Codes.ExactDiagonalization import ExactDiag

class Th_obs():
    def __init__(self, params, dtype=torch.double, device='cpu'):
        self.params = params
        self.Fe = torch.zeros([params.Nb, 2], dtype=dtype)
        self.Pe = torch.zeros(params.Nb, dtype=dtype)
        self.E = torch.zeros(params.Nb, dtype=dtype)
        self.C = torch.zeros(params.Nb, dtype=dtype)
        self.M = torch.zeros(params.Nb, dtype=dtype)
        self.Chi = torch.zeros(params.Nb, dtype=dtype)
        self.Z = torch.zeros([params.Nb, 1], dtype=dtype)
        self.M_paral = torch.zeros(params.Nb, dtype=dtype)
        self.Chi_paral = torch.zeros(params.Nb, dtype=dtype)
        self.size = (params.Lx * params.Ly)
        self.beta = params.beta
        self.T = 1/params.beta
        self.Nb = params.Nb
        self.J = params.J

    def get_obs(self, H, SpinOp, H_aux=None):
        [ESpec, U] = EigenSolver.apply(H)
        Sztot = sum(SpinOp.SzP).double()
        Sxtot = sum(SpinOp.SxP).double()
        Sytot = sum(SpinOp.SyP).double()
        # Sztot = sum(SpinOp.SzP)
        # Sxtot = sum(SpinOp.SxP)
        # Sytot = sum(SpinOp.SyP)
        Mnn = torch.diag(U.transpose(1, 0).mm(Sztot).mm(U))
        Mnn_x = torch.diag(U.transpose(1, 0).mm(Sxtot).mm(U))
        Mnn_y = torch.diag(U.transpose(1, 0).mm(Sytot).mm(U))
        MSqnn = torch.diag(U.transpose(1, 0).mm(Sztot).mm(Sztot).mm(U))
        MSqnn_x = torch.diag(U.transpose(1, 0).mm(Sxtot).mm(Sxtot).mm(U))
        MSqnn_y = -torch.diag(U.transpose(1, 0).mm(Sytot).mm(Sytot).mm(U))
        for i in range(self.Nb):
            self.Z[i] = (torch.exp(-self.beta[i] * ESpec)).sum()

        for i in range(self.Nb):
            self.Fe[i, 0] = self.beta[i]
            self.Fe[i, 1] = -1 / self.beta[i] * torch.log((torch.exp(-self.beta[i] * ESpec)).sum())
            self.M[i] = (Mnn * torch.exp(-self.beta[i] * ESpec)).sum() / self.Z[i]
            # self.M_x[i] = (Mnn_x * torch.exp(-self.beta[i] * ESpec)).sum() / self.Z[i]
            # self.M_y[i] = (Mnn_y * torch.exp(-self.beta[i] * ESpec)).sum() / self.Z[i]
            self.M_paral[i] = (Mnn_x * torch.exp(-self.beta[i] * ESpec)).sum() / self.Z[i]
            self.E[i] = (ESpec * torch.exp(-self.beta[i] * ESpec)).sum() / self.Z[i]

        for i in range(self.Nb):
            self.C[i] = ((ESpec ** 2 * torch.exp(-self.beta[i] * ESpec)).sum() / self.Z[i] - self.E[i] ** 2) * \
                        self.beta[i] ** 2
            self.Chi[i] = ((MSqnn * torch.exp(-self.beta[i] * ESpec)).sum() / self.Z[i] - self.M[i] ** 2) * self.beta[i]
            # self.Chi_paral[i] = ((MSqnn_x * torch.exp(-self.beta[i] * ESpec)).sum() / self.Z[i] - self.M[i] ** 2) * self.beta[i]
            # self.Chi_paral[i] = self.Chi_paral[i]/2 + (((MSqnn_y * torch.exp(-self.beta[i] * ESpec)).sum() / self.Z[i] -
            #                                            self.M[i] ** 2) * self.beta[i])/2

        self.C = self.C/self.size
        self.Chi = self.Chi/self.size
        self.M_paral = self.M_paral/self.size
        self.M = self.M/self.size
        # self.Chi_paral = self.Chi_paral/self.size

    # def get_Chi_num(self, deltah, direction='Paral'):
    #     Fe = self.Fe[:,1]
    #     args_aux = self.args
    #     if direction == 'Vert':
    #         args_aux.hz += deltah
    #     elif direction == 'Paral':
    #         args_aux.hx += deltah
    #
    #     ED_aux = ExactDiag(args=args_aux)
    #     Fe_aux = ED_aux.forward().Fe[:,1]
    #
    #     Chi_num = - (Fe_aux - Fe)/(deltah ** 2/2)
    #     Chi_num = Chi_num/self.args.L
    #     self.Chi_num = Chi_num



    def add_noise(self, level):
        self.Pe = self.Pe * (torch.ones_like(self.Pe) + torch.randn_like(self.Pe) * level)
        self.E = self.E * (torch.ones_like(self.E) + torch.randn_like(self.E) * level)
        self.C = self.C * (torch.ones_like(self.C) + torch.randn_like(self.C) * level)
        self.M = self.M * (torch.ones_like(self.M) + torch.randn_like(self.M) * level)
        self.Chi = self.Chi * (torch.ones_like(self.Chi) + torch.randn_like(self.Chi) * level)
        self.Chi_paral = self.Chi_paral * (torch.ones_like(self.Chi) + torch.randn_like(self.Chi_paral) * level)