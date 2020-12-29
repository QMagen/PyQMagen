import torch
# from .eigh import EigenSolver
import math
# from Codes.ExactDiagonalization import ExactDiag

class Th_obs():
    def __init__(self, params, dtype=torch.double, device='cpu'):
        self.T = params.system_params['T']
        self.beta = 1 / self.T
        self.N = len(self.T)
        self.Fe = torch.zeros([self.N, 2], dtype=dtype)
        self.Pe = torch.zeros(self.N, dtype=dtype)
        self.E = torch.zeros(self.N, dtype=dtype)
        self.C = torch.zeros(self.N, dtype=dtype)
        self.M = torch.zeros(self.N, dtype=dtype)
        self.Chi = torch.zeros(self.N, dtype=dtype)
        self.Z = torch.zeros([self.N, 1], dtype=dtype)
        self.M_paral = torch.zeros(self.N, dtype=dtype)
        self.Chi_paral = torch.zeros(self.N, dtype=dtype)

    def get_obs(self, H, SpinOp, solver_params):
        # [ESpec, U] = EigenSolver.apply(H)
        [ESpec, U] = torch.eig(H, eigenvectors=True)
        ESpec = ESpec[:, 0]
        Sztot = sum(SpinOp.SzP).double()
        Sxtot = sum(SpinOp.SxP).double()
        Mnn = torch.diag(U.transpose(1, 0).mm(Sztot).mm(U))
        Mnn_x = torch.diag(U.transpose(1, 0).mm(Sxtot).mm(U))
        MSqnn = torch.diag(U.transpose(1, 0).mm(Sztot).mm(Sztot).mm(U))
        for i in range(self.N):
            self.Z[i] = (torch.exp(-self.beta[i] * ESpec)).sum()

        for i in range(self.N):
            self.Fe[i, 0] = self.beta[i]
            self.Fe[i, 1] = -1 / self.beta[i] * torch.log((torch.exp(-self.beta[i] * ESpec)).sum())
            self.M[i] = (Mnn * torch.exp(-self.beta[i] * ESpec)).sum() / self.Z[i]
            self.M_paral[i] = (Mnn_x * torch.exp(-self.beta[i] * ESpec)).sum() / self.Z[i]
            self.E[i] = (ESpec * torch.exp(-self.beta[i] * ESpec)).sum() / self.Z[i]

        for i in range(self.N):
            self.C[i] = ((ESpec ** 2 * torch.exp(-self.beta[i] * ESpec)).sum() / self.Z[i] - self.E[i] ** 2) * \
                        self.beta[i] ** 2
            self.Chi[i] = ((MSqnn * torch.exp(-self.beta[i] * ESpec)).sum() / self.Z[i] - self.M[i] ** 2) * self.beta[i]


        self.C = self.C/solver_params['L']
        self.Chi = self.Chi/solver_params['L']
        self.M = self.M/solver_params['L']
        self.M_paral = self.M_paral / solver_params['L']
