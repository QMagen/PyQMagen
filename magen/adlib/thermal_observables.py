import torch
# from .eigh import EigenSolver
import math
# from Codes.ExactDiagonalization import ExactDiag

class ThObs():
    def __init__(self, T, dtype=torch.double, device='cpu'):
        self.T = T
        self.beta = 1 / self.T
        self.N = len(self.T)
        self.Fe = torch.zeros([self.N, 2], dtype=dtype)
        self.Pe = torch.zeros(self.N, dtype=dtype)
        self.E = torch.zeros(self.N, dtype=dtype)
        self.C = torch.zeros(self.N, dtype=dtype)
        self.M = torch.zeros(self.N, dtype=dtype)
        self.Chiz = torch.zeros(self.N, dtype=dtype)
        self.Z = torch.zeros([self.N, 1], dtype=dtype)
        self.M_paral = torch.zeros(self.N, dtype=dtype)
        self.Chixy = torch.zeros(self.N, dtype=dtype)