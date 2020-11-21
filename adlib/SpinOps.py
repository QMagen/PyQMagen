import torch
from .kronecker_product import kronecker_product
import math

class SpinOp:
    def __init__(self, params):
        d = params.d
        Sz = torch.diag(torch.tensor([-(d - 1) / 2 + n - 1 for n in range(1, d + 1)], requires_grad=False))
        Sup = torch.zeros([d, d], requires_grad=False)
        Sdn = torch.zeros([d, d], requires_grad=False)
        Id = torch.eye(d, requires_grad=False)
        for n in range(1, d + 1):
            mz = -(d - 1) / 2 + (n - 1)
            for m in range(1, d + 1):
                if (m == n + 1): Sup[m - 1, n - 1] = math.sqrt((d ** 2 - 1) / 4 - mz ** 2 - mz)
                if (m == n - 1): Sdn[m - 1, n - 1] = math.sqrt((d ** 2 - 1) / 4 - mz ** 2 + mz)
        Sx = (Sup + Sdn) / 2
        Sy = (Sup - Sdn) / (-2)
        Sx.requires_grad_(True)
        Sy.requires_grad_(True)
        Sz.requires_grad_(True)
        Id.requires_grad_(True)
        SxP = []
        SyP = []
        SzP = []

        def DirectProd(S, i, L, Id):
            SP = S
            for si in range(L):
                if si < i:
                    SP = kronecker_product(SP, Id)
                if si > i:
                    SP = kronecker_product(Id, SP)
            return SP

        for i in range(params.L):
            SxP.append(DirectProd(Sx, i, params.L, Id))
            SyP.append(DirectProd(Sy, i, params.L, Id))
            SzP.append(DirectProd(Sz, i, params.L, Id))
        self.Sx = Sx
        self.Sy = Sy
        self.Sz = Sz
        self.Id = Id
        self.SxP = SxP
        self.SyP = SyP
        self.SzP = SzP
        self.Sztot = sum(SzP)
        self.Sxtot = sum(SxP)
        self.Sytot = sum(SyP)