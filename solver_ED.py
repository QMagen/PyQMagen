import torch
from magen.adlib import SpinOp
from magen.adlib import Th_obs

class solver_ED(torch.nn.Module):

    def __init__(self, **argin):
        super(solver_ED, self).__init__()
        self.solver_params = argin

    def forward(self, params):
        obs = Th_obs(params=params)
        H, SpinOps = self.construct_hamiltonian(params.Hamiltonian_params)
        obs.get_obs(H=H, SpinOp=SpinOps, solver_params=self.solver_params)
        return obs

    def construct_hamiltonian(self, Hamilton_params):
        # print(SpinOp.SxP[1])
        L = self.solver_params['L']
        SpinOps = SpinOp(d=Hamilton_params['d'], L=L)
        Jxy = Hamilton_params['Jxy']
        Jz = Hamilton_params['Jz']
        H = torch.zeros_like(Jxy.mul(SpinOps.SxP[1] * SpinOps.SxP[1]) -
                             Jxy.mul(SpinOps.SyP[1] * SpinOps.SyP[1]) +
                             Jz.mul(SpinOps.SzP[1] * SpinOps.SzP[1]),
                             requires_grad=False, dtype=torch.float64)

        def ADD(H, SxP, SyP, SzP, i, j, Jxy, Jz):
            H = H + Jxy.mul(SxP[i].mm(SxP[j])) - Jxy.mul(SyP[i].mm(SyP[j])) + Jz.mul(SzP[i].mm(SzP[j]))
            return H

        for i in range(L):
            if i < L - 1:
                H = ADD(H, SpinOps.SxP, SpinOps.SyP, SpinOps.SzP, i, i + 1, Jxy, Jz)
        H = ADD(H, SpinOps.SxP, SpinOps.SyP, SpinOps.SzP, L - 1, 0, Jxy, Jz)
        H = H - Hamilton_params['hz'] * SpinOps.Sztot
        H = H - Hamilton_params['hx'] * SpinOps.Sxtot
        return H, SpinOps








