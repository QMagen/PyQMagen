import torch

class params:
    def __init__(self, Lx, Ly, d, J, beta, ):
        self.Lx = Lx
        self.Ly = Ly
        self.L = Lx * Ly
        self.J1 = J1
        self.J2 = J2
        self.g = g
        self.d = d
        self.J = J
        self.Nb = len(beta)
        if type(beta) == torch.Tensor:
            self.beta = beta.clone().detach()
        else:
            self.beta = torch.tensor(beta, dtype=torch.float64, requires_grad=False)
        self.hz = hz
        self.hx = hx