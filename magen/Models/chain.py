import torch

class SpinChain:

    def __init__(self, l=8):
        self.l = l

    def generate_interactions(self, params):
        return None

class UniformSpinChain(SpinChain):

    def generate_interactions(self, J):
        J = torch.tensor(J)
        interactions = []
        for i in range(self.l - 1):
            interactions.append(['x', 'x', i, i+1, J])
            interactions.append(['y', 'y', i, i + 1, J])
            interactions.append(['z', 'z', i, i + 1, J])

        interactions.append(['x', 'x', self.l - 1, 0, J])
        interactions.append(['y', 'y', self.l - 1, 0, J])
        interactions.append(['z', 'z', self.l - 1, 0, J])

        return interactions

class XXZSpinChain(SpinChain):

    def generate_interactions(self, Jxy, Jz):
        Jxy = torch.tensor(Jxy)
        Jz = torch.tensor(Jz)
        interactions = []
        for i in range(self.l - 1):
            interactions.append(['x', 'x', i, i+1, Jxy])
            interactions.append(['y', 'y', i, i + 1, Jxy])
            interactions.append(['z', 'z', i, i + 1, Jz])

        interactions.append(['x', 'x', self.l - 1, 0, Jxy])
        interactions.append(['y', 'y', self.l - 1, 0, Jxy])
        interactions.append(['z', 'z', self.l - 1, 0, Jz])

        return interactions

class XYZSpinChain(SpinChain):

    def generate_interactions(self, Jx, Jy, Jz):
        Jx = torch.tensor(Jx)
        Jy = torch.tensor(Jy)
        Jz = torch.tensor(Jz)
        interactions = []
        for i in range(self.l - 1):
            interactions.append(['x', 'x', i, i+1, Jx])
            interactions.append(['y', 'y', i, i + 1, Jy])
            interactions.append(['z', 'z', i, i + 1, Jz])

        interactions.append(['x', 'x', self.l - 1, 0, Jx])
        interactions.append(['y', 'y', self.l - 1, 0, Jy])
        interactions.append(['z', 'z', self.l - 1, 0, Jz])

        return interactions

class AlterXYZChain(SpinChain):

    def generate_interactions(self, Jx, Jy, Jz, alpha):
        Jx = torch.tensor(Jx)
        Jy = torch.tensor(Jy)
        Jz = torch.tensor(Jz)
        alpha = torch.tensor(alpha)

        pass