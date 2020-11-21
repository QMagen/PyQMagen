import torch

class params:
    def __init__(self):
        self.Hamiltonian_params = None
        self.system_params = None

    def get_Hamiltonian_params(self, **argin):
        self.Hamiltonian_params = argin

    def get_system_params(self, **argin):
        self.system_params = argin
