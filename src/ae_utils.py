import numpy as np
from qiskit.circuit import QuantumCircuit
from math import log2
from qiskit.circuit.library.standard_gates.ry import RYGate

# from https://qiskit-community.github.io/qiskit-finance/tutorials/00_amplitude_estimation.html
class Oracle(QuantumCircuit):
    """A circuit representing the Bernoulli Q operator."""

    # m is the number of state qubits
    def __init__(self, feature_vec, M, N=80):
       
        self.features = feature_vec
        self.M = M
        self.N = N

        m = int(log2(self.M) + 1)
        super().__init__(m + 1)  # circuit on m + 1 qubits
        v0 = self.features[:self.N]


        for j in range(1, self.M + 1):
            j_bin = "{0:b}".format(j)
            if len(j_bin) < m:
                j_bin = j_bin.zfill(m)

            vj = self.features[self.N * j:self.N * (j + 1)]
            p = d(v0, vj)
            theta = 4 * np.arcsin(np.sqrt(p))
            ctrl_state = j_bin
            ctrl_state = ctrl_state[::-1]
            mcry = RYGate(theta).control(m, ctrl_state=ctrl_state) #controlled rotation based on superposition state
            qargs = list(range(m + 1))
            self.append(mcry, qargs=qargs)


    def power(self, k):
        # implement the efficient power of Q

        m = int(log2(self.M) + 1)
        q_k = QuantumCircuit(m + 1)
        v0 = self.features[:self.N]

        for j in range(1, self.M + 1):
            j_bin = "{0:b}".format(j)
            if len(j_bin) < m:
                j_bin = j_bin.zfill(m)

            vj = self.features[self.N * j:self.N * (j + 1)]
            p = d(v0, vj)
            theta = 4 * k * np.arcsin(np.sqrt(p)) #likely
            ctrl_state = j_bin
            ctrl_state = ctrl_state[::-1]
            mcry = RYGate(theta).control(m, ctrl_state=ctrl_state) #controlled rotation based on superposition state
            qargs = list(range(m + 1))
            self.append(mcry, qargs=qargs)

        return q_k
    
def d(v0, v):
    return 0.5 - 0.5 * np.square(np.inner(v0, v))