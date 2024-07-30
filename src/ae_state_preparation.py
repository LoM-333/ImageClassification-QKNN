import numpy as np
from ae_utils import *
from qiskit import QuantumCircuit
from qiskit.circuit.library.standard_gates.ry import RYGate
from math import log2


def prep_state_ae(M: int, feature_vec: np.ndarray, N: int = 80):

    v0 = feature_vec[:N]
    # power of 2 that is an upper bound on M (# bits needed to represent M in binary)
    m = int(log2(M) + 1)
    circuit = QuantumCircuit(m + 1)

    #create superposition
    circuit.h(range(m))

    for j in range(1, M + 1):
            j_bin = "{0:b}".format(j)
            if len(j_bin) < m:
                j_bin = j_bin.zfill(m)

            vj = feature_vec[N * j:N * (j + 1)]
            p = d(v0, vj)
            print(p)
            theta = 2 * np.arcsin(np.sqrt(p))
            ctrl_state = j_bin
            ctrl_state = ctrl_state[::-1]
            mcry = RYGate(theta).control(m, ctrl_state=ctrl_state) #controlled rotation based on superposition state
            qargs = list(range(m + 1))
            circuit.append(mcry, qargs=qargs)

    return circuit


