import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import QFT
from src.controlled_swap import *

def controlled_register_swap(len_a, len_b) -> QuantumCircuit:
    circuit = QuantumCircuit(len_a + len_b + 1)
    circuit.h(0)
    for i in range((len_a + len_b) // 2):
        circuit.cswap(0, i + 1, len_a + len_b - i)
    for i in range(len_b // 2):
        circuit.cswap(0, i + 1, len_b - i)
    for i in range(len_a // 2):
        circuit.cswap(0, len_b + i + 1, len_a + len_b - i)
    circuit.h(0)
    return circuit

def d(v0, v):
    return 0.5 - 0.5 * np.square(np.inner(v0, v))

def AmplitudeEstimation(M, v0, V, reg1, reg2, evalQubits):
    swapCircuit = controlled_register_swap(reg1, reg2)
    qc = QuantumRegister(swapCircuit)
    qc.compose(swapCircuit, inplace=True)

    for j in range(M):
        dist = d(v0, V[j])
        theta = 2 * np.arcsin(np.sqrt(dist))

        qc.cry(theta, j + 1, 0)

    # Phase Estimation **SEND HELP**
    evaluation_qr = QuantumRegister(evalQubits)

    qc.barrier()
    qc.append(QFT(evalQubits, inverse=True), evaluation_qr)
    qc.barrier()

    return qc

if __name__ == "__main__":
    # inputs are M, training feature vector v0, set of M test feature vectors V (vj's)
    v0 = np.random.rand(80)
    v0 = v0 / np.sum(v0)

    M = 1

    V = [np.random.rand(80) for _ in range(M)]
    V[0] = V[0] / np.sum(V[0])

    qc = AmplitudeEstimation(v0, M, V)
    print(qc)