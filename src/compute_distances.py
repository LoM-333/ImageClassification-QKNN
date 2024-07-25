import numpy as np
from qiskit import *

def d(v0, v):
    return 0.5 - 0.5 * np.inner(v0, v)

def compute_qubit_gamma(v0, M, V):
    num_qubits = M + 1
    qc = QuantumCircuit(num_qubits, 1)  # M qubits for |j⟩ and 1 auxiliary qubit for |γ⟩

    # Apply Hadamard gate to the auxiliary qubit
    qc.h(0)

    # Apply controlled rotations based on distance measure
    for j in range(M):
        dist = d(v0, V[j])
        theta = 2 * np.arcsin(np.sqrt(dist))
        
        qc.cry(theta, j + 1, 0)

    # Hadamard gate to the auxiliary qubit
    qc.h(0)

    # Measure the auxiliary qubit
    qc.measure(0, 0)

    return qc

if __name__ == "__main__":
    # inputs are M, training feature vector v0, set of M test feature vectors V (vj's)
    v0 = np.random.rand(80)
    v0 = v0 / np.sum(v0)

    M = 1

    V = [np.random.rand(80) for _ in range(M)]
    V[0] = V[0] / np.sum(V[0])

    qc = compute_qubit_gamma(v0, M, V)
    print(qc)
