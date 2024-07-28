from qiskit import QuantumCircuit
from src.training_image_feature_state_preparation import TrainingState
import numpy as np
from math import log2

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

def swap_main(M, feature_vec: np.ndarray, N=80) -> QuantumCircuit:

    # power of 2 that is an upper bound on M and N (# bits needed to represent M and N in binary)
    m = int(log2(M) + 1)
    n = int(log2(N) + 1)


    circuit = QuantumCircuit(m + 2*n + 11)
    circuit.append(TrainingState.prepare_initial(M, feature_vec, N), qargs=list(range(1, circuit.num_qubits)))
    circuit.append(controlled_register_swap(n + 4, m + n + 6), qargs=list(range(circuit.num_qubits)))

    return circuit