import unittest
import numpy as np
from qiskit import *
from qiskit_aer import AerSimulator
from src.compute_distances import d, compute_qubit_gamma

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

# Test cases for unittest
class TestQuantumCircuit(unittest.TestCase):
    def test_distance_computation(self):
        N = 80
        v0 = np.random.rand(N)
        v0 = v0 / np.linalg.norm(v0)

        M = 1

        V = [np.random.rand(N) for _ in range(M)]
        V[0] = V[0] / np.linalg.norm(V[0])

        # Check distance function
        dist = d(v0, V[0])
        self.assertGreaterEqual(dist, 0)
        self.assertLessEqual(dist, 1)

    def test_quantum_circuit(self):
        N = 80
        v0 = np.random.rand(N)
        v0 = v0 / np.linalg.norm(v0)

        M = 1

        V = [np.random.rand(N) for _ in range(M)]
        V[0] = V[0] / np.sum(V[0])

        qc = compute_qubit_gamma(v0, M, V)

        # Check if the circuit has the correct number of qubits and classical bits
        self.assertEqual(qc.num_qubits, M + 1)
        self.assertEqual(qc.num_clbits, 1)

        

        # Simulate the circuit
        simulator = AerSimulator()
        compiled_circuit = transpile(qc, simulator)
        result = AerSimulator().run(compiled_circuit, shots=1024).result()

        # Get measurement results
        counts = result.get_counts(qc)
        self.assertIn('0', counts)
        self.assertIn('1', counts)

if __name__ == "__main__":
    unittest.main()
