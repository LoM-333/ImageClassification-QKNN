import unittest
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import GroverOperator, QFT
from qiskit_aer import AerSimulator
import numpy as np
from src.amplitude_estimation_algorithm import qc

class TestAmplitudeEstimation(unittest.TestCase):

    def testAmplitudeEstimation(self):

        result = AerSimulator().run(qc, shots=1, memory=True).result
        result = result.get_memory()[0]

        expectedPhase = 0.5
        expectedBin = bin(int(expectedPhase * 8))[2:].zfill(3)  # 3 evaluation qubits

        probabilities = {k: v for k, v in result.items()}

        self.assertIn(expectedBin, probabilities)

        highestProbability = max(probabilities, key=probabilities.get)
        self.assertEqual(highestProbability, expectedBin))

if __name__ == '__main__':
    unittest.main() 