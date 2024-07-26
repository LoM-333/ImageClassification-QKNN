import unittest
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import GroverOperator, QFT
from qiskit_aer import AerSimulator
import numpy as np
from src.amplitude_estimation_algorithm import QuantumAmplitudeEstimation

class TestAmplitudeEstimation(unittest.TestCase):

    def testAmplitudeEstimation(self):
        numQubits = 10
        numEvaluationQubits = 3
        amplitudeEstimation = QuantumAmplitudeEstimation(numQubits, numEvaluationQubits)
        result = amplitudeEstimation.run()

        self.assertEqual(len(result), numEvaluationQubits)
        self.assertTrue(all(bit in '01' for bit in result))

    def testSuperposition(self):
        numQubits = 10
        numEvaluationQubits = 3
        amplitudeEstimation = QuantumAmplitudeEstimation(numQubits, numEvaluationQubits)

        qc = amplitudeEstimation.qc
        QuantumAmplitudeEstimation.buildCircuit(qc, amplitudeEstimation.qr, amplitudeEstimation.evaluation_qr, numEvaluationQubits)

        result = AerSimulator().run(qc, shots=1, memory=True).result
        result = result.get_memory()[0]

        self.assertTrue(all(state in result for state in result))

if __name__ == '__main__':
    unittest.main() 