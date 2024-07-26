import unittest
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT
from math import log2, pi
from src.training_image_feature_state_preparation import TrainingState

class TestTrainingState(unittest.TestCase):

    def test_qcmp(self):
        n = 3
        a = 5
        qc = TrainingState.qcmp(n, a)

        # Number of Qubits
        self.assertEqual(qc.num_qubits, n + 1)
        
        # Circuit Depth
        self.assertTrue(qc.depth() > 0)

        operations = [operation[0].name for operation in qc.data]
        self.assertIn(operations)   # QFT
        self.assertIn(operations)   # inverse QFT

        # RZ Gates
        rz_gates = [operation for operation in qc.data if operation[0].name == 'rz']
        self.assertEqual(len(rz_gates), n + n + 1)

    def testRotations(self):
        n = 3
        a = 5
        qc = TrainingState.qcmp(n, a)

        rz_gates = [operation for operation in qc.data if operation[0].name == 'rz']

        # RZ Angles
        expectedAngles = [(-2 * pi * a) / (2 ** (n + 1 - k)) for k in range(n + 1)]
        for rz, expectedAngle in zip(rz_gates[:n+1], expectedAngles):
            self.assertAlmostEqual(rz[0].params[0], expectedAngle, places=6)
        expectedAngles = [(-2 * pi * a) / (2 ** (n + 1 - k)) for k in range(n + 1)]
        for rz, expectedAngle in zip(rz_gates[:n+1], expectedAngles):
            self.assertAlmostEqual(rz[0].params[0], expectedAngle, places=6)

    def testPrepareTrainingFeatureState(self):
        M = 10
        N = 80
        qc = TrainingState.prepare_training_feature_state(M, N)

        m = int(log2(M + 1) + 1)
        n = int(log2(N + 1) + 1)

        # Number of Qubits
        self.assertEqual(qc.num_qubits, m + 2 + n + 2)

        # Circuit Depth
        self.assertTrue(qc.depth() > 0)

        # Hadamard Gates
        hGates = [operation for operation in qc.data if operation[0].name == 'h']
        self.assertEqual(len(hGates), m + n)

        operations = [operation[0].name for operation in qc.data]
        self.assertIn(operations)   # QFT
        self.assertIn(operations)   # inverse QFT

if __name__ == '__main__':
    unittest.main()