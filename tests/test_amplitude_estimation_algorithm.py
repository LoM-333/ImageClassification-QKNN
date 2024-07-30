import unittest
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import GroverOperator, QFT
from qiskit_aer import AerSimulator
import numpy as np
from src.amplitude_estimation_algorithm import QuantumAmplitudeEstimation
from src.ae_state_preparation import prep_state_ae

class TestAmplitudeEstimation(unittest.TestCase):

    def test_prep_state_ae(self):
        v = np.array([x for x in range(800)])
        v = v / np.linalg.norm(v)

        circuit = prep_state_ae(10, v)

        circuit.measure_all()

        checkResult = AerSimulator().run(transpile(circuit, AerSimulator()), shots=1024, memory=True).result()
        measurements = checkResult.get_memory()

        for i in measurements:
            i = i[::-1]
            M = int(i[:4], 2)
            result = i[-1]

            if (M > 10 or M == 0) and result =='1':
                self.fail("false positive")
            print(f"Measurement: {i}")
            print(f"m: {int(i[:4], 2)}")
            print(f"result: {i[-1]}")


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