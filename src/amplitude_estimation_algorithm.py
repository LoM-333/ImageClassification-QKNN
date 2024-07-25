from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import GroverOperator, QFT
from qiskit_aer import AerSimulator
import numpy as np

class QuantumAmplitudeEstimation:

    def __init__(self, numQubits, numEvaluationQubits):
        self.numQubits = numQubits
        self.numEvaluationQubits = numEvaluationQubits

        # Prepare Quantum Circuit
        self.qr = QuantumRegister(self.numQubits)
        self.evaluation_qr = QuantumRegister(self.numEvaluationQubits)
        self.cr = ClassicalRegister(self.numEvaluationQubits)
        self.qc = QuantumCircuit(self.qr, self.evaluation_qr, self.cr)

    def buildCircuit(qc, qr, evaluation_qr, numEvaluationQubits):
        # Superposition
        qc.h(qr)

        # Amplitude
        qc.cry(2 * np.arcsin(np.sqrt(0.5)), qr[0], qr[1])

        # Grover's Oracle
        qc.cz(qr[0], qr[1])

        # Grover Operator
        qc.barrier()
        oracleCircuit = QuantumCircuit(qr)
        oracleCircuit.cz(qr[0], qr[1])
        groverOperator = GroverOperator(oracleCircuit)
        qc.append(groverOperator, qr)
        qc.barrier()

        # Amplitude Estimation
        qc.h(evaluation_qr)

        qc.barrier()

        for i in range(numEvaluationQubits):
            controlledGroverOp = groverOperator.to_instruction().control()
            qc.append(controlledGroverOp, [evaluation_qr[i]] + list(qr))

        qc.append(QFT(numEvaluationQubits, inverse=True).to_gate(), evaluation_qr)

        qc.barrier()

        # Measure
        qc.measure(evaluation_qr, qc.clbits[:numEvaluationQubits])

    # Execution
    def execute(qc):
        result = AerSimulator().run(qc, shots=1, memory=True).result
        result = result.get_memory()[0]

    def run(self):
        QuantumAmplitudeEstimation.buildCircuit(self.qc, self.qr, self.evaluation_qr, self.numEvaluationQubits)
        return QuantumAmplitudeEstimation.execute(self.qc)
    
if __name__ == '__main__':
    numQubits = 10
    numEvaluationQubits = 3
    amplitudeEstimation = QuantumAmplitudeEstimation(numQubits, numEvaluationQubits)
    result = amplitudeEstimation.run()
    print(result)