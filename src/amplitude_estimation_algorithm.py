from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import GroverOperator, QFT
from qiskit_aer import AerSimulator
import numpy as np

# Prepare Quantum Circuit (edit with actual circuit)
numQubits = 10

qr = QuantumRegister(numQubits)
cr = ClassicalRegister(numQubits)
qc = QuantumCircuit(qr, cr)

# Superposition
qc.h(qr)

# Amplitude
qc.cry(2 * np.arcsin(np.sqrt(0.5)), qr[0], qr[1])

# Grover's Oracle
def oracle(qc. qr):
    qc.cz(qr[0], qr[1])

oracle(qc, qr)

# Grover Operator
qc.barrier()
groverOperator = GroverOperator(oracle(qc, qr))
qc.barrier()

qc.append(groverOperator, qr)

# Amplitude Estimation
numEvaluationQubits = 3
evaluation_qr = QuantumRegister(numEvaluationQubits)
qc.add_register(evaluation_qr)

qc.h(evaluation_qr)

qc.barrier()

for i in range(numEvaluationQubits):
    qc.append(groverOperator(qc, qr).to_instruction().control(), [evaluation_qr[i]] + list(qr))

qc.append(QFT(numEvaluationQubits, inverse=True).to_gate(), evaluation_qr)

qc.barrier()

# Measure
qc.measure(evaluation_qr, cr[:numEvaluationQubits])

# Execution
result = AerSimulator().run(qc, shots=1, memory=True).result
result = result.get_memory()[0]

print(result)