from qiskit import QuantumCircuit, QuantumRegister
from math import log2
from qiskit.circuit.library.standard_gates.ry import RYGate
from training_image_feature_state_preparation import TrainingState
import numpy as np

#flip-flop QRAM circuit from https://www.researchgate.net/publication/370468910_Quantum_Random_Access_Memory_For_Dummies
# encodes an array into a queryable quantum state
# requires m qubits for address, n qubits for data value, 1 qubit for controlled rotation, 1 qubit for tracking
# valid superposition states, and 1 qubit for query result

def ffqram(T: list):

    m = int(log2(len(T)) + 1)
    n = int(log2(max(T)) + 1)
    norm = np.linalg.norm(np.array(T))
    
    m_reg = QuantumRegister(m, name="address")
    n_reg = QuantumRegister(n, name="value")
    cr = QuantumRegister(1, name="cr")
    flag = QuantumRegister(2, name="flag")
    query = QuantumRegister(1, name="query")

    circuit = QuantumCircuit(m_reg, n_reg, flag, cr, query)
    circuit.compose(TrainingState.qcmp(m, len(T)), qubits=m_reg[:] + list(flag[0]))
    circuit.compose(TrainingState.qcmp(n, max(T)), qubits=n_reg[:] + list(flag[1]))

    circuit.x(flag)

    for index, value in enumerate(T):
        ind_bin = '{0:b}'.format(index)
        val_bin = '{0:b}'.format(value)
        if len(ind_bin) < m:
            ind_bin = ind_bin.zfill(m)
        if len(val_bin) < n:
            val_bin = val_bin.zfill(n)
        

        
        theta = 2 * np.arcsin(value / norm)
        ctrl_state = ind_bin + val_bin + "00"
        ctrl_state = ctrl_state[::-1]
        cry = RYGate(theta).control(m + n + 2, ctrl_state=ctrl_state)
        circuit.append(cry, qargs=m_reg[:] + n_reg[:] + flag[:] + cr[:])

    return circuit
