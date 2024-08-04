from qiskit import QuantumCircuit, QuantumRegister
from math import log2

#flip-flop QRAM circuit from https://www.researchgate.net/publication/370468910_Quantum_Random_Access_Memory_For_Dummies
# encodes an array into a queryable quantum state
# requires m qubits for address, n qubits for data value, 1 qubit for controlled rotation, 1 qubit for tracking
# valid superposition states, and 1 qubit for query result

def ffqram(T: list):

    