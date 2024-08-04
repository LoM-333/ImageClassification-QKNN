from qiskit import QuantumCircuit, QuantumRegister
from training_image_feature_state_preparation import TrainingState
from random import randint
from math import log2
from qiskit.circuit.library.standard_gates.z import ZGate

class MinSearch():

        

    @staticmethod
    #naive implementation; will adapt for specific use case
    def durr_min_search(T: list):
        N = len(T)
        n = int(log2(N - 1) + 1) 
        y = randint(0, N - 1)
        y_ = int(log2(y - 1) + 1)
        reg_j = QuantumRegister(n + 1)
        reg_y = QuantumRegister(y_ + 1)
        target = QuantumRegister(1)
        circuit = QuantumCircuit(reg_j, reg_y, target)
        circuit.append(TrainingState.qcmp(n, N), qargs=reg_j)
        circuit.append(TrainingState.qcmp(y_, y), qargs=reg_y)
        
        print(circuit)

if __name__ == '__main__':
    MinSearch.durr_min_search([1, 2, 3, 4, 5, 6, 7, 8])
