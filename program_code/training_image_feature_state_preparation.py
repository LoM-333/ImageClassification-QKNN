from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from math import log2

class TrainingState():

    @staticmethod
    #param: M is the number of training images
    #param: N is the dimension of the input feature vector, which in our case, is 80
    def prepare_training_feature_state(M: int, N: int = 80) -> QuantumCircuit:

        # power of 2 that is an upper bound on M
        m = int(log2(M + 1))

        #initialize circuit
        circuit = QuantumCircuit(2 * m + 2)

        #hadamard transform; create initial superposition
        circuit.h(range(m))

        #converts M to binary so it can be encoded into the quantum state
        bin_M = '{0:b}'.format(M)

        for i in range(len(bin_M)):
            
