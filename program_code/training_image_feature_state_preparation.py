from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import QFT
from math import log2, pi

class TrainingState():

    @staticmethod
    # Quantum comparator that takes an equal superposition created by the Hadamard transform
    # and creates a circuit with an ancilla that is 1 when a superposition value is strictly less than the provided threshold
    # (number of qubits in circuit is n + 1)
    # param: n is the number of qubits in the superposition state
    # param: a is the threshold value

    def qcmp(n: int, a: int) -> QuantumCircuit:

        #initialize circuit
        circuit = QuantumCircuit(n + 1)

        # **update ancilla**

        # add QFT
        circuit.compose(QFT(n + 1), inplace=True)

        # apply rotations
        for k in range(n + 1):
            theta = (-2 * pi * a) / (2 ** (n + 1 - k))
            circuit.rz(theta, k)

        # add IQFT
        circuit.compose(QFT(n + 1, inverse=True), inplace=True)

        # **return superposition back to normal**

        # add QFT
        circuit.compose(QFT(n), inplace=True)

        # apply rotations
        for k in range(n):
            theta = (2 * pi * a) / (2 ** (n - k))
            circuit.rz(theta, k)

        # add IQFT
        circuit.compose(QFT(n, inverse=True), inplace=True)

        return circuit


    @staticmethod
    #param: M is the number of training images
    #param: N is the dimension of the input feature vector, which in our case, is 80
    def prepare_training_feature_state(M: int, N: int = 80) -> QuantumCircuit:

        # power of 2 that is an upper bound on M and N (# bits needed to represent M and N in binary)
        m = int(log2(M + 1) + 1)
        n = int(log2(N + 1) + 1)

        #initialize circuit
        M_reg = QuantumRegister(m + 2)
        N_reg = QuantumRegister(n + 2)
        circuit = QuantumCircuit(M_reg, N_reg)

        #hadamard transform; create initial superposition for M and N
        circuit.h(M_reg[:m])
        circuit.h(N_reg[:n])

        # add quantum comparator
        circuit.compose(TrainingState.qcmp(m + 1, M + 1), qubits=M_reg[:m+1])
        circuit.compose(TrainingState.qcmp(n + 1, N + 1), qubits=N_reg[:n+1])

        '''
        #converts M to binary so it can be encoded into the quantum state
        bin_M = '{0:b}'.format(M)

        #encode M into qubits
        for i in range(len(bin_M)):
            if(bin_M[i] == '1'):
                circuit.x(m + i)
        
        
        '''


        # flag for 0 state (we don't want the 0 state)
        for i in range(m):
            circuit.cx(i, circuit.num_qubits-1, ctrl_state=0)

        # make sure flag behavior matches that of original circuit with qcmp
        circuit.cx(circuit.num_qubits - 1, circuit.num_qubits - 2)

        print(circuit)

       