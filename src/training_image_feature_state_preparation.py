from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, AncillaRegister
from qiskit.circuit.library import QFT, UnitaryGate, ZZFeatureMap, IntegerComparator
from math import log2, pi
import numpy as np

class TrainingState():

    @staticmethod
    # gate that makes equations (7) and (8) in the paper (hopefully)
    def ukl(M: int, v: np.ndarray, k: int, l: int, N: int = 80) -> QuantumCircuit:
        # power of 2 that is an upper bound on M and N (# bits needed to represent M and N in binary)
        m = int(log2(M) + 1)
        n = int(log2(N) + 1)

        #initialize circuit
        M_reg = QuantumRegister(m + 2)
        N_reg = QuantumRegister(n + 2)
        vector_encoding = QuantumRegister(2)
        circuit = QuantumCircuit(M_reg, N_reg, vector_encoding)

        for j in range(1, k + 1):
            for i in range(1, l + 1):
                theta = 2 * np.arcsin(v[N * (j - 1) + (i - 1)])
                circuit.ry(theta, vector_encoding[0])

        for j in range(m):
            if j != k:
                circuit.cx(j, vector_encoding[0])
        for i in range(n):
            if i != l:
                circuit.cx(i, vector_encoding[0]) # ?
        
        circuit.cx(vector_encoding[1], vector_encoding[0]) # ?
        return circuit

    @staticmethod
    def u2(M: int, v: np.ndarray, N: int = 80) -> QuantumCircuit:
        # power of 2 that is an upper bound on M and N (# bits needed to represent M and N in binary)
        m = int(log2(M) + 1)
        n = int(log2(N) + 1)

        #initialize circuit
        M_reg = QuantumRegister(m + 2)
        N_reg = QuantumRegister(n + 2)
        vector_encoding = QuantumRegister(2)
        circuit = QuantumCircuit(M_reg, N_reg, vector_encoding)

        for j in range(1, M + 1):
            for i in range(1, N + 1):
                theta = 2 * np.arcsin(v[N * (j - 1) + (i - 1)])
                circuit.ry(theta, vector_encoding[0])
                circuit.compose(TrainingState.ukl(M, v, j, i, N), inplace=True)
        circuit.swap(vector_encoding[0], vector_encoding[1])
        
        #print(circuit)
        return circuit


    @staticmethod
    #param: M is the number of training images
    #param: feature_vec is the input feature vector of dimension M*N
    #param: N is the dimension of the input feature vector (per image), which in our case, is 80
    def prepare_beta_0(M: int, N: int = 80) -> QuantumCircuit:

        # power of 2 that is an upper bound on M and N (# bits needed to represent M and N in binary)
        m = int(log2(M) + 1)
        n = int(log2(N) + 1)

        #initialize circuit
        M_reg = QuantumRegister(m + 2, name="M")
        N_reg = QuantumRegister(n + 2, name="N")
        vector_encoding = QuantumRegister(2, name="encoding")
        
        #for comparator
        M_anc = AncillaRegister(m - 1, name="M_anc") 
        N_anc = AncillaRegister(n - 1, name="N_anc")
        circuit = QuantumCircuit(M_reg, N_reg, vector_encoding, M_anc, N_anc)

        #hadamard transform; create initial superposition for M and N
        circuit.h(M_reg[:m])
        circuit.h(N_reg[:n])



        # add quantum comparator
        circuit.barrier(label="comparator")
        circuit.append(IntegerComparator(m, M + 1), qargs=(M_reg[:m+1] + M_anc[:])) #less than or equal (M+1)
        circuit.append(IntegerComparator(n, N + 1), qargs=(N_reg[:n+1] + N_anc[:]))
        circuit.barrier()

        '''
        #converts M to binary so it can be encoded into the quantum state
        bin_M = '{0:b}'.format(M)

        #encode M into qubits
        for i in range(len(bin_M)):
            if(bin_M[i] == '1'):
                circuit.x(m + i)
        
        
        '''

        # flag for 0 state (we don't want the 0 state)
        circuit.barrier()
        circuit.mcx(M_reg[:m], M_reg[-1], ctrl_state=0)
        circuit.mcx(N_reg[:n], N_reg[-1], ctrl_state=0)
        circuit.barrier()

        # print(circuit)
        return circuit

        

    @staticmethod
    #encoded state after u2
    def prepare_beta_1(M: int, feature_vec: np.ndarray, N: int = 80) -> QuantumCircuit:
        
       # power of 2 that is an upper bound on M and N (# bits needed to represent M and N in binary)
        m = int(log2(M) + 1)
        n = int(log2(N) + 1)

        #initialize circuit
        M_reg = QuantumRegister(m + 2, name="M")
        N_reg = QuantumRegister(n + 2, name="N")
        vector_encoding = QuantumRegister(2, name="encoding")
        #for comparator
        M_anc = AncillaRegister(m - 1, name="M_anc") 
        N_anc = AncillaRegister(n - 1, name="N_anc")
        circuit = QuantumCircuit(M_reg, N_reg, vector_encoding, M_anc, N_anc)

        circuit.compose(TrainingState.prepare_beta_0(M), inplace=True)

        # apply rotations to encode vector state (almost definitely wrong here)
        circuit.compose(TrainingState.u2(M, feature_vec), inplace=True)

        return circuit
