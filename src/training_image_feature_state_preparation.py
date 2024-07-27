from qiskit import QuantumCircuit, QuantumRegister,  AncillaRegister
from qiskit.circuit.library import IntegerComparator
from math import log2
import numpy as np

class TrainingState():

    @staticmethod
    # gate that makes equations (7) and (8) in the paper (hopefully)
    def ukl_beta(M: int, v: np.ndarray, k: int, l: int, N: int = 80) -> QuantumCircuit:
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
    # gate that makes equations (7) and (8) in the paper (hopefully)
    def ukl_alpha(v: np.ndarray, l: int, N: int = 80) -> QuantumCircuit:
        # power of 2 that is an upper bound on N (# bits needed to represent  N in binary)
        n = int(log2(N) + 1)

        #initialize circuit
        N_reg = QuantumRegister(n + 2)
        vector_encoding = QuantumRegister(2)
        circuit = QuantumCircuit(N_reg, vector_encoding)


        for i in range(1, l + 1):
            theta = 2 * np.arcsin(v[(i - 1)])
            circuit.ry(theta, vector_encoding[0])

        for i in range(n):
            if i != l:
                circuit.cx(i, vector_encoding[0]) # ?
        
        circuit.cx(vector_encoding[1], vector_encoding[0]) # ?
        return circuit

    @staticmethod
    def u2_beta(M: int, v: np.ndarray, N: int = 80) -> QuantumCircuit:
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
                circuit.compose(TrainingState.ukl_beta(M, v, j, i, N), inplace=True)
        circuit.swap(vector_encoding[0], vector_encoding[1])

    @staticmethod
    def u2_alpha(v: np.ndarray, N: int = 80) -> QuantumCircuit:
        # power of 2 that is an upper bound on N (# bits needed to represent N in binary)
        n = int(log2(N) + 1)

        #initialize circuit
        N_reg = QuantumRegister(n + 2)
        vector_encoding = QuantumRegister(2)
        circuit = QuantumCircuit(N_reg, vector_encoding)

        for i in range(1, N + 1):
            theta = 2 * np.arcsin(v[(i - 1)])
            circuit.ry(theta, vector_encoding[0])
            circuit.compose(TrainingState.ukl_alpha(v, i, N), inplace=True)
        

    @staticmethod
    #param: M is the number of training images
    #param: feature_vec is the input feature vector of dimension M*N
    #param: N is the dimension of the input feature vector (per image), which in our case, is 80
    def prepare_alpha_beta_0(M: int, N: int = 80) -> QuantumCircuit:

        # power of 2 that is an upper bound on M and N (# bits needed to represent M and N in binary)
        m = int(log2(M) + 1)
        n = int(log2(N) + 1)

        #initialize circuit
        M_beta = QuantumRegister(m + 2, name="M_beta")
        N_beta = QuantumRegister(n + 2, name="N_beta")
        vector_encoding_beta = QuantumRegister(2, name="encoding_beta")

        N_alpha = QuantumRegister(n + 2, name="N_alpha")
        vector_encoding_alpha = QuantumRegister(2, name="encoding_alpha")
        N_anc_alpha = AncillaRegister(n - 1, name="N_anc_alpha")
        
        #for comparator
        M_anc_beta = AncillaRegister(m - 1, name="M_anc_beta") 
        N_anc_beta = AncillaRegister(n - 1, name="N_anc_beta")
        circuit = QuantumCircuit(N_alpha, vector_encoding_alpha, N_anc_alpha, M_beta, N_beta, vector_encoding_beta, M_anc_beta, N_anc_beta)

        #hadamard transform; create initial superposition for M and N
        circuit.h(M_beta[:m])
        circuit.h(N_beta[:n])


        # add quantum comparator
        circuit.barrier(label="comparator")
        circuit.append(IntegerComparator(n, N + 1), qargs=(N_alpha[:n+1] + N_anc_alpha[:]))
        circuit.append(IntegerComparator(m, M + 1), qargs=(M_beta[:m+1] + M_anc_beta[:])) #less than or equal (M+1)
        circuit.append(IntegerComparator(n, N + 1), qargs=(N_beta[:n+1] + N_anc_beta[:]))
        circuit.barrier()


        # flag for 0 state (we don't want the 0 state)
        circuit.barrier()
        circuit.mcx(M_beta[:m], M_beta[-1], ctrl_state=0)
        circuit.mcx(N_beta[:n], N_beta[-1], ctrl_state=0)
        circuit.mcx(N_alpha[:n], N_alpha[-1], ctrl_state=0)
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
        M_beta = QuantumRegister(m + 2, name="M_beta")
        N_beta = QuantumRegister(n + 2, name="N_beta")
        vector_encoding_beta = QuantumRegister(2, name="encoding_beta")

        N_alpha = QuantumRegister(n + 2, name="N_alpha")
        vector_encoding_alpha = QuantumRegister(2, name="encoding_alpha")
        N_anc_alpha = AncillaRegister(n - 1, name="N_anc_alpha")
        
        #for comparator
        M_anc_beta = AncillaRegister(m - 1, name="M_anc_beta") 
        N_anc_beta = AncillaRegister(n - 1, name="N_anc_beta")
        circuit = QuantumCircuit(N_alpha, vector_encoding_alpha, N_anc_alpha, M_beta, N_beta, vector_encoding_beta, M_anc_beta, N_anc_beta)

        circuit.compose(TrainingState.prepare_beta_0(M), inplace=True)

        # apply rotations to encode vector state
        circuit.append(TrainingState.u2_beta(M, feature_vec), qargs=M_beta[:] + N_beta[:] + vector_encoding_beta[:])

        return circuit
    
    @staticmethod
    #replicate prepare_beta_1 but for alpha
    def prepare_initial(M: int, feature_vec: np.ndarray, N: int = 80) -> QuantumCircuit:
        
       # power of 2 that is an upper bound on M and N (# bits needed to represent M and N in binary)
        m = int(log2(M) + 1)
        n = int(log2(N) + 1)

        #initialize circuit
        M_beta = QuantumRegister(m + 2, name="M_beta")
        N_beta = QuantumRegister(n + 2, name="N_beta")
        vector_encoding_beta = QuantumRegister(2, name="encoding_beta")

        N_alpha = QuantumRegister(n + 2, name="N_alpha")
        vector_encoding_alpha = QuantumRegister(2, name="encoding_alpha")
        N_anc_alpha = AncillaRegister(n - 1, name="N_anc_alpha")
        
        #for comparator
        M_anc_beta = AncillaRegister(m - 1, name="M_anc_beta") 
        N_anc_beta = AncillaRegister(n - 1, name="N_anc_beta")
        circuit = QuantumCircuit(N_alpha, vector_encoding_alpha, N_anc_alpha, M_beta, N_beta, vector_encoding_beta, M_anc_beta, N_anc_beta)

        circuit.compose(TrainingState.prepare_beta_1(M), inplace=True)

        # apply rotations to encode vector state
        circuit.append(TrainingState.u2_alpha(feature_vec), qargs=N_alpha[:] + vector_encoding_alpha[:])

        return circuit
