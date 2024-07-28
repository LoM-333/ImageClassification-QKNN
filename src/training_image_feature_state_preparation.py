from qiskit import QuantumCircuit, QuantumRegister,  AncillaRegister
from qiskit.circuit.library import IntegerComparator, QFT
from math import log2, pi
import numpy as np

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

        #reverse register
        if n > 1:
            for i in range(n // 2):
                circuit.swap(i, n - i - 1)

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

        #reverse register
        if n > 1:
            for i in range(n // 2):
                circuit.swap(i, n - i - 1)

        return circuit

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
                circuit.mcx([j, M_reg[-2], M_reg[-1]], vector_encoding[0], ctrl_state="100")
        for i in range(n):
            if i != l:
                circuit.mcx([i, N_reg[-2], N_reg[-1]], vector_encoding[0], ctrl_state="100")
        
        circuit.cx(vector_encoding[1], vector_encoding[0])
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
                circuit.mcx([i, N_reg[-2], N_reg[-1]], vector_encoding[0], ctrl_state="100")
        
        circuit.cx(vector_encoding[1], vector_encoding[0])
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
                #theta = 2 * np.arcsin(v[N * (j - 1) + (i - 1)])
                #circuit.ry(theta, vector_encoding[0])
                circuit.compose(TrainingState.ukl_beta(M, v, j, i, N), inplace=True)
        circuit.swap(vector_encoding[0], vector_encoding[1])

        return circuit

    @staticmethod
    def u2_alpha(v: np.ndarray, N: int = 80) -> QuantumCircuit:
        # power of 2 that is an upper bound on N (# bits needed to represent N in binary)
        n = int(log2(N) + 1)

        #initialize circuit
        N_reg = QuantumRegister(n + 2)
        vector_encoding = QuantumRegister(2)
        circuit = QuantumCircuit(N_reg, vector_encoding)

        for i in range(1, N + 1):
            #theta = 2 * np.arcsin(v[(i - 1)])
            #circuit.ry(theta, vector_encoding[0])
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

        circuit = QuantumCircuit(N_alpha, vector_encoding_alpha, M_beta, N_beta, vector_encoding_beta)

        #hadamard transform; create initial superposition for M and N
        circuit.h(M_beta[:m])
        circuit.h(N_beta[:n])


        # add quantum comparator
        circuit.barrier(label="comparator")
        circuit.append(TrainingState.qcmp(n, N + 1), qargs=(N_alpha[:n+1]))
        circuit.append(TrainingState.qcmp(m, M + 1), qargs=(M_beta[:m+1])) #less than or equal (M+1)
        circuit.append(TrainingState.qcmp(n, N + 1), qargs=(N_beta[:n+1]))
        circuit.barrier()

        #fix flag behavior
        circuit.x(M_beta[-2])
        circuit.x(N_beta[-2])
        circuit.x(N_alpha[-2])

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
        
        circuit = QuantumCircuit(N_alpha, vector_encoding_alpha, M_beta, N_beta, vector_encoding_beta)

        circuit.compose(TrainingState.prepare_alpha_beta_0(M), inplace=True)

        # apply rotations to encode vector state
        circuit.compose(TrainingState.u2_beta(M, feature_vec[N:]), qubits=M_beta[:] + N_beta[:] + vector_encoding_beta[:], inplace=True)

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
        
        circuit = QuantumCircuit(N_alpha, vector_encoding_alpha, M_beta, N_beta, vector_encoding_beta)

        circuit.compose(TrainingState.prepare_beta_1(M), inplace=True)

        # apply rotations to encode vector state
        circuit.append(TrainingState.u2_alpha(feature_vec[:N]), qargs=N_alpha[:] + vector_encoding_alpha[:])

        return circuit
