from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
#from qiskit.circuit.library import GroverOperator, Statevector
from qiskit_algorithms import IterativeAmplitudeEstimation, EstimationProblem
from ae_utils import *
from qiskit.primitives import Sampler
from ae_state_preparation import prep_state_ae
import numpy as np

class IterativeQuantumAmplitudeEstimation():
    
    @staticmethod
    def IQAE(M, N, feature_vec, epsilon, alpha):

        # Initialization Circuit
        statePrep = prep_state_ae(M, feature_vec, N)

        #Grover's Oracle
        oracle = Oracle(feature_vec, M, N)

        m = int(log2(M) + 1)


        # Problem for IQAE
        problem = EstimationProblem(
            state_preparation=statePrep,
            grover_operator=oracle,
            objective_qubits=m,
            
        )
        
        iqae = IterativeAmplitudeEstimation(
            epsilon_target=epsilon,
            alpha=alpha, 
            sampler=Sampler()
        )

        # Execute
        result = iqae.estimate(problem)

        # Results
        print(f"Estimated amplitude: {result.estimation}")
        return result
    
if __name__ == '__main__':
    # M = images
    M = 10
    v = np.array([x for x in range(880)])
    v = v / np.linalg.norm(v)
    N = 80
    epsilon = 0.02  # Precision
    alpha = 0.05    # Confidence Level
    shots = 100

    iqaeResult = IterativeQuantumAmplitudeEstimation.IQAE(M, N, v, epsilon, alpha)
    print(iqaeResult)