from qiskit import QuantumCircuit, Aer
from qiskit.circuit.library import GroverOperator, Statevector
from qiskit.algorithms import IterativeAmplitudeEstimation, EstimationProblem
from qiskit.utils import QuantumInstance

from src.ae_state_preparation import prep_state_ae
import numpy as np

class IterativeQuantumAmplitudeEstimation:
    
    @staticmethod
    def IQAE(M, N, feature_vec, numQubits, epsilon, alpha, shots=100):

        # Initialization Circuit
        statePrep = prep_state_ae(M, feature_vec, N)

        # Problem for IQAE
        problem = EstimationProblem(
            state_preparation=statePrep,
            objective_qubits=[numQubits - 1],
            post_processing=None
        )

        # Backend
        backend = Aer.get_backend('qasm_simulator')
        quantumInstance = QuantumInstance(backend, shots=shots)
        
        iqae = IterativeAmplitudeEstimation(
            epsilon=epsilon,
            alpha=alpha,
            quantum_instance=quantumInstance
        )

        # Execute
        result = iqae.estimate(problem)

        # Results
        print(f"Estimated amplitude: {result.estimation}")
        return result
    
if __name__ == '__main__':
    # M = images
    N = 80
    numQubits = 10
    epsilon = 0.01  # Precision
    alpha = 0.05    # Confidence Level
    shots = 100

    iqaeResult = IterativeQuantumAmplitudeEstimation.IQAE(numQubits, epsilon, alpha, shots)
    print(iqaeResult)