from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, Aer, execute
import numpy as np
from src import amplitude_estimation_algorithm
from amplitude_estimation_algorithm import amplitude_estimation
from src import compute_distances
from compute_distances import compute_qubit_gamma

def initialize_quantum_state_with_distances(distance_list):
    n = len(distance_list)
    distance_qubits = QuantumRegister(n, 'distance')  # Distance qubits
    ancilla_qubits = QuantumRegister(n - 1, 'ancilla')  # Ancilla qubits for multi-controlled gates
    qc = QuantumCircuit(distance_qubits, ancilla_qubits)
    
    # Initialize distance values into the quantum state
    for i, distance in enumerate(distance_list):
        qc.ry(2 * np.arccos(np.sqrt(distance)), distance_qubits[i])
    
    return qc

def oracle_min_distance(num_qubits):
    control_qubits = QuantumRegister(num_qubits, 'control')  # Control qubits
    target_qubit = QuantumRegister(1, 'target')  # Target qubit
    ancilla_qubits = QuantumRegister(num_qubits - 1, 'ancilla')  # Ancilla qubits
    qc = QuantumCircuit(control_qubits, target_qubit, ancilla_qubits)
    
    # Oracle to mark the state
    qc.mcx(list(range(num_qubits)), target_qubit[0], ancilla_qubits)  # Multi-controlled X gate using ancilla qubits
    qc.name = "Oracle"
    return qc

def diffuser(num_qubits):
    qc = QuantumCircuit(num_qubits)
    ancilla_qubits = QuantumRegister(num_qubits - 1, 'ancilla')  # Ancilla qubits
    qc.add_register(ancilla_qubits)
    
    # Apply Hadamard gates
    qc.h(range(num_qubits))
    
    # Apply X gates
    qc.x(range(num_qubits))
    
    # Apply multi-controlled X gate with ancilla qubits
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1, ancilla_qubits)
    qc.h(num_qubits - 1)
    
    # Apply X gates again
    qc.x(range(num_qubits))
    
    # Apply Hadamard gates again
    qc.h(range(num_qubits))
    
    qc.name = "Diffuser"
    return qc

def grover_iteration(qc, num_qubits, oracle, diffuser, iterations):
    for _ in range(iterations):
        qc.append(oracle, range(num_qubits))
        qc.append(diffuser, range(num_qubits))
    return qc

def find_k_minimum_distances(distance_list, k):
    num_qubits = len(distance_list)

    # Step 1: Initialize Quantum State with distance values
    qc = initialize_quantum_state_with_distances(distance_list)
    
    # Step 2: Amplitude Estimation
    qc = amplitude_estimation(qc, num_qubits)
    
    # Step 3a: Distance Comparison (Placeholder)
    gamma_circuit = compute_qubit_gamma(v0, M, V)  # Use the imported function
    qc = qc.compose(gamma_circuit, qubits=range(M+1))  # Integrate distance comparison circuit
    

    
    # Step 3b: Dürr's Algorithm - Use Grover's search to find k minimum distances
    for _ in range(int(np.sqrt(k * num_qubits))):
        oracle = oracle_min_distance(num_qubits)
        diffusion_operator = diffuser(num_qubits)
        qc = grover_iteration(qc, num_qubits, oracle, diffusion_operator, 1)

        # Measure to find the current minimum distance
        backend = Aer.get_backend('qasm_simulator')
        result = execute(qc, backend, shots=1).result()
        counts = result.get_counts(qc)
        measured_distances = [int(key, 2) for key in counts.keys()]
        current_min_distance = min(measured_distances)
    
    # Step 4: Measurement to get the indexes of the k minimum distances
    qc.measure_all()
    
    backend = Aer.get_backend('qasm_simulator')
    result = execute(qc, backend, shots=1024).result()
    counts = result.get_counts()
    
    # Get the k most frequent results which correspond to k minimum distances
    sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    k_min_indexes = [int(index, 2) for index, count in sorted_counts[:k]]
    
    return k_min_indexes 