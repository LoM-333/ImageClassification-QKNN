import unittest
import numpy as np
from qiskit_aer import AerSimulator
from src.minimum_distances import initialize_quantum_state_with_distances, oracle_min_distance, diffuser, find_k_minimum_distances

class TestMinimumDistances(unittest.TestCase):

    def test_initialize_quantum_state_with_distances(self):
        distance_list = [0.1, 0.2, 0.3]
        qc = initialize_quantum_state_with_distances(distance_list)
        
        self.assertEqual(len(qc.qubits), 2 * len(distance_list) - 1)  # Distance qubits + ancilla qubits
        self.assertTrue(any(op.name == 'ry' for op in qc.data))

    def test_oracle_min_distance(self):
        num_qubits = 3
        oracle = oracle_min_distance(num_qubits)
        
        self.assertEqual(len(oracle.qubits), 2 * num_qubits)  # Control qubits + target qubit + ancilla qubits
        self.assertTrue(any(op.name == 'mcx' for op in oracle.data))

    def test_diffuser(self):
        num_qubits = 3
        diffusion_operator = diffuser(num_qubits)
        
        self.assertEqual(len(diffusion_operator.qubits), 2 * num_qubits - 1)  # Diffuser with ancilla qubits
        self.assertTrue(any(op.name == 'h' for op in diffusion_operator.data))
        self.assertTrue(any(op.name == 'x' for op in diffusion_operator.data))
        self.assertTrue(any(op.name == 'mcx' for op in diffusion_operator.data))

    def test_find_k_minimum_distances(self):
        # Generate a random distance list for testing
        np.random.seed(42)
        distance_list = np.random.rand(8)

        # Test finding the 3 minimum distances
        k = 3
        k_min_indexes = find_k_minimum_distances(distance_list, k)

        # Ensure the returned value is a list of integers with length k
        self.assertEqual(len(k_min_indexes), k)
        self.assertTrue(all(isinstance(i, int) for i in k_min_indexes))

        # Check if indexes are within the range of the distance_list
        self.assertTrue(all(0 <= i < len(distance_list) for i in k_min_indexes))

        # Check if the returned indexes correspond to the smallest distances
        sorted_distances = np.argsort(distance_list)
        expected_indexes = sorted_distances[:k].tolist()
        self.assertEqual(set(k_min_indexes), set(expected_indexes))

if __name__ == '__main__':
    unittest.main()


#will run and pass with (python -m unittest discover -s src -p "test_minimum_distances.py") 
# but not with (python -m unittest src/minimum_distances.py)
#NO CLUE WHY