from src.training_image_feature_state_preparation import TrainingState
import unittest
from qiskit.compiler import transpile
import numpy as np
from qiskit_aer import AerSimulator
from math import log2
from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.circuit.library import IntegerComparator

class Tests(unittest.TestCase):

    # testcase for the initial encoded state after u2 (u2 is the bane of my existence)
    def test_beta_1(self):
        # M=1, N=80
        test_vec = [i for i in range(1, 81)]
        test_vec = np.array(test_vec)
        norm = np.linalg.norm(test_vec)
        test_vec = test_vec / norm
        N=80
        m = int(log2(1) + 1)
        n = int(log2(N) + 1)

        circuit = TrainingState.prepare_beta_1(1, test_vec)

        circuit.measure_all()

        checkResult = AerSimulator().run(transpile(circuit, AerSimulator()), shots=1024, memory=True).result()
        measurements = checkResult.get_memory()
        print(measurements)
        desired = []
        for measurement in measurements():
            measured_M = measurement[m+2*n+4:]
            measured_N = measurement[m+n+2:m+2*n+2] #account for flags
            M_flags = measurement[m+2*n+2:m+2*n+4]
            N_flags = measurement[m+n:m+n+2]
            

        #print(circuit)


     # testcase for the initial superposition
    def test_beta_0(self):
       for M in range(1, 17):
            circuit = TrainingState.prepare_beta_0(M)
            circuit.measure_all()
            checkResult = AerSimulator().run(transpile(circuit, AerSimulator()), shots=25, memory=True).result()
            measurements = checkResult.get_memory()
            #measurements = [x[::-1] for x in measurements] #convert to big endian
            N=80
            m = int(log2(M) + 1)
            n = int(log2(N) + 1)
            for measurement in measurements:
                
                measured_M = measurement[m+2*n+4:]
                measured_N = measurement[m+n+2:m+2*n+2] #account for flags
                M_flags = measurement[m+2*n+2:m+2*n+4]
                N_flags = measurement[m+n:m+n+2]
                #print(measurement)
                #print(measured_M)
                #print(measured_N)
                #print(M_flags)
                #print(N_flags)
                if int(measured_M, 2) <= M:
                    self.assertTrue(M_flags[1] == '0')
                if int(measured_M, 2) > M:
                    self.assertTrue(M_flags[1] == '1')
                if int(measured_N, 2) <= N:
                    self.assertTrue(N_flags[1] == '0')
                if int(measured_N, 2) > N:
                    self.assertTrue(N_flags[1] == '1')
                if int(measured_M, 2) == 0:
                    self.assertTrue(M_flags[0] == '1')
                if int(measured_M, 2) != 0:
                    self.assertTrue(M_flags[0] == '0')
                if int(measured_N, 2) == 0:
                    self.assertTrue(N_flags[0] == '1')
                if int(measured_N, 2) != 0:
                    self.assertTrue(N_flags[0] == '0')

        
    def test_comparator(self):
        for n in range(1, 65):
            size = int(log2(n) + 1)
            reg = QuantumRegister(size + 1)
            anc = AncillaRegister(size)
            circuit = QuantumCircuit(reg, anc)
            circuit.h(range(size))
            circuit.compose(IntegerComparator(size, n), inplace=True)
            #circuit.decompose(reps=3)
            

            circuit.measure_all()
            checkResult = AerSimulator().run(transpile(circuit, AerSimulator()), shots=1, memory=True).result()
            measurements = checkResult.get_memory()
            #measurements = [x[::-1] for x in measurements] #convert to big endian

            for measurement in measurements:
                #print(n)
                #print(measurement)
                num = int(measurement[size:], 2)
                target = measurement[size]
                if num >= n:
                    self.assertTrue(target == '1')
                if num < n:
                    self.assertTrue(target == '0')






        
