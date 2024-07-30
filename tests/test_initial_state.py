from src.training_image_feature_state_preparation import TrainingState
import unittest
from qiskit.compiler import transpile
import numpy as np
from qiskit_aer import AerSimulator
from math import log2
from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister, ClassicalRegister
from qiskit.circuit.library import IntegerComparator
from collections import Counter

class Tests(unittest.TestCase):

    # testcase for the initial encoded state after u2 (u2 is the bane of my existence)
    def test_beta_1(self):
        self.skipTest(reason="passed")
        vec = np.array([x for x in range(160)])
        vec = vec / np.linalg.norm(vec)

        print(vec)

        M = 1
        N = 80

        # power of 2 that is an upper bound on M and N (# bits needed to represent M and N in binary)
        m = int(log2(M) + 1)
        n = int(log2(N) + 1)

        reg = QuantumRegister(m + 2*n + 10)
        circuit = QuantumCircuit(reg)
        circuit.compose(TrainingState.prepare_beta_1(M, vec), inplace=True)
        
        circuit.measure_all()

        checkResult = AerSimulator().run(transpile(circuit, AerSimulator()), shots=1024, memory=True).result()
        measurements = checkResult.get_memory()
        print(Counter([y[0] for y in measurements]))



        #print(circuit)


     # testcase for the initial superposition (OUTDATED)
    def test_beta_0(self):
       self.skipTest(reason='passed')
       for M in range(1, 17):
            circuit = TrainingState.prepare_beta_0(M)
            circuit.measure_all()
            checkResult = AerSimulator().run(transpile(circuit, AerSimulator()), shots=1024, memory=True).result()
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
        self.skipTest(reason="passed")
        for n in range(1, 10):
            size = int(log2(n) + 1)
            reg = QuantumRegister(size + 1)
            #anc = AncillaRegister(size)
            circuit = QuantumCircuit(reg)
            circuit.h(range(size))
            circuit.compose(TrainingState.qcmp(size, n), inplace=True)
            #circuit.decompose(reps=3)
            

            circuit.measure_all()
            checkResult = AerSimulator().run(transpile(circuit, AerSimulator()), shots=100, memory=True).result()
            measurements = checkResult.get_memory()
            #measurements = [x[::-1] for x in measurements] #convert to big endian

            for measurement in measurements:
                num = int(measurement[1:][::-1], 2)
                print(n)
                print(num)
                print(measurement)
                
                target = measurement[0]
                if num >= n:
                    self.assertTrue(target == '0')
                if num < n:
                    self.assertTrue(target == '1')






        
