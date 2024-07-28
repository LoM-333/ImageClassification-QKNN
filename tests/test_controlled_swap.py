import unittest
from qiskit import QuantumCircuit, transpile
from src.controlled_swap import controlled_register_swap
from qiskit_aer import AerSimulator

class TestControlledSwap(unittest.TestCase):

    #testing w/o h on control
    def test_general_register_swap(self):
        self.skipTest(reason='passed')
        for a in range(2, 10):
            for b in range(2, 10):
                circuit1 = QuantumCircuit(a + b + 1)
                circuit1.x(0)
                for x in range(1, a + 1):
                    circuit1.x(x)
                circuit1.compose(controlled_register_swap(a, b), inplace=True)
                circuit1.measure_all()
                checkResult = AerSimulator().run(transpile(circuit1, AerSimulator()), shots=1, memory=True).result()
                measurement = checkResult.get_memory()[0]
                print(measurement)
                for i in measurement[::-1][b + 1:]:
                    if not i == '1':
                        self.fail()
                
                circuit2 = QuantumCircuit(a + b + 1)
                circuit2.x(0)
                circuit2.x(range(a + 1, a + b + 1))
                circuit2.compose(controlled_register_swap(a, b), inplace=True)
                circuit2.measure_all()
                checkResult = AerSimulator().run(transpile(circuit2, AerSimulator()), shots=1, memory=True).result()
                measurement = checkResult.get_memory()[0]
                print(measurement)
                for i in measurement[::-1][1:b+1]:
                    if not i == '1':
                        self.fail()


if __name__ == '__main__':
    unittest.main()