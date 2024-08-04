# QKNN (Quantum K-Nearest Neighbors) Image Classifier
<br>

## Abstract
This algorithm is a quantum analog to the K-Nearest-Neighbors classification paradigm frequently used in machine learning. Compared to other QML algorithms, ours has a smaller circuit depth and uses less qubits with respect to its input size, making it suitable for NISQ devices. However, this comes at the cost of lower accuracy than other QML methods. We leverage multiple quantum subroutines, i.e., quantum comparator, controlled swap test, amplitude estimation, qRAM, and Dürr's Search Algorithm. 

<br>

Our project consists of two parts. The first computes the distance between two vectors and the second finds the k nearest training vectors to a test vector, subsequently performing a classification. Our project was inspired by and designed for the 2024 MIT BWSI Quantum Software course. The primary technology used is IBM's Qiskit API.

## Goals
☐ Integrate amplitude estimation with Dürr's Search Algorithm
<br>
☐ Compile metrics across a wide range of classification tasks and input sizes
<br>
☐ Perform resource estimation using Qiskit's Aer Simulator and a classical KNN (sklearn)
<br>
☐ Run our model with a small sample size on a real quantum backend (tentative)

## Credits
We adapted the methods of scientific literature and other resources in the creation of this project.

<br>

<b>Brassard, G.</b>, <b>Hoyer, P.</b>, et al. Quantum Amplitude Amplification and Estimation, 2000, https://doi.org/10.48550/ARXIV.QUANT-PH/0005055.
<br>
<b>Dang, Y.</b>, <b>Jiang, N.</b>, et al. Image classification based on quantum K-Nearest-Neighbor algorithm, 2018, https://doi.org/10.1007/s11128-018-2004-9.
<br>
<b>Dürr, C.</b>, <b>Hoyer, P.</b> A Quantum Algorithm for Finding the Minimum, 1999, https://doi.org/10.48550/arXiv.quant-ph/9607014.
<br>
<b>Feng, C.</b>, <b>Zhao, B.</b>, et al. An Enhanced Quantum K-Nearest Neighbor Classification Algorithm Based on Polar Distance. Entropy 2023, 25, 127.
     https://doi.org/10.3390/e25010127
<br>
<b>Javadi-Abhari, A.</b>, <b>Treinish, M.</b>, et al. Quantum computing with Qiskit, 2024, arxiv.org/abs/2405.08810.
<br>
<b>Phalak, K.</b>, <b>Chatterjee, A.</b>, <b>Ghosh, S</b>. Quantum Random Access Memory for Dummies. Sensors 2023, 23, 7462. https://doi.org/10.3390/s23177462
<br>
<b>Quantum Amplitude Estimation.</b> Qiskit, qiskit-community.github.io/qiskit-finance/tutorials/00_amplitude_estimation.html.
<br>
<b>Yuan, Y.</b>, <b>Chao, W.</b>, et al. "An Improved QFT-based Quantum Comparator and Extended Modular Arithmetic Using One Ancilla Qubit." New Journal of 
     Physics, vol. 25, no. 10, 1 Oct. 2023, p. 103011, https://doi.org/10.1088/1367-2630/acfd52.
<br>
<b>Y. Kang</b> and <b>J. Heo</b>, "Quantum Minimum Searching Algorithm and Circuit Implementation," 2020 International Conference on Information and Communication Technology Convergence (ICTC), Jeju, Korea (South), 2020, pp. 214-219, doi: 10.1109/ICTC49870.2020.9289455.


