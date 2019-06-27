# RobustTransformLearning

This work introduces robust transform learning. 

solves ||TX - Z||_1 - mu*logdet(T) + eps*mu||T||_Fro + tau||Z||_1


 Inputs
 X          - Training Data
 numOfAtoms - dimensionaity after Transform
 mu         - regularizer for Tranform
 lambda     - regularizer for coefficient

 Output
 T          - learnt Transform
 Z          - learnt sparse coefficients


Related Publication:

J. Maggu and A. Majumdar, "Robust transform learning," 2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), New Orleans, LA, 2017, pp. 1467-1471.

Link: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7952400&isnumber=7951776

Run classifyRTL.m for experiments. The demo is given for MNIST_basic dataset. 
