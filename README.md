## JumpReLU: A Retrofit Defense Strategy for Adversarial Attacks

This repository provides research code for the JumpReLU, a new activation function which helps to improve the model robustness. The idea is to introduces a slight jump discontinuity to improve the robustness of the model during the inference time, as illustrated below:

![JumpReLU](https://github.com/erichson/JumpReLU/blob/master/plots/jumprelu.png)


The jump size poses a trade-off between predictive accuracy and robustness, and can be trained during a validation stage. This means, the user needs to decide how much accuracy on clean data he is willing to sacrifice in order to gain more robustness. 

We consider the following attack methods to demonstrate the JumpReLU:

* Projected gradient descent (PGD) / iterative fast gradient sign method (IFGSM).
* Deep Fool (DF) with 2-norm and inf-norm.
* Trust Region (TR) method.

We run both gray-box and white-box attacks. For more details see the corresponding paper: xxx.
