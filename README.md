## JumpReLU: A Retrofit Defense Strategy for Adversarial Attacks

This repository provides research code for the JumpReLU, a new activation function which helps to improve the model robustness. For more details see the corresponding paper: xxx.

The idea is to introduces a slight jump discontinuity to improve the robustness of the model during the inference time, as illustrated below:

![JumpReLU](https://github.com/erichson/JumpReLU/blob/master/plots/jumprelu.png)


The jump size poses a trade-off between predictive accuracy and robustness, and can be trained during a validation stage. This means, the user needs to decide how much accuracy on clean data he is willing to sacrifice in order to gain more robustness. 

We consider the following attack methods to demonstrate the JumpReLU:

* Projected gradient descent (PGD) / iterative fast gradient sign method (IFGSM): [https://arxiv.org/abs/1706.06083](https://arxiv.org/abs/1706.06083).
* Deep Fool (DF) with 2-norm and inf-norm: [https://arxiv.org/abs/1511.04599](https://arxiv.org/abs/1511.04599).
* Trust Region (TR) method: [https://arxiv.org/abs/1812.06371](https://arxiv.org/abs/1812.06371).

We run both gray-box and white-box attacks for the MNIST and CIFAR10 dataset, considering several different network architectures such as:

* LeNet5 like architecture.
* AlexLike architecture.
* Wide ResNet (30-4) architecture [https://arxiv.org/abs/1605.07146](https://arxiv.org/abs/1605.07146).
* MobileNetV2 architecture: [https://arxiv.org/abs/1801.04381](https://arxiv.org/abs/1801.04381).



