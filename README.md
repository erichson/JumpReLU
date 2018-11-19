# Jump ReLU

## MNIST

### NetW with JumpReLU: train and attack 
> export CUDA_VISIBLE_DEVICES=0; python main_baseline.py --name mnist --epochs 90 --arch JumpNet --lr 0.01 --lr-decay 0.2 --lr-schedule normal --lr-decay-epoch 30 60 2>&1 | tee mnist_result/net.log

> export CUDA_VISIBLE_DEVICES=0; python adv_accuracy_mnist.py --test-batch-size 1000 --eps 0.01 --arch JumpNet --resume mnist_result/JumpNetbaseline1.pkl  --iter 100



## EMNIST

### NetW with JumpReLU: train and attack 
> export CUDA_VISIBLE_DEVICES=0; python main_baseline.py --name emnist --epochs 120 --arch JumpNet_EMNIST --lr 0.02 --lr-decay 0.2 --lr-schedule normal --lr-decay-epoch 30 60 2>&1 | tee mnist_result/net.log

> export CUDA_VISIBLE_DEVICES=0; python adv_accuracy_mnist.py --test-batch-size 1000 --eps 0.01 --arch JumpNet_EMNIST --resume emnist_result/JumpNet_EMNISTbaseline1.pkl --dataset emnist --iter 1


