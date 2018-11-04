# Jump ReLU

## MNIST

### NetW: train and attack 
> export CUDA_VISIBLE_DEVICES=0; python main_baseline.py --name mnist --epochs 90 --arch NetW --lr 0.01 --lr-decay 0.2 --lr-schedule normal --lr-decay-epoch 30 60 2>&1 | tee mnist_result/net.log

> export CUDA_VISIBLE_DEVICES=0; python adv_accuracy_mnist.py --test-batch-size 1000 --norm 1 --eps 0.01 --arch NetW --resume mnist_result/NetWbaseline1.pkl  --iter 100

### NetW with JumpReLU: train and attack 
> export CUDA_VISIBLE_DEVICES=0; python main_baseline.py --name mnist --epochs 90 --arch JumpNetW --lr 0.01 --lr-decay 0.2 --lr-schedule normal --lr-decay-epoch 30 60 2>&1 | tee mnist_result/net.log

> export CUDA_VISIBLE_DEVICES=0; python adv_accuracy_mnist.py --test-batch-size 1000 --norm 1 --eps 0.01 --arch JumpNetW --resume mnist_result/JumpNetWbaseline1.pkl  --iter 100



## EMNIST

### NetW: train and attack 
> export CUDA_VISIBLE_DEVICES=0; python main_baseline.py --name emnist --epochs 120 --arch NetW_EMNIST --lr 0.02 --lr-decay 0.2 --lr-schedule normal --lr-decay-epoch 30 60 2>&1 | tee mnist_result/net.log

> export CUDA_VISIBLE_DEVICES=0; python adv_accuracy_mnist.py --test-batch-size 1000 --norm 1 --eps 0.01 --arch NetW_EMNIST --resume emnist_result/NetW_EMNISTbaseline1.pkl --dataset emnist --iter 100

### NetW with JumpReLU: train and attack 
> export CUDA_VISIBLE_DEVICES=0; python main_baseline.py --name emnist --epochs 120 --arch JumpNetW_EMNIST --lr 0.02 --lr-decay 0.2 --lr-schedule normal --lr-decay-epoch 30 60 2>&1 | tee mnist_result/net.log

> export CUDA_VISIBLE_DEVICES=0; python adv_accuracy_mnist.py --test-batch-size 1000 --norm 1 --eps 0.01 --arch JumpNetW_EMNIST --resume emnist_result/JumpNetW_EMNISTbaseline1.pkl --dataset emnist --iter 1


