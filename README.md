# Jump ReLU

## MNIST

### train and attack NetW
```export CUDA_VISIBLE_DEVICES=0; python main_baseline.py --name mnist --epochs 90 --arch NetW --lr 0.01 --lr-decay 0.2 --lr-schedule normal --lr-decay-epoch 30 60 2>&1 | tee mnist_result/net.log```
```export CUDA_VISIBLE_DEVICES=0; python adv_accuracy_mnist.py --test-batch-size 1000 --norm 1 --eps 0.01 --arch NetW --resume mnist_result/NetWbaseline1.pkl --data-set test --iter 100```

### train and attack NetW with Jump ReLU
```export CUDA_VISIBLE_DEVICES=0; python main_baseline.py --name mnist --epochs 90 --arch JumpNetW --lr 0.01 --lr-decay 0.2 --lr-schedule normal --lr-decay-epoch 30 60 2>&1 | tee mnist_result/net.log```
```export CUDA_VISIBLE_DEVICES=0; python adv_accuracy_mnist.py --test-batch-size 1000 --norm 1 --eps 0.01 --arch JumpNetW --resume mnist_result/JumpNetWbaseline1.pkl --data-set test --iter 100```


