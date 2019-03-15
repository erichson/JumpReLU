# Jump ReLU

# MNIST: LeNetLike

## Baseline Training
> export CUDA_VISIBLE_DEVICES=0; python train_baseline.py --name mnist --epochs 90 --arch LeNetLike --lr 0.02 --lr-decay 0.2 --lr-decay-epoch 30 60 80 --weight-decay 5e-4


### White-Box Attack
> export CUDA_VISIBLE_DEVICES=0; python attack_WhiteBox.py --eps 0.01 --arch LeNetLike --resume mnist_result/LeNetLike_baseline.pkl  --iter 40 --iter_df 40 --runs 1 --jump 0.0 0.5 1.0 1.5


### Black-Box Attack
> export CUDA_VISIBLE_DEVICES=0; python attack_BlackBox.py --eps 0.01 --arch LeNetLike --resume mnist_result/LeNetLike_baseline.pkl  --iter 40 --iter_df 40 --runs 1 --jump 0.0 1.0



## Robust Training
> export CUDA_VISIBLE_DEVICES=0; python train_robust.py --name mnist --epochs 90 --arch LeNetLike --lr 0.02 --lr-decay 0.2 --lr-decay-epoch 30 60 80  --adv_ratio 0.3 --eps 0.3


### Robust White-Box Attack
> export CUDA_VISIBLE_DEVICES=0; python attack_WhiteBox.py --eps 0.01 --arch LeNetLike --resume mnist_result/LeNetLike_robust.pkl  --iter 40 --iter_df 40 --runs 1 --jump 0.0 0.5 1.0 1.5


### Robust Black-Box Attack
> export CUDA_VISIBLE_DEVICES=0; python attack_BlackBox.py --eps 0.01 --arch LeNetLike --resume mnist_result/LeNetLike_robust.pkl  --iter 40 --iter_df 40 --runs 1 --jump 0.0 0.5 1.0 1.5







# CIFAR10: AlexNetLike 

## Baseline Training
> export CUDA_VISIBLE_DEVICES=0; python train_baseline.py --name cifar10 --epochs 120 --arch JumpNet_CIFAR --lr 0.02 --lr-decay 0.2 --lr-schedule normal --lr-decay-epoch 30 60 90


> export CUDA_VISIBLE_DEVICES=0; python attack_simulation.py --eps 0.01 --test-batch-size 500 --arch JumpNet_CIFAR --resume cifar10_result/JumpNet_CIFARbaseline1.pkl --dataset cifar10 --iter 100 --iter_df 200 --runs 10 --jump  0.0 0.05 0.1 0.15 0.2


> export CUDA_VISIBLE_DEVICES=0; python attack_simulation_black.py --eps 0.01 --test-batch-size 500 --arch JumpNet_CIFAR --resume cifar10_result/JumpNet_CIFARbaseline1.pkl --dataset cifar10 --iter 100 --iter_df 100 --runs 1 --jump 0.0 0.05 0.1





# CIFAR10: ResNet

## Baseline Training
> export CUDA_VISIBLE_DEVICES=0; python train_baseline.py --name cifar10 --epochs 120 --arch JumpResNet --lr 0.1 --lr-decay 0.2 --lr-schedule normal --lr-decay-epoch 30 60 90 --batch-size 128 --test-batch-size 200 --weight-decay 5e-4


### White-Box Attack
> export CUDA_VISIBLE_DEVICES=0; python attack_WhiteBox.py --eps 0.01 --test-batch-size 500 --arch JumpResNet --resume cifar10_result/JumpResNetbaseline.pkl --dataset cifar10 --iter 7 --iter_df 7 --runs 1 --jump  0.0 0.05 0.1 0.15 0.2

### Black-Box Attack
> export CUDA_VISIBLE_DEVICES=0; python attack_BlackBox.py --eps 0.031 --test-batch-size 500 --arch JumpResNet --resume cifar10_result/JumpResNetbaseline.pkl --dataset cifar10 --iter 7 --iter_df 7 --runs 1 --jump 0.0 0.05 0.06 0.07


## Robust Training
> export CUDA_VISIBLE_DEVICES=0; python train_robust.py --name cifar10 --epochs 120 --arch JumpResNet --lr 0.1 --lr-decay 0.2 --lr-decay-epoch 30 60 90  --batch-size 128 --test-batch-size 200 --weight-decay 5e-4 --adv_ratio 0.6 --eps 0.031


### White-Box Attack
> export CUDA_VISIBLE_DEVICES=0; python attack_WhiteBox.py --eps 0.01 --test-batch-size 500 --arch JumpResNet --resume cifar10_result/JumpResNet_robust.pkl --dataset cifar10 --iter 7 --iter_df 7 --runs 1 --jump  0.0 0.05 0.06 0.07

### Black-Box Attack
> export CUDA_VISIBLE_DEVICES=0; python attack_BlackBox.py --eps 0.01 --arch JumpResNet --resume cifar10_result/JumpResNet_robust.pkl  --dataset cifar10 --test-batch-size 500 --iter 7 --iter_df 7 --runs 1 --jump 0.0 0.05 0.06 0.07








# CIFAR10: MobileNetV2

## Baseline Training
> export CUDA_VISIBLE_DEVICES=0; python train_baseline.py --name cifar10 --epochs 120 --arch MobileNetV2 --lr 0.1 --lr-decay 0.2 --lr-schedule normal --lr-decay-epoch 30 60 90 --batch-size 128 --test-batch-size 200 --weight-decay 5e-4


### White-Box Attack
> export CUDA_VISIBLE_DEVICES=0; python attack_WhiteBox.py --eps 0.01 --test-batch-size 500 --arch MobileNetV2 --resume cifar10_result/MobileNetV2_baseline.pkl --dataset cifar10 --iter 7 --iter_df 7 --runs 1 --jump  0.0 0.05 0.1 0.15 0.2

### Black-Box Attack
> export CUDA_VISIBLE_DEVICES=0; python attack_BlackBox.py --eps 0.031 --test-batch-size 500 --arch MobileNetV2 --resume cifar10_result/MobileNetV2_baseline.pkl --dataset cifar10 --iter 7 --iter_df 7 --runs 1 --jump 0.0 0.05 0.06 0.07


## Robust Training
> export CUDA_VISIBLE_DEVICES=0; python train_robust.py --name cifar10 --epochs 120 --arch MobileNetV2 --lr 0.1 --lr-decay 0.2 --lr-decay-epoch 30 60 90  --batch-size 128 --test-batch-size 200 --weight-decay 5e-4 --adv_ratio 0.6 --eps 0.031


### White-Box Attack
> export CUDA_VISIBLE_DEVICES=0; python attack_WhiteBox.py --eps 0.01 --test-batch-size 500 --arch MobileNetV2 --resume cifar10_result/MobileNetV2_robust.pkl --dataset cifar10 --iter 7 --iter_df 7 --runs 1 --jump  0.0 0.05 0.06 0.07

### Black-Box Attack
> export CUDA_VISIBLE_DEVICES=0; python attack_BlackBox.py --eps 0.01 --arch JumpResNet --resume cifar10_result/MobileNetV2_robust.pkl  --dataset cifar10 --test-batch-size 500 --iter 7 --iter_df 7 --runs 1 --jump 0.0 0.05 0.06 0.07


