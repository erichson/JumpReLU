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

> export CUDA_VISIBLE_DEVICES=0; python attack_WhiteBox_cw.py --eps 0.01 --arch LeNetLike --resume mnist_result/LeNetLike_baseline.pkl  --iter 1 --iter_df 1 --runs 1 --jump 0.0 1.0


### Robust Black-Box Attack
> export CUDA_VISIBLE_DEVICES=0; python attack_BlackBox.py --eps 0.01 --arch LeNetLike --resume mnist_result/LeNetLike_robust.pkl  --iter 40 --iter_df 40 --runs 1 --jump 0.0 0.5 1.0 1.5







# CIFAR10: AlexNetLike 

## Baseline Training
> export CUDA_VISIBLE_DEVICES=0; python train_baseline.py --name cifar10 --epochs 90 --arch AlexLike --lr 0.02 --lr-decay 0.2 --lr-schedule normal --lr-decay-epoch 30 60 80

### White-Box Attack
> export CUDA_VISIBLE_DEVICES=0; python attack_WhiteBox.py --eps 0.01 --test-batch-size 500 --arch AlexLike --resume cifar10_result/AlexLike_baseline.pkl --dataset cifar10 --iter 7 --iter_df 7 --runs 1 --jump  0.0 0.2 0.4 0.6 0.8

### Black-Box Attack
> export CUDA_VISIBLE_DEVICES=0; python attack_BlackBox.py --eps 0.01 --test-batch-size 500 --arch AlexLike --resume cifar10_result/AlexLike_baseline.pkl --dataset cifar10 --iter 7 --iter_df 7 --runs 1 --jump 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8


## Robust Training
> export CUDA_VISIBLE_DEVICES=0; python train_robust.py --name cifar10 --epochs 120 --arch AlexLike --lr 0.02 --lr-decay 0.2 --lr-decay-epoch 30 60 90  --batch-size 128 --test-batch-size 200 --weight-decay 5e-4 --adv_ratio 0.6 --eps 0.031


### White-Box Attack
> export CUDA_VISIBLE_DEVICES=0; python attack_WhiteBox.py --eps 0.01 --test-batch-size 500 --arch AlexLike --resume cifar10_result/AlexLike_robust.pkl --dataset cifar10 --iter 7 --iter_df 7 --runs 1 --jump  0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8

### Black-Box Attack
> export CUDA_VISIBLE_DEVICES=0; python attack_BlackBox.py --eps 0.01 --arch AlexLike --resume cifar10_result/AlexLike_robust.pkl  --dataset cifar10 --test-batch-size 500 --iter 7 --iter_df 7 --runs 1 --jump 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8





# CIFAR10: ResNet

## Baseline Training
> export CUDA_VISIBLE_DEVICES=0; python train.py --name cifar10 --epochs 120 --arch ResNet --lr 0.1 --lr-decay 0.1 --lr-schedule normal --lr-decay-epoch 30 60 90 --batch-size 128 --test-batch-size 200 --weight-decay 5e-4


### White-Box Attack
> export CUDA_VISIBLE_DEVICES=0; python attack_WhiteBox.py --eps 0.01 --test-batch-size 500 --arch ResNet --resume cifar10_result/ResNet_baseline.pkl --dataset cifar10 --iter 7 --iter_df 7 --runs 1 --jump   0.0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09

### Black-Box Attack
> export CUDA_VISIBLE_DEVICES=0; python attack_BlackBox.py --eps 0.031 --test-batch-size 500 --arch ResNet --resume cifar10_result/ResNet_baseline.pkl --dataset cifar10 --iter 7 --iter_df 7 --runs 1 --jump  0.0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09


## Robust Training
> export CUDA_VISIBLE_DEVICES=0; python train.py --name cifar10 --epochs 120 --arch ResNet --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 30 60 90  --batch-size 128 --test-batch-size 200 --weight-decay 5e-4 --adv_ratio 0.6 --eps 0.031


### White-Box Attack
> export CUDA_VISIBLE_DEVICES=0; python attack_WhiteBox.py --eps 0.01 --test-batch-size 500 --arch ResNet --resume cifar10_result/ResNet_robust.pkl --dataset cifar10 --iter 7 --iter_df 7 --runs 1 --jump   0.0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09

### Black-Box Attack
> export CUDA_VISIBLE_DEVICES=0; python attack_BlackBox.py --eps 0.01 --arch JumpResNet --resume cifar10_result/ResNet_robust.pkl  --dataset cifar10 --test-batch-size 500 --iter 7 --iter_df 7 --runs 1 --jump  0.0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09





# CIFAR10: WideResNet

## Baseline Training
> export CUDA_VISIBLE_DEVICES=0; python train_baseline.py --name cifar10 --epochs 120 --arch WideResNet --lr 0.1 --lr-decay 0.1 --lr-schedule normal --lr-decay-epoch 30 60 90 --batch-size 128 --test-batch-size 200 --weight-decay 5e-4 --depth 34 --widen_factor 4


### White-Box Attack
> export CUDA_VISIBLE_DEVICES=0; python attack_WhiteBox.py --eps 0.01 --test-batch-size 500 --arch WideResNet --resume cifar10_result/WideResNet_baseline.pkl --dataset cifar10 --iter 7 --iter_df 7 --runs 1 --depth 34 --widen_factor 4 --jump 0.0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 

### Black-Box Attack
> export CUDA_VISIBLE_DEVICES=0; python attack_BlackBox.py --eps 0.031 --test-batch-size 500 --arch WideResNet --resume cifar10_result/WideResNet_baseline.pkl --dataset cifar10 --iter 7 --iter_df 7 --runs 1 --depth 34 --widen_factor 4 --jump 0.0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09


## Robust Training
> export CUDA_VISIBLE_DEVICES=0; python train_robust.py --name cifar10 --epochs 120 --arch WideResNet --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 30 60 90  --batch-size 128 --test-batch-size 200 --weight-decay 5e-4 --adv_ratio 0.6 --eps 0.031 --depth 34 --widen_factor 4


### White-Box Attack
> export CUDA_VISIBLE_DEVICES=0; python attack_WhiteBox.py --eps 0.01 --test-batch-size 500 --arch WideResNet --resume cifar10_result/WideResNet_robust.pkl --dataset cifar10 --iter 7 --iter_df 7 --runs 1 --depth 34 --widen_factor 4 --jump 0.0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09

### Black-Box Attack
> export CUDA_VISIBLE_DEVICES=0; python attack_BlackBox.py --eps 0.01 --arch WideResNet --resume cifar10_result/WideResNet_robust.pkl  --dataset cifar10 --test-batch-size 500 --iter 7 --iter_df 7 --runs 1 --depth 34 --widen_factor 4 --jump 0.0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09







# CIFAR10: WideResNetThermo

## Baseline Training
> export CUDA_VISIBLE_DEVICES=0; python train_thermo.py --name cifar10 --epochs 90 --arch WideResNetThermo --lr 0.1 --lr-decay 0.2 --lr-schedule normal --lr-decay-epoch 30 60 80 --batch-size 128 --test-batch-size 200 --weight-decay 5e-4 --depth 16 --widen_factor 4 --level 16


### White-Box Attack
> export CUDA_VISIBLE_DEVICES=0; python attack_WhiteBox.py --eps 0.01 --test-batch-size 500 --arch WideResNet --resume cifar10_result/WideResNet_baseline.pkl --dataset cifar10 --iter 7 --iter_df 7 --runs 1 --depth 34 --widen_factor 4 --level 16 --jump   0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8

### Black-Box Attack
> export CUDA_VISIBLE_DEVICES=0; python attack_BlackBox.py --eps 0.031 --test-batch-size 500 --arch WideResNet --resume cifar10_result/WideResNet_baseline.pkl --dataset cifar10 --iter 7 --iter_df 7 --runs 1 --depth 34 --widen_factor 4 --level 16 --jump  0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8


## Robust Training
> export CUDA_VISIBLE_DEVICES=0; python train_thermo_robust.py --name cifar10 --epochs 120 --arch WideResNetThermo --lr 0.1 --lr-decay 0.2 --lr-decay-epoch 30 60 90  --batch-size 128 --test-batch-size 200 --weight-decay 5e-4 --adv_ratio 0.6 --eps 0.031 --depth 16 --widen_factor 4 --level 16


### White-Box Attack
> export CUDA_VISIBLE_DEVICES=0; python attack_WhiteBox.py --eps 0.01 --test-batch-size 500 --arch JumpResNet --resume cifar10_result/WideResNet_robust.pkl --dataset cifar10 --iter 7 --iter_df 7 --runs 1 --depth 34 --widen_factor 4 --level 16 --jump 0.0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09

### Black-Box Attack
> export CUDA_VISIBLE_DEVICES=0; python attack_BlackBox.py --eps 0.01 --arch JumpResNet --resume cifar10_result/WideResNet_robust.pkl  --dataset cifar10 --test-batch-size 500 --iter 7 --iter_df 7 --runs 1 --depth 34 --widen_factor 4 --level 16 --jump 0.0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09



# CIFAR10: MobileNetV2

## Baseline Training
> export CUDA_VISIBLE_DEVICES=0; python train_baseline.py --name cifar10 --epochs 120 --arch MobileNetV2 --lr 0.1 --lr-decay 0.1 --lr-schedule normal --lr-decay-epoch 30 60 90 --batch-size 128 --test-batch-size 200 --weight-decay 5e-4


### White-Box Attack
> export CUDA_VISIBLE_DEVICES=0; python attack_WhiteBox.py --eps 0.01 --test-batch-size 400 --arch MobileNetV2 --resume cifar10_result/MobileNetV2_baseline.pkl --dataset cifar10 --iter 7 --iter_df 7 --runs 1 --jump  0.0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08

### Black-Box Attack
> export CUDA_VISIBLE_DEVICES=0; python attack_BlackBox.py --eps 0.031 --test-batch-size 400 --arch MobileNetV2 --resume cifar10_result/MobileNetV2_baseline.pkl --dataset cifar10 --iter 7 --iter_df 7 --runs 1 --jump 0.0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08


## Robust Training
> export CUDA_VISIBLE_DEVICES=0; python train_robust.py --name cifar10 --epochs 120 --arch MobileNetV2 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 30 60 90  --batch-size 128 --test-batch-size 200 --weight-decay 5e-4 --adv_ratio 0.6 --eps 0.031


### White-Box Attack
> export CUDA_VISIBLE_DEVICES=0; python attack_WhiteBox.py --eps 0.01 --test-batch-size 400 --arch MobileNetV2 --resume cifar10_result/MobileNetV2_robust.pkl --dataset cifar10 --iter 7 --iter_df 7 --runs 1 --jump  0.0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08

### Black-Box Attack
> export CUDA_VISIBLE_DEVICES=0; python attack_BlackBox.py --eps 0.01 --arch MobileNetV2 --resume cifar10_result/MobileNetV2_robust.pkl  --dataset cifar10 --test-batch-size 400 --iter 7 --iter_df 7 --runs 1 --jump 0.0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08




export CUDA_VISIBLE_DEVICES=0; python attack_WhiteBox.py --eps 0.01 --test-batch-size 400 --arch MobileNetV2 --resume cifar10_result/MobileNetV2_baseline.pkl --dataset cifar10 --iter 7 --iter_df 7 --runs 1 --jump  0.06



