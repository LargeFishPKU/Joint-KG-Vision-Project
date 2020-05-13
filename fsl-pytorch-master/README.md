# Few Shot Learning Pytorch

## Environment

 - Python3
 - Pytorch 0.4

## Getting started
All parameters are recorded in yaml files in fold cfgs, refer to the following:

- dataset/method/model

    | Setting | Options |
    | -- | -- |
    | dataset |**Omniglot**，CUB，**mini-ImageNet**， EMNIST(only for domain adaption) |
    | method | baseline，baseline++， **protonet**， **matchingnet**， **relationnet**(relationnet_softmax)， **maml**(maml_approx) |
    | model | **Conv4**， Conv4S，Conv4SNP， Conv4NP， Conv6， Conv6NP， ResNet10， ResNet18， ResNet34， ResNet50， ResNet101 |


- epoch

    | Method \ Dataset | omniglot | CUB | miniImagenet |
    | -- | -- | -- | -- |
    | baseline，baseline++ | 5 | 200 | 400 |
    | maml，matchingNet，protoNet，relationNet | 600(1 shot), 400(5 shot) | 600(1 shot), 400(5 shot) | 600(1 shot), 400(5 shot) |


- optim: Adam
- base_lr: 0.001
- weight decay: 0
- gamma: 1(fix learning rate)
- 5-way 5-shot
- train_aug: True


## Train
partition: the number of GPU (0, ..., n if you have n GPUS)
config: .yaml file

```
   sh scripts/train.sh [partition] [config]

   like：sh scripts/train.sh 0 cfgs/CUB/baseline.yaml

```

## Test
```
   sh scripts/test.sh [partition] [config]

   示例：sh scripts/test.sh 0 cfgs/CUB/baseline.yaml

```
