Fast Autoaugment / RandAugnment
===

**Fast Autoaugment -- NeurlIPS (2019)**
[paper site](https://papers.nips.cc/paper_files/paper/2019/file/6add07cf50424b14fdf649da87843d01-Paper.pdf)
Author : Sungbin Lim, Ildoo Kim, Taesup Kim, Chiheon Kim, Sungwoong Kim


**RandAugment: Practical Automated Data Augmentation with a Reduced Search Space -- NeurlIPS (2020)**
[paper site](https://papers.nips.cc/paper_files/paper/2020/file/d85b63ef0ccb114d0a3bb7b7d808028f-Paper.pdf)
Author : Ekin Dogus Cubuk, Barret Zoph, Jon Shlens, Quoc V. Le


----
Run train.py to train a model with the config and get the performance of the model,

e.g. `python RandAugment/train.py -c confs/wresnet28x10_cifar10_b256.yaml --save path_name.pth`
