Fast AutoAugment / RandAugment
===

**Fast AutoAugment -- NeurlIPS (2019)**

[paper site](https://papers.nips.cc/paper_files/paper/2019/file/6add07cf50424b14fdf649da87843d01-Paper.pdf)

Author : Sungbin Lim, Ildoo Kim, Taesup Kim, Chiheon Kim, Sungwoong Kim  
<br>
<br>
**RandAugment: Practical Automated Data Augmentation with a Reduced Search Space -- NeurlIPS (2020)**

[paper site](https://papers.nips.cc/paper_files/paper/2020/file/d85b63ef0ccb114d0a3bb7b7d808028f-Paper.pdf)

Author : Ekin Dogus Cubuk, Barret Zoph, Jon Shlens, Quoc V. Le


----
We can run train.py with the config (which performed on the course) and get the performance of the model,

e.g. `python RandAugment/train.py -c confs/wresnet28x10_cifar10_b256.yaml --save path_name.pth`

Reference : [Fast AutoAugment](https://github.com/kakaobrain/fast-autoaugment) / [RandAugment](https://github.com/ildoonet/pytorch-randaugment)
