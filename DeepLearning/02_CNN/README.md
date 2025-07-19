## Brief
Introductory experiment on Convolutional Networks (CNN) following the example in [Training a Classifier](https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) in which the handling of the datasets provided by `torchvision.datasets` is also explored.

## Comment
First it was explored the [`torchvision` datasets](https://docs.pytorch.org/vision/master/datasets.html) and it was chosen to use the [CIFAR10 Dataset](https://docs.pytorch.org/vision/master/generated/torchvision.datasets.CIFAR10.html#torchvision.datasets.CIFAR10) to follow the tutorial. Then the model was defined using the `torch.nn.Conv2D` layers while learning how to create a CNN.

The procedure followed and explained is available in [02_CNN](./02_CNN.py) as an interactive notebook of [Marimo](https://marimo.io/).

## Results
Performing a first CNN architecture allowed me several of the theoretical concepts that allow it to be so efficient for tasks such as image classification.