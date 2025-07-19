import marimo

__generated_with = "0.14.12"
app = marimo.App()


@app.cell
def _():
    # Import required libraries 
    import marimo as mo

    # Basic modules and class of PyTorch
    import torch
    from torch import nn , Tensor
    from torch.utils.data import DataLoader
    from torchvision import datasets , transforms

    # Auxiliar modules
    import matplotlib.pyplot as plt
    return DataLoader, Tensor, datasets, mo, nn, torch, transforms


@app.cell
def _(torch):
    # Useful variables

    TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    TORCH_DEVICE
    return


@app.cell
def _(mo):
    mo.md(r"# 1. Load Dataset")
    return


@app.cell
def _(mo):
    mo.md(r"When a CNN is trained, each image is required to be a tensor, thus the representation of a image is transformed into a `torch.Tensor` object or using `torchvision.io.decode_image` to get a `torch.Tensor` object that represents a image.")
    return


@app.cell
def _(datasets, transforms):
    # Loading train and test datasets

    path_dataset = './DeepLearning/02_CNN/Dataset'
    Transformation = transforms.Compose( # Each image is applied a pipeline of transformations
        [
            transforms.ToTensor(), # Convert image to a torch.Tensor object
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # Transforming the range of values to [-1,1]
        ]
    )

    dataset_train = datasets.CIFAR10(
        path_dataset,
        train=True,
        download=True,
        transform=Transformation,
    )

    dataset_test = datasets.CIFAR10(
        path_dataset,
        train=False,
        download=True,
        transform=Transformation,
    )
    return dataset_test, dataset_train


@app.cell
def _(DataLoader, dataset_test, dataset_train):
    # Defining dataloaders for training and testing

    batch_size = 16
    dataloader_train = DataLoader(dataset_train,batch_size=batch_size,shuffle=True)
    dataloader_test = DataLoader(dataset_test,batch_size=batch_size,shuffle=True)
    return


@app.cell
def _(mo):
    mo.md(r"# 2. Model Definition")
    return


@app.cell
def _(mo):
    mo.md(r"The basic construction blocks for a CNN are convolutional layers, that consists of a convolutional operation, activation function and pooling operation. A CNN is built by stacking several convolutional layers and a dense network.")
    return


@app.cell
def _(Tensor, nn):
    class CNN(nn.Module):
        def __init__(
                self,
            ):
            super().__init__()

            # Defining Convolutional Network
            ConvLayer1 = nn.Sequential(
                nn.Conv2d(3,5,kernel_size=3),
                nn.ReLU(),
                nn.MaxPool2d(2,2),
            )
            ConvLayer2 = nn.Sequential(
                nn.Conv2d(5,4,kernel_size=2,stride=2),
                nn.ReLU(),
                nn.AvgPool2d(2,2),
            )
            self.ConvNet = nn.Sequential(
                ConvLayer1,
                ConvLayer2,
                nn.Flatten(1),
            )

            # Defining Dense Network
            self.DenseNet = nn.Sequential(
                nn.Linear(36,12),
                nn.Tanh(),
                nn.Linear(12,10),
            )

        def forward(
                self,
                x: Tensor
            ) -> Tensor:
            x = self.ConvNet(x)
            logits = self.DenseNet(x)
            return logits
    return (CNN,)


@app.cell
def _(CNN):
    # Init of model

    ModelCNN = CNN()

    ModelCNN
    return


if __name__ == "__main__":
    app.run()
