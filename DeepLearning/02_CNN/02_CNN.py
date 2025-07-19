import marimo

__generated_with = "0.14.10"
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
    return DataLoader, Tensor, datasets, mo, nn, transforms


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
def _(Tensor, nn):
    class CNN(nn.Module):
        def __init__(
                self,
            ):
            super().__init__()

        def forward(
                self,
                x: Tensor
            ) -> Tensor:
            pass
    return


if __name__ == "__main__":
    app.run()
