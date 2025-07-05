import marimo

__generated_with = "0.14.10"
app = marimo.App()


@app.cell
def _():
    # Import required libraries 

    import marimo as mo

    import numpy as np
    import pandas as pd

    ## Basic modules and class
    import torch
    from torch import nn , Tensor

    ## Processing and load datasets
    from torch.utils.data import Dataset , DataLoader

    ## Module for CV
    from torchvision import datasets, transforms

    ## Additional utils
    from typing import Any
    return Any, Dataset, Tensor, mo, nn, pd


@app.cell
def _(mo):
    mo.md(r"# 1. Datasets and DataLoader")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        `torch.utils.data.Dataset` and `torch.utils.data.DataLoader` are used for loading data to neural networks training.
    
        `torch.utils.data.Dataset` allows storing data samples and, by subclassing, creates a custom interface to get own data. For doing the last, `__getitem__` method must be overriden, `__init__` and `__len__` optionally.
        """
    )
    return


@app.cell
def _(Any, Dataset, Tensor, pd):
    class ExampleDataset(Dataset):
        """
        Custom dataset for fetching/getting 
        data from a file/source with PyTorch
        """

        def __init__(
                self,
                src: str,
                feature_columns: list[str],
                target_column: str,
                transform = None,
                target_transform = None,
            ):
            """
            Init some metadata, sources 
            and formats about the dataset
            """
            self.pd_dataset = pd.read_csv(src)
            self.feature_columns = feature_columns
            self.target_column = target_column
            self.transform = transform
            self.target_transform = target_transform

        def __len__(self) -> int:
            """
            Method used by DataLoader to 
            draw samples/instances for  
            training
            """
            return len(self.pd_dataset)

        def __getitem__(
                self,
                idx: int
            ) -> tuple[Tensor, Any]:
            """
            Return a instance/sample from 
            the dataset given its index
            """
            instance = self.pd_dataset.iloc[idx]
            instance_x = instance[self.feature_columns]
            label = instance[self.target_column]
            return self.transform(instance_x) , self.target_transform(label)
    return


@app.cell
def _():
    PATH = './DeepLearning/01_IntroPyTorch/'
    src_data = PATH + 'personality_dataset.csv'


    return


@app.cell
def _(mo):
    mo.md(r"# 2. First Example of Neural Network")
    return


@app.cell
def _(mo):
    mo.md(r"A good practice for building a neural network with PyTorch is defining a subclass of `torch.nn.Module` (base class for layers and others modules in PyTorch). Both `__init__` and `forward` methods must be overriden.")
    return


@app.cell
def _(Tensor, nn):
    class InitialNeuralNetwork(nn.Module):
        """
        First introductory example to build a 
        Neural Network with PyTorch
        """

        def __init__(self):
            """
            Init neural network layers 
            in __init__ method
            """
            super().__init__()

        def forward(
                self,
                x: Tensor
            ) -> Tensor:
            """
            Operations over data/input 
            and its flow throught the 
            newtwork is defined here
            """
            return x
    return


if __name__ == "__main__":
    app.run()
