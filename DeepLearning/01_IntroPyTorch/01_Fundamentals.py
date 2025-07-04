import marimo

__generated_with = "0.14.10"
app = marimo.App()


@app.cell
def _():
    # Import required libraries 

    import marimo as mo

    import numpy as np

    ## Basic modules and class
    import torch
    from torch import nn , Tensor

    ## Processing and load datasets
    from torch.utils.data import Dataset , DataLoader

    ## Module for CV
    from torchvision import datasets, transforms
    return Tensor, mo, nn


@app.cell
def _(mo):
    mo.md(r"# 1. First Example of Neural Network")
    return


@app.cell
def _(mo):
    mo.md(r"A good practice for building a neural network with PyTorch is define a subclass of `torch.nn.Module` (base class for layers and others modules in PyTorch). Both `__init__` and `forward` methods must be overriden.")
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
