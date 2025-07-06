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
    from torch.utils.data import Dataset , DataLoader , random_split

    ## Module for CV
    from torchvision import datasets, transforms

    ## Additional utils
    from typing import Any
    return Any, DataLoader, Dataset, Tensor, mo, nn, pd, random_split, torch


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

            if self.transform: instance_x = self.transform(instance_x)
            if self.target_transform: label = self.target_transform(label)
            return instance_x , label
    return (ExampleDataset,)


@app.cell
def _(Tensor, pd):
    # Functions for preprocessing values from dataset

    def TransformationFeature(instance: pd.Series) -> Tensor:
        for categorical_feature in ['Stage_fear','Drained_after_socializing']:
            instance[categorical_feature] = int(instance.loc[categorical_feature][0] == 'Y')
        return Tensor(instance) # Must return a Tensor type

    def TransformationLabel(label: str) -> int:
        return int(label[0] == 'E') # For Classification problems, return can be a scalar
    return TransformationFeature, TransformationLabel


@app.cell
def _(ExampleDataset, TransformationFeature, TransformationLabel, pd):
    # Definition of dataset using the custom dataset subclass

    PATH = './DeepLearning/01_IntroPyTorch/'

    src_data = PATH + 'personality_dataset.csv'
    _dataset_columns = pd.read_csv(src_data).columns
    features = _dataset_columns[:-1]
    target = _dataset_columns[-1]

    dataset = ExampleDataset(
        src_data,
        features,
        target,
        TransformationFeature,
        TransformationLabel,
    )
    return (dataset,)


@app.cell
def _(mo):
    mo.md(r"`torch.utils.data.DataLoader` enables easy access/interface to the samples/instances in `torch.utils.data.Dataset` during training stage. Retrieves data in batches of `batch_size` size at each epoch or iteration.")
    return


@app.cell
def _(DataLoader, dataset, random_split, torch):
    # Spliting dataset into train and test sets

    rand_gen = torch.random.manual_seed(8013)
    dataset_train , dataset_test = random_split(dataset,[0.8,0.2],rand_gen)

    # Definition and init of dataloaders

    BATCH_SIZE = 32
    dataloader_train = DataLoader(dataset_train,batch_size=BATCH_SIZE,shuffle=True)
    dataloader_test = DataLoader(dataset_test,batch_size=BATCH_SIZE,shuffle=True)
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

            # Architecture/Topology
            ## torch.nn.Sequential chains a sequence of layers in one interface
            self.NN = nn.Sequential( 
                nn.Linear(in_features=7,out_features=4),
                nn.Dropout(p=0.3), # Deactivate some neurons with probability p
                nn.ReLU(), 
                nn.Linear(in_features=4,out_features=3),
                nn.Dropout(p=0.7),
                nn.ReLU(),
                nn.Linear(in_features=3,out_features=2), # Output layer
            )

        def forward(
                self,
                x: Tensor
            ) -> Tensor:
            """
            Operations over data/input 
            and its flow throught the 
            newtwork is defined here
            """
            logits = self.NN(x) # Compute final outputs (logits for clasification problems)
            return logits
    return (InitialNeuralNetwork,)


@app.cell
def _(InitialNeuralNetwork):
    # Instance of NN

    model = InitialNeuralNetwork()
    print(model)
    return


if __name__ == "__main__":
    app.run()
