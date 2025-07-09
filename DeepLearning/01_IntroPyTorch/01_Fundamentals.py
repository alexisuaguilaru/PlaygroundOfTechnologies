import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


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

    batch_size = 32
    dataloader_train = DataLoader(dataset_train,batch_size=batch_size,shuffle=True)
    dataloader_test = DataLoader(dataset_test,batch_size=batch_size,shuffle=True)
    return batch_size, dataloader_test, dataloader_train


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
    return (model,)


@app.cell
def _(mo):
    mo.md(r"# 3. Optmizer and Loss Function")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        After defining/building a model (Neural Network) it is time for training and fitting. For this, `Optimizer` and `Loss Function` must be selected to learn the best parameters (weights and biases).
    
        `Loss Function` or `loss_fn` (subclass of `torch.NN.Module`) measures the perfomance of neural network predictions against the true values. `loss_fn.backward()` method computes the gradients based on loss value and is called after `torch.NN.Module.__call__` with a train instance.
    
        `Optimizer` (subclass of `torch.optim.Optimizer`) allows adjusting parameters after each epoch (train loop). `Optimizer.step()` method updates model parameters using the computed gradients with `loss_fn.backward()`
    
        `torcheval.metrics` offers other evaluation metrics build on PyTorch
        """
    )
    return


@app.cell
def _(nn):
    # Init loss function

    loss_fn = nn.CrossEntropyLoss()
    return (loss_fn,)


@app.cell
def _(model, torch):
    # Init optimizer

    learning_rate = 10e-5 # Hiperparameter for training
    optimizer = torch.optim.Adam(
        model.parameters(), # Model parameters to fit/adjust/optimize
        # Other Optimizer parameterss like learning rate, weight decay
        lr=learning_rate,
    )
    return learning_rate, optimizer


@app.cell
def _(batch_size, torch):
    # Source code :: https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html#optimization-loop

    def train_loop(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        # Set the model to training mode - important for batch normalization and dropout layers
        # Unnecessary in this situation but added for best practices
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            # Compute prediction and loss
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad() # Reset computed gradients

            if batch % 100 == 0:
                loss, current = loss.item(), batch * batch_size + len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


    def test_loop(dataloader, model, loss_fn):
        # Set the model to evaluation mode - important for batch normalization and dropout layers
        # Unnecessary in this situation but added for best practices
        model.eval()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0

        # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
        # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
        with torch.no_grad():
            for X, y in dataloader:
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loop, train_loop


@app.cell
def _(
    dataloader_test,
    dataloader_train,
    loss_fn,
    model,
    optimizer,
    test_loop,
    train_loop,
):
    # Source code :: https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html#optimization-loop

    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(dataloader_train, model, loss_fn, optimizer)
        test_loop(dataloader_test, model, loss_fn)
    print("Done!")
    return


@app.cell
def _(batch_size, torch):
    def train_loop_custom(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)

        model.train()
        for batch, (X, y) in enumerate(dataloader):
            pred = model(X)
            loss = loss_fn(pred, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 10 == 0:
                loss, current = loss.item(), batch * batch_size + len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test_loop_custom(dataloader, model, loss_fn, metric):
        model.eval()
        num_batches = len(dataloader)
        test_loss = 0

        with torch.no_grad():
            for X, y in dataloader:
                pred = model(X)
                test_loss += loss_fn(pred, y).item()

                metric.update(pred.argmax(1), y) # Method for loading/saving predictions-target pairs

        test_loss /= num_batches
        print(f"Test Error: \nF1: {(metric.compute()*100):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        metric.reset() # Method for reseting/deleting pred-target pairs 
    return test_loop_custom, train_loop_custom


@app.cell
def _(
    InitialNeuralNetwork,
    dataloader_test,
    dataloader_train,
    learning_rate,
    loss_fn,
    test_loop_custom,
    torch,
    train_loop_custom,
):
    from torcheval.metrics import MulticlassF1Score

    _model = InitialNeuralNetwork()
    _optimizer = torch.optim.Adam(
        _model.parameters(),
        lr=learning_rate,
    )

    # Init metric for model evaluation
    _metric = MulticlassF1Score(
        num_classes=2,
        average='weighted',
    )

    # Using custom train and test loops
    _epochs = 5
    for _t in range(_epochs):
        print(f"Epoch {_t+1}\n-------------------------------")
        train_loop_custom(dataloader_train, _model, loss_fn, _optimizer)
        test_loop_custom(dataloader_test, _model, loss_fn, _metric)
    return


if __name__ == "__main__":
    app.run()
