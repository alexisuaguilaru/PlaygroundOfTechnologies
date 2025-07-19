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
    return (TORCH_DEVICE,)


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

    batch_size = 128
    dataloader_train = DataLoader(dataset_train,batch_size=batch_size,shuffle=True)
    dataloader_test = DataLoader(dataset_test,batch_size=batch_size,shuffle=True)
    return batch_size, dataloader_test, dataloader_train


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
                nn.Conv2d(3,9,kernel_size=2),
                nn.ReLU(),
                nn.MaxPool2d(2,2),
            )
            ConvLayer2 = nn.Sequential(
                nn.Conv2d(9,6,kernel_size=2,stride=2),
                nn.ReLU(),
                nn.MaxPool2d(2,2),
            )
            self.ConvNet = nn.Sequential(
                ConvLayer1,
                ConvLayer2,
                nn.Flatten(1),
            )

            # Defining Dense Network
            self.DenseNet = nn.Sequential(
                nn.Linear(54,128),
                nn.Tanh(),
                nn.Linear(128,64),
                nn.ReLU(),
                nn.Linear(64,10),
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
def _(CNN, TORCH_DEVICE):
    # Init of model

    ModelCNN = CNN().to(TORCH_DEVICE)

    ModelCNN
    return (ModelCNN,)


@app.cell
def _(mo):
    mo.md(r"# 3. Model Training")
    return


@app.cell
def _(TORCH_DEVICE, batch_size, torch):
    # Based on :: https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html#optimization-loop

    def train_loop(
            dataloader, 
            model,
            loss_fn,
            optimizer,
        ):

        size = len(dataloader.dataset)

        model.train()
        for batch , data_train in enumerate(dataloader):
            X , y = data_train[0].to(TORCH_DEVICE) , data_train[1].to(TORCH_DEVICE)

            pred = model(X)
            loss = loss_fn(pred, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 10 == 0:
                loss, current = loss.item(), batch * batch_size + len(X)
                print(f"loss: {loss:>5f}  [{current:>10d}/{size:>10d}]")

    def test_loop(dataloader, model, loss_fn):
        model.eval()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0

        with torch.no_grad():
            for data_test in dataloader:
                X , y = data_test[0].to(TORCH_DEVICE) , data_test[1].to(TORCH_DEVICE)

                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loop, train_loop


@app.cell
def _(ModelCNN, nn, torch):
    # Init loss function and optimizer

    loss_fn = nn.CrossEntropyLoss()

    learning_rate = 10e-5
    optimizer = torch.optim.Adam(
        ModelCNN.parameters(),
        lr=learning_rate,
    )
    return loss_fn, optimizer


@app.cell
def _(
    ModelCNN,
    dataloader_test,
    dataloader_train,
    loss_fn,
    optimizer,
    test_loop,
    train_loop,
):
    # Training the model/CNN

    epochs = 3
    for _t in range(epochs):
        print(f"Epoch {_t+1}\n-------------------------------")
        train_loop(dataloader_train, ModelCNN, loss_fn, optimizer)
        test_loop(dataloader_test, ModelCNN, loss_fn)
    return


if __name__ == "__main__":
    app.run()
