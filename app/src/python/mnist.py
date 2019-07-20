from mpi4py import MPI
import numpy as np
import torch
import torchvision
import torch.nn as nn

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

class ConvNNet(nn.Module):
    def __init__(self):
        super(ConvNNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 =nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(7 * 7 * 64, 1000)
        self.fc2 = nn.Linear(1000, 10)
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

if _name_ == "_main_":

    epochs = 10
    num_classes = 10
    batch_size = 100
    learn_rate = 0.001
    class_names = np.array(["T-shirt/top","Trouser","Pullover","Dress",
        "Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"])

    # transforms to apply to the data
    trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])

    # MNIST dataset
    train_dataset = torchvision.datasets.FashionMNIST(root=".", train=True, transform=trans, download=True)
    test_dataset = torchvision.datasets.FashionMNIST(root=".", train=False, transform=trans)

    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    cnnModel = ConvNNet()

    loss_f = nn.CrossEntropyLoss()
    adam = torch.optim.Adam(cnnModel.parameters(), lr=learn_rate)

    losses = []
    class_rate_per_class = [0.0] * num_classes
    avg_class_rate = 0.0

    if(rank == 0):
           for e in range(0,epochs/2):
                for i, (batch, labels) in enumerate(train_dataloader):
            # Forward Propogation
                    y = cnnModel(batch)
                    loss = loss_f(y, labels)
                    losses.append(loss.item())

                    # Backward Propogation
                    adam.zero_grad()
                    loss.backward()
                    adam.step()


    else:
        start = rank*epochs/2
        end = start + epochs/2
        for e in range(start,end):
                for i, (batch, labels) in enumerate(train_dataloader):
            # Forward Propogation
                    y = cnnModel(batch)
                    loss = loss_f(y, labels)
                    losses.append(loss.item())

                    # Backward Propogation
                    adam.zero_grad()
                    loss.backward()
                    adam.step()

    # Test CNN
    cnnModel.eval() # This command turns off gradient calculation, which is not needed during testing
    with torch.no_grad():
        total = 0
        for images, labels in test_dataloader:
            y = cnnModel(images)
            _, predicted = torch.max(y.data, 1)
            # predictions = np.hstack((predictions, predicted))
            # y_test = np.hstack((y_test, labels.item()))
            total += labels.size(0)
            avg_class_rate += (predicted == labels).sum().item()
   

