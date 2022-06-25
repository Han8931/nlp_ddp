"""
Ref: https://theaisummer.com/distributed-training-pytorch/

nn.DataParallel splits the batch and processes it independently in all the available GPU’s. 
In each forward pass, the module is replicated on each GPU, which is a significant overhead. 
Each replica handles a portion of the batch (batch_size / gpus). 
During the backwards pass, gradients from each replica are summed into the original module.

It is recommended to use nn.DistributedDataParallel, 
instead of this class, to do multi-GPU training, even if there is only a single node.

The reason is that DistributedDataParallel uses one process per worker (GPU) 
while DataParallel encapsulates all the data communication in a single process.

According to the docs, the data can be on any device before they are passed into the model.

In my experiment, DataParallel was slower than training on a single GPU.  Even with 4 GPUs. 
After increasing the number of workers I reduced the time, but still worse than a single GPU. 
I measure and report the time required to train the model for one epoch, that is 50K 32x32 images.
"""
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

def create_data_loader_cifar10():
    transform = transforms.Compose([ 
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 256

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=10, pin_memory=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=10)
    return trainloader, testloader

def train(net, trainloader):
    print("Start training...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    epochs = 1
    num_of_batches = len(trainloader)

    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            images, labels = inputs.cuda(), labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

        print(f'[Epoch {epoch + 1}/{epochs}] loss: {running_loss / num_of_batches:.3f}')
    print('Finished Training')

def test(net, PATH, testloader):
    net.load_state_dict(torch.load(PATH))

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data

            images, labels = images.cuda(), labels.cuda()
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct // total
    print(f'Accuracy of the network on the 10000 test images: {acc} %')


if __name__ == '__main__':
    start = time.time()  
    PATH = './cifar_net.pth'
    trainloader, testloader = create_data_loader_cifar10()
    net = torchvision.models.resnet50(False).cuda()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # Batch size should be divisible by number of GPUs
        net = nn.DataParallel(net)

    start_train = time.time()
    train(net, trainloader)
    end_train = time.time()
    # save
    torch.save(net.state_dict(), PATH)
    # test
    test(net, PATH, testloader)

    end = time.time()
    seconds = (end - start)
    seconds_train = (end_train - start_train)
    print(f"Total elapsed time: {seconds:.2f} seconds, Train 1 epoch {seconds_train:.2f} seconds")
