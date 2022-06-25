import os

gpu_list = "0,1,2,3"
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list 

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

import time
import pdb

from utils import setup_for_distributed, save_on_master, is_main_process, get_rank


#def prepare(rank, world_size, batch_size=256, pin_memory=False, num_workers=0):
#
#    transform = transforms.Compose([ 
#        transforms.RandomCrop(32),
#        transforms.RandomHorizontalFlip(),
#        transforms.ToTensor(),
#        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#
#    batch_size = 256
#
#    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
#    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
#
#    train_sampler = DistributedSampler(dataset=trainset, shuffle=True, num_replicas=world_size, rank=rank, drop_last=True)
#    test_sampler = DistributedSampler(dataset=testset, shuffle=True, num_replicas=world_size, rank=rank, drop_last=True)
#
#    trainloader = DataLoader(trainset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=pin_memory)
#    testloader = DataLoader(testset, batch_size=batch_size, sampler=test_sampler, shuffle=False, num_workers=num_workers)
#
#    return trainloader, testloader

def create_data_loader_cifar10(num_workers=0):

    transform = transforms.Compose([ 
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 256

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_sampler = DistributedSampler(dataset=trainset, shuffle=True)
    trainloader = DataLoader(trainset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=False, drop_last=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_sampler = DistributedSampler(dataset=testset, shuffle=True, drop_last=True)
    testloader = DataLoader(testset, batch_size=batch_size, sampler=test_sampler, shuffle=False, num_workers=num_workers)

    return trainloader, testloader

def train(net, trainloader):
    print("Start training...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    epochs = 1
    num_of_batches = len(trainloader)

    for epoch in range(epochs):  # loop over the dataset multiple times

        trainloader.sampler.set_epoch(epoch)
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

#            if i%100==0:
#                loss_log = loss.clone().detach()
#                loss_mean = dist.reduce(loss_log, rank=0) / dist.get_world_size()
#                if dist.get_rank() == 0:
#                    # collect results into rank0
#                    print(f"epoch: {epoch}, loss: {loss_mean} ")

        print(f'[Epoch {epoch + 1}/{epochs}] loss: {running_loss / num_of_batches:.3f}')
    print('Finished Training')

def test(net, PATH, testloader):

    #Convert BatchNorm to SyncBatchNorm
#    net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
#
#    local_rank = int(os.environ['LOCAL_RANK'])
#    net = nn.parallel.DistributedDataParallel(net, device_ids=[local_rank])

    #net.load_state_dict(torch.load(PATH))

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

def cleanup():
    dist.destroy_process_group()

if __name__ == '__main__':
    start = time.time()  
    init_distributed()
    PATH = './cifar_net.pth'

    trainloader, testloader = create_data_loader_cifar10(num_workers=16)
    net = torchvision.models.resnet50(False).cuda()

    #Convert BatchNorm to SyncBatchNorm
    net = nn.SyncBatchNorm.convert_sync_batchnorm(net)

    local_rank = int(os.environ['LOCAL_RANK'])
    net = nn.parallel.DistributedDataParallel(net, device_ids=[local_rank])

    start_train = time.time()
    train(net, trainloader)
    end_train = time.time()

    # save
    if is_main_process:
        save_on_master(net.state_dict(), PATH)
    dist.barrier()

    # test
    print("Start test")
    if is_main_process:
        map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
        print(map_location)
        net_ = torch.load(PATH, map_location=map_location)
        net.load_state_dict(net_)
    dist.barrier()

    test(net, PATH, testloader)
    print("Finish test")

    end = time.time()
    seconds = (end - start)
    seconds_train = (end_train - start_train)
    print(f"Total elapsed time: {seconds:.2f} seconds, Train 1 epoch {seconds_train:.2f} seconds")
    #cleanup()
