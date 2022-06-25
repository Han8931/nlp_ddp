"""
Ref: https://medium.com/codex/a-comprehensive-tutorial-to-pytorch-distributeddataparallel-1f4b42bb1b51
- master node: the main gpu responsible for synchronizations, making copies, loading models, writing logs;

- process group: if you want to train/test the model over K gpus, 
then the K process forms a group, which is supported by a backend 
(pytorch managed that for you, according to the documentation, "nccl" is the most recommended backend);

- rank: within the process group, each process is identified by its rank, from 0 to K-1;

- world size: the number of processes in the group i.e. gpu number, K.

In short, DDP is faster, more flexible than DP. 
The fundamental thing DDP does is to copy the model to multiple gpus, 
gather the gradients from them, average the gradients to update the model, 
then synchronize the model over all K processes.

In case the model can fit on one gpu (it can be trained on one gpu with batch_size=1) 
and we want to train/test it on K gpus, 
the best practice of DDP is to copy the model onto the K gpus 
(the DDP class automatically does this for you) and split the dataloader 
to K non-overlapping groups to feed into K models respectively.

1. setup the process group, which is three lines of code and needs no modification;
2. split the dataloader to each process in the group, which can be easily achieved by 
    torch.utils.data.DistributedSampler or any customized sampler;
3. wrap our model with DDP, which is one line of code and barely needs modification;
4. train/test our model, which is the same as is on 1 gpu;
5. clean up the process groups (like free in C), which is one line of code.
6. optional: gather extra data among processes (possibly needed for distributed testing), which is basically one line of code;
"""

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

def setup(rank, world_size):    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'    
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def init_distributed():

    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    dist_url = "env://" # default

    # only works with torch.distributed.launch // torch.run
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])

    print(f"WorldSize: {world_size}") 
    print(f"LocalRank: {local_rank}") 
    print(f"Rank: {rank}") 

    dist.init_process_group(
            backend="nccl",
            init_method=dist_url,
            world_size=world_size,
            rank=rank)

    # this will make all .cuda() calls work properly
    torch.cuda.set_device(local_rank)

    # synchronizes all the threads to reach this point before moving on
    dist.barrier()
    setup_for_distributed(rank == 0)

def prepare(rank, world_size, batch_size=256, pin_memory=False, num_workers=0):

    transform = transforms.Compose([ 
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 256

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_sampler = DistributedSampler(dataset=trainset, shuffle=True, num_replicas=world_size, rank=rank, drop_last=True)
    test_sampler = DistributedSampler(dataset=testset, shuffle=True, num_replicas=world_size, rank=rank, drop_last=True)

    trainloader = DataLoader(trainset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=pin_memory)
    testloader = DataLoader(testset, batch_size=batch_size, sampler=test_sampler, shuffle=False, num_workers=num_workers)

    return trainloader, testloader

def create_data_loader_cifar10():

    transform = transforms.Compose([ 
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 256

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_sampler = DistributedSampler(dataset=trainset, shuffle=True, num_replicas=4)
    trainloader = DataLoader(trainset, batch_size=batch_size, sampler=train_sampler, num_workers=10, pin_memory=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_sampler = DistributedSampler(dataset=testset, shuffle=True, num_replicas=4)
    testloader = DataLoader(testset, batch_size=batch_size, sampler=test_sampler, shuffle=False, num_workers=10)

    return trainloader, testloader

def train(net, trainloader):

#    setup(rank, world_size)    # prepare the dataloader
#    # instantiate the model(it's your own model) and move it to the right device
#    net = torchvision.models.resnet50(False).to(rank)
#    net = DDP(net, device_ids=[rank], output_device=rank, find_unused_parameters=True)

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

        print(f'[Epoch {epoch + 1}/{epochs}] loss: {running_loss / num_of_batches:.3f}')
    print('Finished Training')

def test(net, PATH, testloader):
    if is_main_process:
        net.load_state_dict(torch.load(PATH))
    dist.barrier()

    #Convert BatchNorm to SyncBatchNorm
    net = nn.SyncBatchNorm.convert_sync_batchnorm(net)

    local_rank = int(os.environ['LOCAL_RANK'])
    net = nn.parallel.DistributedDataParallel(net, device_ids=[local_rank])

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


def main(rank, world_size):
    # setup the process groups
    setup(rank, world_size)    # prepare the dataloader
    trainloader, testloader = prepare(rank, world_size)

    # instantiate the model(it's your own model) and move it to the right device
    net = torchvision.models.resnet50(False).to(rank)
    model = DDP(net, device_ids=[rank], output_device=rank, find_unused_parameters=True)

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

        print(f'[Epoch {epoch + 1}/{epochs}] loss: {running_loss / num_of_batches:.3f}')
    print('Finished Training')

    # wrap the model with DDP
    # device_ids tell DDP where is your model
    # output_device tells DDP where to output, in our case, it is rank
    # find_unused_parameters=True instructs DDP to find unused output of the forward() function of any module in the model    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

import torch.multiprocessing as mp

if __name__ == '__main__':
    start = time.time()  
#    init_distributed()
#    PATH = './cifar_net.pth'
#
#    trainloader, testloader = create_data_loader_cifar10()
#    net = torchvision.models.resnet50(False).cuda()
#
#    #Convert BatchNorm to SyncBatchNorm
#    net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
#
#    local_rank = int(os.environ['LOCAL_RANK'])
#
##    if is_main_process:
##        print(f"Local Rank: {local_rank}") 
##    dist.barrier()
#
#    net = nn.parallel.DistributedDataParallel(net, device_ids=[local_rank])

    world_size = 4        
    mp.spawn(
        main,
        args=(world_size),
        nprocs=world_size
    )

#    start_train = time.time()
#    main(net, trainloader)
#    #train(net, trainloader)
#    end_train = time.time()
#
#    # save
#    if is_main_process:
#        save_on_master(net.state_dict(), PATH)
#    dist.barrier()
#
#    # test
#    test(net, PATH, testloader)
#
#    end = time.time()
#    seconds = (end - start)
#    seconds_train = (end_train - start_train)
#    print(f"Total elapsed time: {seconds:.2f} seconds, Train 1 epoch {seconds_train:.2f} seconds")
