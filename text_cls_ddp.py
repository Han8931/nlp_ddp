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
from utils import setup_for_distributed, save_on_master, is_main_process, get_rank
from utils import save_checkpoint, load_checkpoint

from transformers import AdamW

from datasets import load_dataset, Dataset, list_datasets

import argparse
import time
import pdb

def init_distributed():

    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    dist_url = "env://" # default

    # only works with torch.distributed.launch // torch.run
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])

#    print(f"WorldSize: {world_size}") 
#    print(f"LocalRank: {local_rank}") 
#    print(f"Rank: {rank}") 

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



class ClsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])

        return item

    def __len__(self):
        return len(self.encodings.input_ids)

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def get_parser():
    #group = parser.add_argument_group("Trainer")

    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--exp_dir', type=str, default="./experiments/cls/", help='Experiment directory.')
    parser.add_argument('--exp_msg', type=str, default="CLS Transformer", help='Simple log for experiment')
    parser.add_argument('--gpu_idx', type=int, default=10, help='GPU Index')

    # parser.add_argument('--deterministic', type=boolean_string, default=True, help='Deterministic')
    parser.add_argument('--runs', type=int, default=1, help='# runs for experiments')
    parser.add_argument('--eval', type=boolean_string, default=False, help='Evaluation')
    parser.add_argument('--model_dir_path', default='./checkpoint/', type=str, help='Save Model dir')
    parser.add_argument('--save_model', default='cls_trans', type=str, help='Save Model name')
    parser.add_argument('--load_model', default='cls_trans', type=str, help='Model name')
    parser.add_argument('--save', type=boolean_string, default=False, help='Evaluation')

    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
#    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
#                        help="Device (cuda or cpu)")
    parser.add_argument('--dataset', default='ag', type=str, help='Dataset', choices=['mr', 'ag'])
    parser.add_argument('--num_workers', type=int, default=16, help='Num Workers')

    parser.add_argument('--epochs', type=int, default=1000, help='Training Epochs')
    parser.add_argument('--batch_size', type=int, default=14, help='batch size')
    parser.add_argument('--model', default='roberta', type=str, help='Transformer model',
                        choices=['albert', 'bert', 'rnn', 'distil', 'roberta', 'roberta-large'])
    parser.add_argument('--optim', default='adamw', type=str, help='optimizer')
    parser.add_argument('--scheduler', default='linear', type=str, help='optimizer')
    parser.add_argument('--lr', default=0.00001, type=float, help='Agent learning rate')
    parser.add_argument('--clip', default=1.0, type=float, help='Agent learning rate')

    parser.add_argument('--num_classes', type=int, default=2, help='number of classes')
    parser.add_argument('--pad_idx', type=int, default=0, help='Padding idx')
    parser.add_argument('--max_seq_length', type=int, default=512, help='Maximum sequence length')

    args = parser.parse_known_args()[0]

    return args


def data_loader(args, tokenizer):

    if args.dataset == 'ag':
        dataset = load_dataset("ag_news")
        test, train = dataset['test'], dataset['train']

        split = train.train_test_split(test_size=0.90, seed=0)
        train = split['train']
        dev = split['test']

    print(f"Trainset Size: {len(train)}")
    print(f"Testset Size: {len(test)}")
    print(f"Devset Size: {len(dev)}")

    train_data = tokenizer(train['text'], padding=True, truncation=True, max_length=args.max_seq_length)
    test_data = tokenizer(test['text'], padding=True, truncation=True, max_length=args.max_seq_length)
    dev_data = tokenizer(dev['text'], padding=True, truncation=True, max_length=args.max_seq_length)

    train_label, test_label, dev_label = train['label'], test['label'], dev['label']

    train_set = ClsDataset(train_data, train_label)
    test_set = ClsDataset(test_data, test_label)
    dev_set = ClsDataset(dev_data, dev_label)

    train_sampler = DistributedSampler(dataset=train_set, shuffle=True, drop_last=True)
    test_sampler = DistributedSampler(dataset=test_set, shuffle=True, drop_last=True)
    val_sampler = DistributedSampler(dataset=dev_set, shuffle=True, drop_last=True)

    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, sampler = train_sampler)
    test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, sampler = test_sampler)
    dev_dataloader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=False, sampler = val_sampler)

    return train_dataloader, test_dataloader, dev_dataloader

def train(model, trainloader, args):
    print("Start training...")

    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(model.parameters(), lr=args.lr)
    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    num_of_batches = len(trainloader)

    for epoch in range(args.epochs):  # loop over the dataset multiple times

        trainloader.sampler.set_epoch(epoch)
        running_loss = 0.0
        ddp_loss = torch.zeros(2).cuda()
        for i, batch in enumerate(trainloader, 0):

            # get the inputs; data is a list of [inputs, labels]
            batch = tuple(value.cuda() for key, value in batch.items())
            input_ids, attention_mask, labels = batch

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            ddp_loss[0]+=loss.item()
            ddp_loss[1]+=input_ids.shape[0]


            # print statistics
            running_loss += loss.item()
            if is_main_process:
                if i%100==0:
                    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
                    print(f"epoch: {i}/{epoch}, loss: {loss.item():.4f} ")
                    print(f"Loss: {ddp_loss[0]/ddp_loss[1]} || {ddp_loss[0]} || {ddp_loss[1]}") 
                    print(f"Loss: {ddp_loss[0]/4} || {ddp_loss[0]} || {ddp_loss[1]}") 

#                loss_log = loss.clone().detach()
#                loss_mean = dist.reduce(loss_log, rank=0) / dist.get_world_size()
#                if dist.get_rank() == 0:
#                    # collect results into rank0
#                    print(f"epoch: {epoch}, loss: {loss_mean} ")

        print(f'[Epoch {epoch + 1}/{args.epochs}] loss: {running_loss / num_of_batches:.3f}')

        # save
        if is_main_process:

            ckpt_name = args.save_model + f"_{epoch}"
            print(f"Save: {ckpt_name}")
            state = {
                'model': model.state_dict(),
                'epoch': epoch,
            }
            ckpt_dir = args.model_dir_path
            if not os.path.isdir(ckpt_dir):
                os.makedirs(ckpt_dir)
            torch.save(state, os.path.join(ckpt_dir, ckpt_name))

        dist.barrier()

    print('Finished Training')

def test(model, testloader):

    TP = 0
    total = len(testloader.dataset)

    with torch.no_grad():
        for batch in testloader:
            # get the inputs; data is a list of [inputs, labels]
            batch = tuple(value.cuda() for key, value in batch.items())
            input_ids, attention_mask, labels = batch

            # calculate outputs by running images through the modelwork
            output = model(input_ids, attention_mask)
            preds = output['logits']
            correct = preds.argmax(dim=-1).eq(labels)
            TP += correct.sum()

    if is_main_process:
        dist.all_reduce(TP, op=dist.ReduceOp.SUM)
        acc = 100 * (TP.item()/total)
        print(f'Accuracy: {acc} %')
    dist.barrier()

def cleanup():
    dist.destroy_process_group()

if __name__ == '__main__':
    start = time.time()  
    init_distributed()
    PATH = './cifar_net.pth'

    args = get_parser()

    from transformers import RobertaTokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    print("Load Dataset...")
    train_dataloader, test_dataloader, dev_dataloader = data_loader(args, tokenizer)

    print("Load Model...")
    from transformers import RobertaForSequenceClassification
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=4)
    model.cuda()

    local_rank = int(os.environ['LOCAL_RANK'])
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    print("Start Training...")
    start_train = time.time()
    train(model, train_dataloader, args)
    end_train = time.time()

#    # Load
    if is_main_process:
        load_path = os.path.join(args.model_dir_path, 'cls_trans_0')
        map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
        checkpoint = torch.load(load_path, map_location=map_location)
        model.load_state_dict(checkpoint['model'])
    dist.barrier()

    print("Start test")
    test(model, test_dataloader)
    print("Finish test")

    end = time.time()
    seconds = (end - start)
    seconds_train = (end_train - start_train)
    print(f"Total elapsed time: {seconds:.2f} seconds, Train 1 epoch {seconds_train:.2f} seconds")
    cleanup()
