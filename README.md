# PyTorch Distributed Data Parallel for Transformer Models

A simple example for distributed training and evaluating a Transformer-based text classification model on a single node with multiple GPUs. 
This code is based on https://theaisummer.com/distributed-training-pytorch/

Train:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 text_cls_ddp.py --batch_size 8
```

Test only:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 text_cls_test_ddp.py --batch_size 8
```


https://pytorch.org/docs/master/notes/ddp.html
