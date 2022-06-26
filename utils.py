import torch.distributed as dist
import torch 

import os

def load_checkpoint(model, model_name, ckpt_dir):
    print(f"Load: {model_name}") 
    load_path = os.path.join(ckpt_dir, model_name)
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model'])
    return model

def save_checkpoint(save_model, model, epoch, ckpt_dir):
    ckpt_name = save_model + f"_{epoch}"
    print(f"Save: {ckpt_name}")
    state = {
        'model': model.state_dict(),
        'epoch': epoch,
    }
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    torch.save(state, os.path.join(ckpt_dir, ckpt_name))

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False

    if not dist.is_initialized():
        return False

    return True

def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0

    return dist.get_rank()

def is_main_process():

    return get_rank() == 0

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
