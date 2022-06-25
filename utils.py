import torch.distributed as dist
import torch 

def load_checkpoint_path(model, optimizer, rank, checkpoint_path):
    # configure map_location properly
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    checkpoint_state = torch.load(checkpoint_path, map_location=map_location)

    model.load_state_dict(checkpoint_state['model'])
    iter_init = checkpoint_state['iter_no'] + 1  # next iteration
    optimizer.load_state_dict(checkpoint_state['optimizer'])
    return iter_init


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
