# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).
# To view a copy of this license, visit
# https://nvlabs.github.io/few-shot-vid2vid/License.txt
import os
import functools

import torch
import torch.distributed as dist

def init_dist(launcher='pytorch', backend='nccl', **kwargs):
    if dist.is_initialized():
        return torch.cuda.current_device()
    # if mp.get_start_method(allow_none=True) is None:
        # mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    gpu_id = rank % num_gpus
    torch.cuda.set_device(gpu_id)
    dist.init_process_group(backend=backend, **kwargs)
    return gpu_id


def get_rank():
    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0
    return rank


def get_world_size():
    if dist.is_initialized():
        world_size = dist.get_world_size()
    else:
        world_size = 1
    return world_size


def master_only(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if get_rank() == 0:
            return func(*args, **kwargs)
        else:
            return None
    return wrapper


def is_master():
    """check if current process is the master"""
    return get_rank() == 0


@master_only
def master_only_print(*args):
    """master-only print"""
    print(*args)


def dist_reduce_tensor(tensor):
    """ Reduce to rank 0 """
    world_size = get_world_size()
    if world_size < 2:
        return tensor
    with torch.no_grad():
        dist.reduce(tensor, dst=0)
        if get_rank() == 0:
            tensor /= world_size
    return tensor


def dist_all_reduce_tensor(tensor):
    """ Reduce to all ranks """
    world_size = get_world_size()
    if world_size < 2:
        return tensor
    with torch.no_grad():
        dist.all_reduce(tensor)
        tensor.div_(world_size)
    return tensor


def dist_all_gather_tensor(tensor):
    """ gather to all ranks """
    world_size = get_world_size()
    if world_size < 2:
        return [tensor]
    tensor_list = [
        torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    with torch.no_grad():
        dist.all_gather(tensor_list, tensor)
    return tensor_list
