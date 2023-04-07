import torch
import numpy as np
import random
import time
import os
import sys
import functools
from pathlib import Path
from torch.backends import cudnn
from torch import nn, Tensor
from torch.autograd import profiler
from typing import Union
from torch import distributed as dist
from tabulate import tabulate
from semseg import models
import logging
from fvcore.nn import flop_count_table, FlopCountAnalysis
import datetime

def fix_seeds(seed: int = 3407) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def setup_cudnn() -> None:
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    cudnn.benchmark = True
    cudnn.deterministic = False

def time_sync() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

def get_model_size(model: Union[nn.Module, torch.jit.ScriptModule]):
    tmp_model_path = Path('temp.p')
    if isinstance(model, torch.jit.ScriptModule):
        torch.jit.save(model, tmp_model_path)
    else:
        torch.save(model.state_dict(), tmp_model_path)
    size = tmp_model_path.stat().st_size
    os.remove(tmp_model_path)
    return size / 1e6   # in MB

@torch.no_grad()
def test_model_latency(model: nn.Module, inputs: torch.Tensor, use_cuda: bool = False) -> float:
    with profiler.profile(use_cuda=use_cuda) as prof:
        _ = model(inputs)
    return prof.self_cpu_time_total / 1000  # ms

def count_parameters(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6      # in M

def setup_ddp():
    # print(os.environ.keys())
    if 'SLURM_PROCID' in os.environ and not 'RANK' in os.environ:
        # --- multi nodes
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ["SLURM_PROCID"])
        gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
        gpu = rank - gpus_per_node * (rank // gpus_per_node)
        torch.cuda.set_device(gpu)
        dist.init_process_group(backend="nccl", world_size=world_size, rank=rank, timeout=datetime.timedelta(seconds=7200))
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        # gpu = int(os.environ(['LOCAL_RANK']))
        # ---
        gpu = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(gpu)
        dist.init_process_group('nccl', init_method="env://",world_size=world_size, rank=rank, timeout=datetime.timedelta(seconds=7200))
        dist.barrier()
    else:
        gpu = 0
    return gpu

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

def reduce_tensor(tensor: Tensor) -> Tensor:
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt

@torch.no_grad()
def throughput(dataloader, model: nn.Module, times: int = 30):
    model.eval()
    images, _  = next(iter(dataloader))
    images = images.cuda(non_blocking=True)
    B = images.shape[0]
    print(f"Throughput averaged with {times} times")
    start = time_sync()
    for _ in range(times):
        model(images)
    end = time_sync()

    print(f"Batch Size {B} throughput {times * B / (end - start)} images/s")


def show_models():
    model_names = models.__all__
    model_variants = [list(eval(f'models.{name.lower()}_settings').keys()) for name in model_names]

    print(tabulate({'Model Names': model_names, 'Model Variants': model_variants}, headers='keys'))


def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        print(f"Elapsed time: {elapsed_time * 1000:.2f}ms")
        return value
    return wrapper_timer


# _default_level_name = os.getenv('ENGINE_LOGGING_LEVEL', 'INFO')
# _default_level = logging.getLevelName(_default_level_name.upper())

def get_logger(log_file=None):
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: - %(message)s',datefmt='%Y%m%d %H:%M:%S')
    logger = logging.getLogger()
    # logger.setLevel(logging.DEBUG)
    logger.setLevel(logging.INFO)
    del logger.handlers[:]

    if log_file:
        file_handler = logging.FileHandler(log_file, mode='w')
        # file_handler.setLevel(logging.DEBUG)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    # stream_handler.setLevel(logging.DEBUG)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)
    return logger


def cal_flops(model, modals, logger):
    x = [torch.zeros(1, 3, 512, 512) for _ in range(len(modals))]
    # x = [torch.zeros(2, 3, 512, 512) for _ in range(len(modals))] #--- PGSNet
    # x = [torch.zeros(1, 3, 512, 512) for _ in range(len(modals))] # --- for HRFuser
    if torch.distributed.is_initialized():
        if 'HR' in model.module.__class__.__name__:
            x = [torch.zeros(1, 3, 512, 512) for _ in range(len(modals))] # --- for HorNet
    else:
        if 'HR' in model.__class__.__name__:
            x = [torch.zeros(1, 3, 512, 512) for _ in range(len(modals))] # --- for HorNet

    if torch.cuda.is_available:
        x = [xi.cuda() for xi in x]
        model = model.cuda()
    logger.info(flop_count_table(FlopCountAnalysis(model, x)))        

def print_iou(epoch, iou, miou, acc, macc, class_names):
    assert len(iou) == len(class_names)
    assert len(acc) == len(class_names)
    lines = ['\n%-8s\t%-8s\t%-8s' % ('Class', 'IoU', 'Acc')]
    for i in range(len(iou)):
        if class_names is None:
            cls = 'Class %d:' % (i+1)
        else:
            cls = '%d %s' % (i+1, class_names[i])
        lines.append('%-8s\t%.2f\t%.2f' % (cls, iou[i], acc[i]))
    lines.append('== %-8s\t%d\t%-8s\t%.2f\t%-8s\t%.2f' % ('Epoch:', epoch, 'mean_IoU', miou, 'mean_Acc',macc))
    line = "\n".join(lines)
    return line


def nchw_to_nlc(x):
    """Flatten [N, C, H, W] shape tensor to [N, L, C] shape tensor.

    Args:
        x (Tensor): The input tensor of shape [N, C, H, W] before conversion.

    Returns:
        Tensor: The output tensor of shape [N, L, C] after conversion.
    """
    assert len(x.shape) == 4
    return x.flatten(2).transpose(1, 2).contiguous()

def nlc_to_nchw(x, hw_shape):
    """Convert [N, L, C] shape tensor to [N, C, H, W] shape tensor.

    Args:
        x (Tensor): The input tensor of shape [N, L, C] before conversion.
        hw_shape (Sequence[int]): The height and width of output feature map.

    Returns:
        Tensor: The output tensor of shape [N, C, H, W] after conversion.
    """
    H, W = hw_shape
    assert len(x.shape) == 3
    B, L, C = x.shape
    assert L == H * W, 'The seq_len does not match H, W'
    return x.transpose(1, 2).reshape(B, C, H, W).contiguous()

def nlc2nchw2nlc(module, x, hw_shape, contiguous=False, **kwargs):
    """Convert [N, L, C] shape tensor `x` to [N, C, H, W] shape tensor. Use the
    reshaped tensor as the input of `module`, and convert the output of
    `module`, whose shape is.
    [N, C, H, W], to [N, L, C].
    Args:
        module (Callable): A callable object the takes a tensor
            with shape [N, C, H, W] as input.
        x (Tensor): The input tensor of shape [N, L, C].
        hw_shape: (Sequence[int]): The height and width of the
            feature map with shape [N, C, H, W].
        contiguous (Bool): Whether to make the tensor contiguous
            after each shape transform.
    Returns:
        Tensor: The output tensor of shape [N, L, C].
    Example:
        >>> import torch
        >>> import torch.nn as nn
        >>> conv = nn.Conv2d(16, 16, 3, 1, 1)
        >>> feature_map = torch.rand(4, 25, 16)
        >>> output = nlc2nchw2nlc(conv, feature_map, (5, 5))
    """
    H, W = hw_shape
    assert len(x.shape) == 3
    B, L, C = x.shape
    assert L == H * W, 'The seq_len doesn\'t match H, W'
    if not contiguous:
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = module(x, **kwargs)
        x = x.flatten(2).transpose(1, 2)
    else:
        x = x.transpose(1, 2).reshape(B, C, H, W).contiguous()
        x = module(x, **kwargs)
        x = x.flatten(2).transpose(1, 2).contiguous()
    return x
