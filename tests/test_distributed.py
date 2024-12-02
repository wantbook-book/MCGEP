import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

def setup(rank, world_size):
    """初始化分布式环境"""
    dist.init_process_group(
        backend="nccl",  # GPU 上推荐使用 nccl，CPU 使用 gloo
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )
    torch.cuda.set_device(rank)  # 为每个进程分配特定 GPU

def cleanup():
    """销毁分布式环境"""
    dist.destroy_process_group()

def train(rank, world_size):
    """分布式训练逻辑"""
    print(f"Rank {rank}/{world_size} initializing...")

    # 设置分布式环境
    setup(rank, world_size)


    cleanup()

def main():
    """主函数，使用 torch.multiprocessing 启动分布式训练"""
    world_size = torch.cuda.device_count()  # 获取 GPU 数量
    torch.multiprocessing.spawn(
        train,
        args=(world_size,),
        nprocs=world_size,  # 启动每个 GPU 一个进程
        join=True,
    )

if __name__ == "__main__":
    master_addr = "localhost"
    master_port = "42355"
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    main()
