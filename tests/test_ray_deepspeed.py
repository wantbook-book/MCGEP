import os
import torch
import torch.distributed as dist
import deepspeed
import ray

def setup_torch_distributed(rank, world_size, backend="nccl"):
    """
    使用 DeepSpeed 初始化分布式通信。
    """
    os.environ['MASTER_ADDR'] = 'localhost'  # 主节点地址
    os.environ['MASTER_PORT'] = '62355'     # 主节点端口
    deepspeed.init_distributed(dist_backend=backend)
    print(f"[Rank {rank}] Initialized distributed group with world size {world_size}")

def cleanup_torch_distributed():
    """
    销毁分布式进程组。
    """
    dist.destroy_process_group()

@ray.remote(num_gpus=1)
def distributed_task(rank, world_size, task_id):
    """
    一个分布式任务，用于处理不同的任务。
    """
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(rank)
    setup_torch_distributed(rank, world_size)
    
    # 模拟不同的任务逻辑
    print(f"[Rank {rank}] Processing task {task_id}...")
    if task_id == 0:
        print(f"[Rank {rank}] Task {task_id}: Performing computation...")
    elif task_id == 1:
        print(f"[Rank {rank}] Task {task_id}: Performing data processing...")
    
    # 使用分布式通信
    tensor = torch.zeros(1).to('cuda')
    if rank == 0:
        tensor += 1
        dist.send(tensor, dst=1)
        print(f"[Rank {rank}] Sent tensor: {tensor.item()} to Rank 1")
    elif rank == 1:
        dist.recv(tensor, src=0)
        print(f"[Rank {rank}] Received tensor: {tensor.item()} from Rank 0")
    
    cleanup_torch_distributed()

def main():
    # ray.init()  # 初始化 Ray
    world_size = 2  # 两个分布式进程

    # 启动两个任务，每个任务对应一个进程
    tasks = [
        distributed_task.remote(rank=0, world_size=world_size, task_id=0),
        distributed_task.remote(rank=1, world_size=world_size, task_id=1),
    ]
    
    # 等待任务完成
    ray.get(tasks)

if __name__ == "__main__":
    main()
